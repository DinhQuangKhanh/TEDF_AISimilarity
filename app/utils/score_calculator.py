import json
import math
import re
import uuid
from functools import lru_cache

from sqlalchemy.orm import Session

from app.core import config
from app.core.config import SEDO_FALLBACK_TOKENS, WPATH_K
from app.models.similarity import Similarity
from app.models.thesis import Thesis
from app.ontology.sedo import get_sedo
from app.services import capability as _capability
from app.services.semantic_encoder import semantic_similarity
from app.utils.text_cleaner import normalize_key, tokenize

# Structural reads the SEDO layer set chosen in config (default: TaskType — "what the system does").
# The expert ground truth judges by function, not technology; a tech-based structural score is
# anti-correlated with the human label, while TaskType is the function-centric signal the ground
# truth rewards (evaluate_ground_truth.py; ICTA_REVIEW_ANALYSIS §F.3/§F.6). Override via the
# STRUCT_LAYERS env var. Domain reads DomainEntity.
_STRUCT_LAYERS = config.STRUCT_LAYERS
_DOMAIN_LAYERS = {"DomainEntity"}

# MDDM fusion weights (DASSF paper, Eq. 1): default alpha=0.30, beta=0.20, gamma=0.30, delta=0.20.
# Overridable via env (see config.MDDM_WEIGHTS / P3.3).
WEIGHTS = config.MDDM_WEIGHTS

# Structural-duplication rule (paper Sect. 3.5): S_str >= TAU_STR and S_dom < TAU_DOM.
# The paper does not fix these numerically; tune them on a labeled set.
TAU_STR = 0.65
TAU_DOM = 0.40

# A dimension is only quoted as a reason once it carries real signal.
REASON_THRESHOLD = 0.50

# Four-level decision scale (paper Table 3).
_ACTIONS = {
    "Low": "Accept",
    "Moderate": "Warn; committee reviews",
    "High": "Require substantial revision",
    "Critical": "Reject",
}


@lru_cache(maxsize=1)
def _tech_mask_pattern():
    """Regex matching every SEDO TechnicalStack surface term (built once). None if SEDO has none."""
    sedo = get_sedo()
    kws = sorted(
        {normalize_key(k) for node in sedo.nodes.values() if node.get("layer") == "TechnicalStack"
         for k in node.get("keywords", []) if normalize_key(k)},
        key=len, reverse=True,  # longest first so multi-word tech phrases win in the alternation
    )
    if not kws:
        return None
    return re.compile(r"(?<![\w])(" + "|".join(re.escape(k) for k in kws) + r")(?![\w])")


@lru_cache(maxsize=8192)
def mask_technology(text: str | None) -> str:
    """Strip TechnicalStack terms (React, ASP.NET Core, SQL Server …) so the semantic encoder scores
    the *business meaning*, not the shared stack. The expert ground truth ignores technology, and a
    tech-laden text inflates the similarity of two same-stack topics; masking lifts the semantic
    signal's correlation with the human label markedly (ICTA_REVIEW_ANALYSIS §H / 2B)."""
    pattern = _tech_mask_pattern()
    if pattern is None or not text:
        return text or ""
    masked = pattern.sub(" ", f" {normalize_key(text)} ")
    return re.sub(r"\s+", " ", masked).strip()


def _content_text(thesis: Thesis) -> str:
    """Concatenate all five topic fields (paper Sect. 3.4). Robust to thesis-like objects that omit
    some attributes (getattr → None)."""
    fields = ("title", "description", "scope", "objectives", "expected_result")
    return " ".join(filter(None, (getattr(thesis, field, None) for field in fields)))


def _surface_text(thesis: Thesis) -> str:
    """Title-only text (kept for callers/tests that want the paper's §V-E title form)."""
    return (thesis.title or "").strip() or _content_text(thesis)


def _has_body(thesis: Thesis) -> bool:
    """True when the topic carries content beyond its title (description/scope/objectives/expected)."""
    return any(
        (getattr(thesis, field, None) or "").strip()
        for field in ("description", "scope", "objectives", "expected_result")
    )


def _pair_texts(a: Thesis, b: Thesis) -> tuple[str, str]:
    """Text pair for the semantic/lexical dimensions.

    The department wants full-content matching: when BOTH topics carry the extra fields, compare all
    six fields (title + description + objectives + scope + expected result). When either topic is
    title-only (nothing but titleEn), fall back to a fair title-vs-title comparison — as the paper
    describes — instead of matching a short title against a long paragraph.
    """
    if _has_body(a) and _has_body(b):
        return _content_text(a), _content_text(b)
    return _surface_text(a), _surface_text(b)


_SEASON_ORDER = {"spring": 1, "summer": 2, "fall": 3, "autumn": 3, "winter": 0}


def _semester_key(semester: str | None):
    """Sortable key for a "Season Year" string, e.g. 'Summer 2026' → (2026, 2). None if unparseable."""
    parts = (semester or "").strip().split()
    if len(parts) != 2:
        return None
    season = _SEASON_ORDER.get(parts[0].lower())
    if season is None:
        return None
    try:
        return (int(parts[1]), season)
    except ValueError:
        return None


def _recent_semesters(theses: list, k: int) -> set[str]:
    """The k most-recent semester labels present in the corpus (by year then season)."""
    keyed = {t.semester: _semester_key(t.semester) for t in theses if _semester_key(getattr(t, "semester", None))}
    ordered = sorted(keyed, key=lambda s: keyed[s], reverse=True)
    return set(ordered[:k])


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def weighted_jaccard(a: set[str], b: set[str], idf: dict[str, float] | None = None) -> float:
    """TF-IDF weighted Jaccard; idf=None falls back to plain Jaccard."""
    union = a | b
    if not union:
        return 0.0
    if idf is None:
        return len(a & b) / len(union)
    numerator = sum(idf.get(token, 1.0) for token in a & b)
    denominator = sum(idf.get(token, 1.0) for token in union)
    return numerator / denominator if denominator else 0.0


def build_idf(theses: list[Thesis]) -> dict[str, float]:
    """Smoothed IDF over the corpus titles (the lexical dimension is title-level)."""
    total = len(theses)
    document_frequency: dict[str, int] = {}
    for thesis in theses:
        for token in tokenize(_surface_text(thesis)):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: math.log((total + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }


def clamp(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


# Four-level scale as (lower bound, level) — exposed so the trace/UI can quote the exact cut-offs.
# Cut points come from config.LEVEL_CUTS (Moderate, High, Critical), calibrated on the ground truth
# by default (ICTA §H / 3A) and falling back to the paper's 0.40/0.65/0.85.
_CUTS = config.LEVEL_CUTS
LEVEL_THRESHOLDS = ((_CUTS[2], "Critical"), (_CUTS[1], "High"), (_CUTS[0], "Moderate"), (0.0, "Low"))


def level_for(score: float, thresholds=None) -> str:
    """Map a composite to a named level. Uses the deployed ``LEVEL_THRESHOLDS`` (calibrated by
    default, §3A) unless an explicit ``thresholds`` sequence of (lower_bound, level) is given."""
    for lower, level in (thresholds or LEVEL_THRESHOLDS):
        if score >= lower:
            return level
    return "Low"


def action_for(level: str) -> str:
    return _ACTIONS.get(level, "")


def composite_score(semantic: float, lexical: float, structure: float, domain: float,
                    weights: dict | None = None) -> float:
    """MDDM weighted fusion (paper Eq. 1). Uses the deployed ``WEIGHTS`` unless ``weights`` is given."""
    w = weights or WEIGHTS
    return clamp(
        semantic * w["semantic"]
        + lexical * w["lexical"]
        + structure * w["structure"]
        + domain * w["domain"]
    )


def is_structural_duplication(structure_score: float, domain_score: float,
                              overall_score: float | None = None) -> bool:
    """Structural duplication: same core architecture, different business domain (paper §3.5) — but
    only when the pair is a GENUINE duplicate. On the expert ground truth the bare pattern
    (structure ≥ τ_str and domain < τ_dom) fires on 233/300 pairs, 221 of them non-duplicates
    (gold level 0/1): to humans a domain-swap is only *mildly* duplicate (ICTA §F.3). Gating it on
    the calibrated level reaching High/Critical cuts those false alarms 221 → 1 (ICTA §H / 3B).
    ``overall_score`` omitted ⇒ the bare paper pattern (backward compatible)."""
    pattern = structure_score >= TAU_STR and domain_score < TAU_DOM
    if overall_score is None:
        return pattern
    return pattern and level_for(overall_score) in ("High", "Critical")


def _join(*parts) -> str:
    return " ".join(part for part in parts if part)


def _ontology_dimension(sedo, text_a, text_b, layers, measure, fallback_a, fallback_b,
                        concept_weights=None) -> float:
    """SEDO-grounded similarity for one dimension.

    When the ontology recognizes nothing on this dimension, the score drops
    (paper Sect. 5.5); with SEDO_FALLBACK_TOKENS it falls back to token Jaccard.

    ``concept_weights`` (optional corpus-IDF map) down-weights ubiquitous concepts (P3.2).
    """
    concepts_a = {c for c in sedo.recognize(text_a) if sedo.nodes[c]["layer"] in layers}
    concepts_b = {c for c in sedo.recognize(text_b) if sedo.nodes[c]["layer"] in layers}
    similarity = sedo.set_similarity(concepts_a, concepts_b, measure, concept_weights)
    if similarity is None:
        return jaccard(fallback_a, fallback_b) if SEDO_FALLBACK_TOKENS else 0.0
    return similarity


def _thesis_all_text(thesis) -> str:
    """All text a thesis carries (content fields + classification tags) — used to measure how
    often each ontology concept occurs across the corpus (for concept-IDF)."""
    return _join(
        thesis.title, thesis.scope, thesis.description, thesis.objectives, thesis.expected_result,
        *(t.name for t in getattr(thesis, "technologies", [])),
        *(s.name for s in getattr(thesis, "structures", [])),
        *(d.name for d in getattr(thesis, "domains", [])),
    )


def build_concept_idf(theses: list) -> dict[str, float]:
    """Corpus IDF for each SEDO concept: log((N+1)/(df+1))+1. Ubiquitous concepts (React, CRUD)
    approach 1.0; rare ones (IoT, Blockchain) get a much larger weight (P3.2)."""
    sedo = get_sedo()
    total = len(theses)
    document_frequency: dict[str, int] = {}
    for thesis in theses:
        for concept in sedo.recognize(_thesis_all_text(thesis)):
            document_frequency[concept] = document_frequency.get(concept, 0) + 1
    return {
        concept: math.log((total + 1) / (count + 1)) + 1.0
        for concept, count in document_frequency.items()
    }


def _capability_text(thesis) -> str:
    """Functional text a topic carries (the five content fields) for capability extraction."""
    return _join(getattr(thesis, "title", None), getattr(thesis, "scope", None),
                 getattr(thesis, "description", None), getattr(thesis, "objectives", None),
                 getattr(thesis, "expected_result", None))


def build_capability_model(theses: list) -> dict:
    """Corpus capability model: per-capability IDF + a stoplist of ubiquitous platform functions
    (present in > ``config.CAPABILITY_STOP_FRACTION`` of the pool). Backs the function-centric
    structural dimension when ``config.CAPABILITY_STRUCTURAL`` is on. See ``app/services/capability.py``
    and ICTA_REVIEW_ANALYSIS §H."""
    total = len(theses) or 1
    document_frequency: dict[str, int] = {}
    for thesis in theses:
        for tag in _capability.extract(_capability_text(thesis)):
            document_frequency[tag] = document_frequency.get(tag, 0) + 1
    idf = {tag: math.log((total + 1) / (df + 1)) + 1.0 for tag, df in document_frequency.items()}
    stop = {tag for tag, df in document_frequency.items() if df / total > config.CAPABILITY_STOP_FRACTION}
    return {"idf": idf, "stop": stop}


def capability_similarity(text_a: str, text_b: str, model: dict) -> float:
    """IDF-weighted Jaccard of the two topics' core-function tags, after dropping ubiquitous platform
    functions — the signal the expert ground truth judges by (function, not technology)."""
    stop = model.get("stop", set())
    a = set(_capability.extract(text_a)) - stop
    b = set(_capability.extract(text_b)) - stop
    return weighted_jaccard(a, b, model.get("idf"))


def calculate_scores(a: Thesis, b: Thesis, lexical_model=None, concept_idf=None,
                    capability_model=None) -> dict:
    """Score one pair across the four MDDM dimensions.

    ``lexical_model`` is optional. Pass a fitted lexical scorer (``app.services.lexical
    .LexicalScorer``, duck-typed on ``.similarity``) for the paper's TF-IDF cosine; pass an IDF
    ``dict`` (legacy) or ``None`` to fall back to TF-IDF-weighted Jaccard on the surface tokens.

    ``concept_idf`` (optional) is a corpus-IDF map from ``build_concept_idf`` used to down-weight
    ubiquitous ontology concepts in the structural/domain dimensions (P3.2). Ignored when
    CONCEPT_IDF_WEIGHTING is off or None.
    """
    sedo = get_sedo()
    concept_weights = concept_idf if config.CONCEPT_IDF_WEIGHTING else None

    # Semantic and lexical compare the FULL content when both topics have it, else fall back to the
    # title (paper §V-E). semantic = SBERT cosine (meaning), lexical = TF-IDF cosine (Module-1 terms).
    text_a, text_b = _pair_texts(a, b)
    lexical_tokens_a = tokenize(text_a)
    lexical_tokens_b = tokenize(text_b)

    # Structural = tech stack + methodology, read from all content fields (+ tech/structure tags).
    struct_text_a = _join(a.title, a.scope, a.description, a.objectives, a.expected_result,
                          *(t.name for t in a.technologies), *(s.name for s in a.structures))
    struct_text_b = _join(b.title, b.scope, b.description, b.objectives, b.expected_result,
                          *(t.name for t in b.technologies), *(s.name for s in b.structures))
    struct_tokens_a = tokenize(struct_text_a)
    struct_tokens_b = tokenize(struct_text_b)

    # Domain = the business domain, taken from the TITLE + author-provided domain tags. Descriptions
    # are deliberately excluded: they carry operational nouns ("staff", "stock", "inventory", "customers")
    # that falsely match DomainEntity concepts and blur the hotel-vs-pharmacy distinction the paper cares about.
    domain_text_a = _join(a.title, *(d.name for d in a.domains))
    domain_text_b = _join(b.title, *(d.name for d in b.domains))
    domain_tokens_a = tokenize(domain_text_a)
    domain_tokens_b = tokenize(domain_text_b)

    if config.MASK_TECH_IN_SEMANTIC:
        semantic = clamp(semantic_similarity(mask_technology(text_a), mask_technology(text_b)))
    else:
        semantic = clamp(semantic_similarity(text_a, text_b))
    if lexical_model is not None and hasattr(lexical_model, "similarity"):
        lexical = clamp(lexical_model.similarity(text_a, text_b))
    else:
        # Legacy / fallback: an IDF dict (or None) → TF-IDF-weighted Jaccard on surface tokens.
        idf = lexical_model if isinstance(lexical_model, dict) else None
        lexical = clamp(weighted_jaccard(lexical_tokens_a, lexical_tokens_b, idf))
    if capability_model is not None:
        # Function-centric structural: overlap of core capabilities (config.CAPABILITY_STRUCTURAL,
        # ICTA §H) — measures what the systems DO, ignoring ubiquitous platform functions.
        structure = clamp(capability_similarity(_capability_text(a), _capability_text(b), capability_model))
    else:
        structure = clamp(
            _ontology_dimension(
                sedo, struct_text_a, struct_text_b, _STRUCT_LAYERS, sedo.wu_palmer,
                struct_tokens_a, struct_tokens_b, concept_weights
            )
        )
    domain = clamp(
        _ontology_dimension(
            sedo,
            domain_text_a,
            domain_text_b,
            _DOMAIN_LAYERS,
            lambda c1, c2: sedo.wpath(c1, c2, WPATH_K),
            domain_tokens_a,
            domain_tokens_b,
            concept_weights,
        )
    )
    overall = composite_score(semantic, lexical, structure, domain)
    level = level_for(overall)
    structural_dup = is_structural_duplication(structure, domain, overall)

    # Only report a dimension that actually carries signal, so the explanation
    # stays consistent (a structural duplication must not also read "same domain").
    reasons = []
    if structural_dup:
        reasons.append("same tech stack with a different business domain")
    elif domain >= REASON_THRESHOLD:
        reasons.append("same business domain")
    if structure >= REASON_THRESHOLD:
        reasons.append("similar architecture or scope")
    if lexical >= REASON_THRESHOLD:
        reasons.append("shared weighted terms across fields")
    if semantic >= REASON_THRESHOLD:
        reasons.append("similar semantic content")
    return {
        "semantic_score": semantic,
        "lexical_score": lexical,
        "structure_score": structure,
        "domain_score": domain,
        "overall_score": overall,
        "level": level,
        "action": action_for(level),
        "structural_duplication": structural_dup,
        "reason": reasons,
    }


def calculate_similarity_for_new(db: Session, new_ids: list[uuid.UUID]) -> None:
    all_theses = db.query(Thesis).filter(Thesis.is_deleted.is_(False)).all()
    thesis_map = {thesis.thesis_id: thesis for thesis in all_theses}
    new_theses = [thesis_map[thesis_id] for thesis_id in new_ids if thesis_id in thesis_map]

    # Department policy: compare only against the N most-recent semesters (default 2). If no semester
    # is parseable, fall back to the whole corpus so nothing silently drops out.
    recent = _recent_semesters(all_theses, config.SIMILARITY_RECENT_SEMESTER_COUNT)
    comparison_pool = [t for t in all_theses if t.semester in recent] if recent else all_theses
    # Fit corpus-level models over the topics actually involved (pool + the new ones).
    pool_ids = {t.thesis_id for t in comparison_pool}
    corpus_for_models = comparison_pool + [t for t in new_theses if t.thesis_id not in pool_ids]

    # Lazy import avoids a module-level cycle (lexical → score_calculator).
    from app.services.lexical import build_lexical_scorer

    lexical_model = build_lexical_scorer(corpus_for_models)
    concept_idf = build_concept_idf(corpus_for_models)
    capability_model = build_capability_model(corpus_for_models) if config.CAPABILITY_STRUCTURAL else None

    seen: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for new_thesis in new_theses:
        for other in comparison_pool:
            if new_thesis.thesis_id == other.thesis_id:
                continue
            thesis_a_id, thesis_b_id = sorted([new_thesis.thesis_id, other.thesis_id])
            pair = (thesis_a_id, thesis_b_id)
            if pair in seen:
                continue
            seen.add(pair)
            exists = (
                db.query(Similarity)
                .filter(Similarity.thesis_a_id == thesis_a_id, Similarity.thesis_b_id == thesis_b_id)
                .first()
            )
            if exists:
                continue
            scores = calculate_scores(new_thesis, other, lexical_model, concept_idf, capability_model)
            db.add(
                Similarity(
                    thesis_a_id=thesis_a_id,
                    thesis_b_id=thesis_b_id,
                    semantic_score=scores["semantic_score"],
                    lexical_score=scores["lexical_score"],
                    structure_score=scores["structure_score"],
                    domain_score=scores["domain_score"],
                    overall_score=scores["overall_score"],
                    level=scores["level"],
                    reason=json.dumps(scores["reason"]),
                )
            )
    db.flush()
