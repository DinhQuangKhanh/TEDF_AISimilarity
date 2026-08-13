import json
import math
import uuid

from sqlalchemy.orm import Session

from app.core.config import SEDO_FALLBACK_TOKENS, WPATH_K
from app.models.similarity import Similarity
from app.models.thesis import Thesis
from app.ontology.sedo import get_sedo
from app.services.semantic_encoder import semantic_similarity
from app.utils.text_cleaner import tokenize

# Which ontology layers each dimension consults (paper Sect. 3.4).
_STRUCT_LAYERS = {"TechnicalStack", "Methodology", "TaskType"}
_DOMAIN_LAYERS = {"DomainEntity"}

# MDDM fusion weights (DASSF paper, Eq. 1): alpha=0.30, beta=0.20, gamma=0.30, delta=0.20.
WEIGHTS = {
    "semantic": 0.30,
    "lexical": 0.20,
    "structure": 0.30,
    "domain": 0.20,
}

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


def _content_text(thesis: Thesis) -> str:
    """Concatenate all five topic fields (paper Sect. 3.4)."""
    return " ".join(
        filter(
            None,
            [thesis.title, thesis.description, thesis.scope, thesis.objectives, thesis.expected_result],
        )
    )


def _surface_text(thesis: Thesis) -> str:
    """Text for the semantic/lexical dimensions: the TITLE (paper §V-E — both operate on the
    title's surface form). Falls back to the full content only when a title is missing, so we
    do not compare on boilerplate-heavy descriptions that inflate token overlap."""
    return (thesis.title or "").strip() or _content_text(thesis)


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


def level_for(score: float) -> str:
    if score >= 0.85:
        return "Critical"
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Moderate"
    return "Low"


def action_for(level: str) -> str:
    return _ACTIONS.get(level, "")


def composite_score(semantic: float, lexical: float, structure: float, domain: float) -> float:
    """MDDM weighted fusion (paper Eq. 1)."""
    return clamp(
        semantic * WEIGHTS["semantic"]
        + lexical * WEIGHTS["lexical"]
        + structure * WEIGHTS["structure"]
        + domain * WEIGHTS["domain"]
    )


def is_structural_duplication(structure_score: float, domain_score: float) -> bool:
    """Same tech stack, different business domain (paper Sect. 3.5)."""
    return structure_score >= TAU_STR and domain_score < TAU_DOM


def _join(*parts) -> str:
    return " ".join(part for part in parts if part)


def _ontology_dimension(sedo, text_a, text_b, layers, measure, fallback_a, fallback_b) -> float:
    """SEDO-grounded similarity for one dimension.

    When the ontology recognizes nothing on this dimension, the score drops
    (paper Sect. 5.5); with SEDO_FALLBACK_TOKENS it falls back to token Jaccard.
    """
    concepts_a = {c for c in sedo.recognize(text_a) if sedo.nodes[c]["layer"] in layers}
    concepts_b = {c for c in sedo.recognize(text_b) if sedo.nodes[c]["layer"] in layers}
    similarity = sedo.set_similarity(concepts_a, concepts_b, measure)
    if similarity is None:
        return jaccard(fallback_a, fallback_b) if SEDO_FALLBACK_TOKENS else 0.0
    return similarity


def calculate_scores(a: Thesis, b: Thesis, lexical_model=None) -> dict:
    """Score one pair across the four MDDM dimensions.

    ``lexical_model`` is optional. Pass a fitted lexical scorer (``app.services.lexical
    .LexicalScorer``, duck-typed on ``.similarity``) for the paper's TF-IDF cosine; pass an IDF
    ``dict`` (legacy) or ``None`` to fall back to TF-IDF-weighted Jaccard on the surface tokens.
    """
    sedo = get_sedo()

    # Semantic and lexical both read the title (paper §V-E); they differ in method:
    # semantic = SBERT cosine (meaning), lexical = TF-IDF cosine over Module-1 folded terms.
    surface_a = _surface_text(a)
    surface_b = _surface_text(b)
    lexical_tokens_a = tokenize(surface_a)
    lexical_tokens_b = tokenize(surface_b)

    # Structural = tech stack + methodology, read from title + scope + description (+ tech/structure tags).
    # The title is included so title-only topics (e.g. a freshly submitted proposal) still map into SEDO.
    struct_text_a = _join(a.title, a.scope, a.description, *(t.name for t in a.technologies), *(s.name for s in a.structures))
    struct_text_b = _join(b.title, b.scope, b.description, *(t.name for t in b.technologies), *(s.name for s in b.structures))
    struct_tokens_a = tokenize(struct_text_a)
    struct_tokens_b = tokenize(struct_text_b)

    # Domain = the business domain, taken from the TITLE + author-provided domain tags. Descriptions
    # are deliberately excluded: they carry operational nouns ("staff", "stock", "inventory", "customers")
    # that falsely match DomainEntity concepts and blur the hotel-vs-pharmacy distinction the paper cares about.
    domain_text_a = _join(a.title, *(d.name for d in a.domains))
    domain_text_b = _join(b.title, *(d.name for d in b.domains))
    domain_tokens_a = tokenize(domain_text_a)
    domain_tokens_b = tokenize(domain_text_b)

    semantic = clamp(semantic_similarity(surface_a, surface_b))
    if lexical_model is not None and hasattr(lexical_model, "similarity"):
        lexical = clamp(lexical_model.similarity(surface_a, surface_b))
    else:
        # Legacy / fallback: an IDF dict (or None) → TF-IDF-weighted Jaccard on surface tokens.
        idf = lexical_model if isinstance(lexical_model, dict) else None
        lexical = clamp(weighted_jaccard(lexical_tokens_a, lexical_tokens_b, idf))
    structure = clamp(
        _ontology_dimension(
            sedo, struct_text_a, struct_text_b, _STRUCT_LAYERS, sedo.wu_palmer, struct_tokens_a, struct_tokens_b
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
        )
    )
    overall = composite_score(semantic, lexical, structure, domain)
    level = level_for(overall)
    structural_dup = is_structural_duplication(structure, domain)

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

    # The lexical model (TF-IDF cosine) is corpus-level, so fit it once per run.
    # Lazy import avoids a module-level cycle (lexical → score_calculator).
    from app.services.lexical import build_lexical_scorer

    lexical_model = build_lexical_scorer(all_theses)

    seen: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for new_thesis in new_theses:
        for other in all_theses:
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
            scores = calculate_scores(new_thesis, other, lexical_model)
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
