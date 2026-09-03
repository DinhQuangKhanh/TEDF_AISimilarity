"""Field-aligned overlap highlighting for the reviewer's side-by-side view (P5.1).

For a pair (topic under review = ``query``, matched topic = ``match``) it returns, PER CONTENT
FIELD, the strongest overlap between the two topics and the exact spans to paint on each side —
so the UI can show "A's description overlaps B's description HERE, from THIS angle" instead of a
scatter of generic tokens sprayed across every field.

Response shape::

    {"fields": [
        {"field": "description", "angle": "semantic", "score": 0.82,
         "a": [{"text": "...manage boat docks, schedules, tours...", "angle": "semantic"},
               {"text": "waterway", "angle": "lexical"}],
         "b": [{"text": "...manage users, services, appointments...", "angle": "semantic"},
               {"text": "waterway", "angle": "lexical"}]},
        {"field": "objectives", "angle": "structural", "score": 0.71,
         "a": [{"text": "real time tracking", "angle": "structural"}],
         "b": [{"text": "gps tracking", "angle": "structural"}]}
    ]}

The ``technologies`` field is never present: the structural dimension reads core functions, not the
tech stack, so React/.NET are never painted as "structural" (see ``_TEXT_FIELDS`` below).

Angles: ``semantic`` (best SBERT sentence pair WITHIN the field) · ``lexical`` (distinctive shared
terms, ranked by corpus IDF) · ``structural`` (shared distinctive CORE FUNCTIONS — capability, e.g.
booking/matching/ocr — **not** technology) · ``domain`` (shared SEDO business-domain concepts).
``field`` uses the frontend FieldKey so the UI needs no translation. A field with no meaningful
overlap is omitted (the UI leaves it plain).

Kept in sync with scoring: the structural DIMENSION reads core functions (capability), not the tech
stack, so the highlight paints functions too — never React/.NET as "structural" (ICTA §H).
"""

from __future__ import annotations

import re

from app.core import config
from app.ontology.sedo import get_sedo
from app.services import capability
from app.services.preprocessing import preprocess
from app.services.semantic_encoder import semantic_similarity
from app.utils.text_cleaner import normalize_key

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|;\s*")
_SENTENCE_MIN = 25            # ignore fragments shorter than this (except title)
_SENTENCE_CAP = 8            # cap sentences per field to keep the SBERT grid small
_SEM_THRESHOLD = 0.62        # a field's best sentence pair must be at least this similar

# The five text fields, each mapped to the frontend FieldKey so the UI needs no translation.
_TEXT_FIELDS = (
    ("title", "title"),
    ("description", "description"),
    ("objectives", "objectives"),
    ("scope", "scope"),
    ("expected_result", "expectedResults"),
)

# Highlight-only stop list: grammar words + nouns almost every SE topic shares. Deliberately NOT
# added to Module-1 preprocessing (that would shift the TF-IDF vectors and force a re-tune) — this
# only prunes what we PAINT as "distinctive shared terms".
_HIGHLIGHT_STOPWORDS = {
    # grammar / filler
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "such", "which", "who", "whom", "can", "will", "would", "should", "may", "might",
    "allow", "allows", "enable", "enables", "provide", "provides", "provided", "various", "etc",
    "also", "through", "between", "within", "their", "them", "they", "other", "others", "more",
    "most", "each", "both", "any", "all", "not", "but", "only", "then", "than", "from", "have",
    "has", "had", "where", "when", "while", "into", "onto", "about",
    # generic SE nouns shared by nearly every topic
    "user", "users", "system", "systems", "service", "services", "management", "manage", "managing",
    "information", "info", "data", "feature", "features", "role", "roles", "access", "profile",
    "profiles", "function", "functions", "functional", "platform", "project", "support", "supporting",
    "process", "processes", "web", "website", "application", "applications", "app", "apps", "software",
    "solution", "solutions", "module", "modules", "interface", "dashboard", "database", "based",
    # boilerplate verbs / plumbing that describe almost any CRUD app
    "expected", "help", "track", "tracking", "create", "add", "update", "view", "list", "register",
    "login", "search", "keep", "record", "records", "activity", "activities", "including", "include",
}

_MIN_TERM_LEN = 3
_TERMS_PER_FIELD = 4
_MIN_IDF = 1.2               # drop corpus-wide terms (low IDF) — keep only distinctive ones


def _field_text(topic, attr: str) -> str:
    return (getattr(topic, attr, None) or "").strip()


def _sentences(text: str, *, allow_short: bool) -> list[str]:
    """Sentences within one field's text (so a highlighted passage never crosses a field boundary)."""
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(text):
        piece = part.strip()
        if piece and (allow_short or len(piece) >= _SENTENCE_MIN):
            out.append(piece)
    if not out and text.strip():
        out.append(text.strip())
    return out[:_SENTENCE_CAP]


def _best_sentence_pair(a_text: str, b_text: str, *, allow_short: bool):
    """The single most-similar (A-sentence, B-sentence) pair in this field, or None below threshold."""
    a_sents = _sentences(a_text, allow_short=allow_short)
    b_sents = _sentences(b_text, allow_short=allow_short)
    if not a_sents or not b_sents:
        return None
    best = None
    for qa in a_sents:
        for mb in b_sents:
            score = semantic_similarity(qa, mb)
            if best is None or score > best[2]:
                best = (qa, mb, score)
    return best if best and best[2] >= _SEM_THRESHOLD else None


def _distinctive_terms(a_text: str, b_text: str, lexical_model, top_k: int = _TERMS_PER_FIELD) -> list[str]:
    """Shared Module-1 terms, minus grammar/generic words, ranked by corpus IDF (most distinctive first)."""
    shared = preprocess(a_text).tokens & preprocess(b_text).tokens
    candidates = [t for t in shared if len(t) >= _MIN_TERM_LEN and t not in _HIGHLIGHT_STOPWORDS]
    if not candidates:
        return []
    if lexical_model is not None:
        # Keep only terms distinctive enough in the corpus (high IDF); rank the rest by IDF.
        candidates = [t for t in candidates if lexical_model.idf(t) >= _MIN_IDF]
        candidates.sort(key=lambda t: (lexical_model.idf(t), t), reverse=True)
    else:
        candidates.sort()
    return candidates[:top_k]


def _sedo_surfaces(a_text: str, b_text: str, layers) -> dict:
    """Surface forms of the SEDO concepts (in ``layers``) shared by both texts, present on each side."""
    sedo = get_sedo()
    shared = sedo.recognize(a_text) & sedo.recognize(b_text)
    a_norm = f" {normalize_key(a_text)} "
    b_norm = f" {normalize_key(b_text)} "
    out = {"a": [], "b": []}
    for concept in shared:
        if sedo.nodes[concept]["layer"] not in layers:
            continue
        for keyword in sedo.nodes[concept]["keywords"]:
            key = normalize_key(keyword)
            if not key:
                continue
            if f" {key} " in a_norm and keyword not in out["a"]:
                out["a"].append(keyword)
            if f" {key} " in b_norm and keyword not in out["b"]:
                out["b"].append(keyword)
    return out


def _shared_domain_surfaces(a_text: str, b_text: str) -> dict:
    """Shared business-DOMAIN surfaces (SEDO DomainEntity) — the domain angle."""
    return _sedo_surfaces(a_text, b_text, {"DomainEntity"})


def _shared_structural_surfaces(a_text: str, b_text: str, stop=None) -> dict:
    """Structural evidence = shared DISTINCTIVE core functions (capability) — NOT technology. ``stop``
    is the corpus stoplist so highlights match the score per pool. Falls back to SEDO structural
    layers when the capability dimension is off, matching how the score runs."""
    if config.CAPABILITY_STRUCTURAL:
        shared = capability.distinctive(a_text, stop) & capability.distinctive(b_text, stop)
        return {"a": capability.matched_surfaces(a_text, shared),
                "b": capability.matched_surfaces(b_text, shared)}
    return _sedo_surfaces(a_text, b_text, config.STRUCT_LAYERS)


def _dedupe(spans: list[dict]) -> list[dict]:
    """Drop duplicate span texts on one side, keeping the first (highest-priority angle added first)."""
    seen: set[str] = set()
    out: list[dict] = []
    for span in spans:
        key = span["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def _text_field_alignment(field_key: str, a_text: str, b_text: str, lexical_model, *, allow_short: bool, stop=None):
    """Assemble one text field's overlap: semantic passage + concepts + distinctive terms."""
    if not a_text or not b_text:
        return None

    a_spans: list[dict] = []
    b_spans: list[dict] = []
    angle = None
    score = None

    # 1) semantic passage pair (added first → wins the colour when a term sits inside it)
    pair = _best_sentence_pair(a_text, b_text, allow_short=allow_short)
    if pair:
        a_spans.append({"text": pair[0], "angle": "semantic"})
        b_spans.append({"text": pair[1], "angle": "semantic"})
        angle, score = "semantic", round(pair[2], 4)

    # 2) shared concepts: domain (SEDO) + structural (shared distinctive core FUNCTIONS, not tech)
    domain = _shared_domain_surfaces(a_text, b_text)
    struct = _shared_structural_surfaces(a_text, b_text, stop)
    for surf in domain["a"]:
        a_spans.append({"text": surf, "angle": "domain"})
    for surf in domain["b"]:
        b_spans.append({"text": surf, "angle": "domain"})
    for surf in struct["a"]:
        a_spans.append({"text": surf, "angle": "structural"})
    for surf in struct["b"]:
        b_spans.append({"text": surf, "angle": "structural"})

    # 3) distinctive shared terms
    for term in _distinctive_terms(a_text, b_text, lexical_model):
        a_spans.append({"text": term, "angle": "lexical"})
        b_spans.append({"text": term, "angle": "lexical"})

    a_spans, b_spans = _dedupe(a_spans), _dedupe(b_spans)
    if not a_spans or not b_spans:
        return None

    if angle is None:
        if domain["a"] and domain["b"]:
            angle = "domain"
        elif struct["a"] and struct["b"]:
            angle = "structural"
        else:
            angle = "lexical"

    return {"field": field_key, "angle": angle, "score": score, "a": a_spans, "b": b_spans}


def compute_highlights(query, match, lexical_model=None, capability_model=None) -> dict:
    """Field-aligned highlight map for one (query, match) pair; fields with no overlap are omitted.

    The ``technologies`` field is deliberately NOT highlighted: the structural dimension no longer
    scores on the tech stack, so painting React/.NET as "structural" would misrepresent the score.
    ``capability_model`` (from ``build_capability_model``) makes the structural spans use the SAME
    corpus stoplist as the score; omit it and a static platform-function fallback is used.
    """
    stop = capability_model.get("stop") if capability_model else None
    fields: list[dict] = []
    for attr, field_key in _TEXT_FIELDS:
        alignment = _text_field_alignment(
            field_key, _field_text(query, attr), _field_text(match, attr),
            lexical_model, allow_short=(attr == "title"), stop=stop,
        )
        if alignment:
            fields.append(alignment)
    return {"fields": fields}
