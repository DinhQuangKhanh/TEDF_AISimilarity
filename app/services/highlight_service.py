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
        {"field": "technologies", "angle": "structural", "score": 1.0,
         "a": [{"text": "react", "angle": "structural"}, {"text": "sql server", "angle": "structural"}],
         "b": [{"text": "react", "angle": "structural"}, {"text": "sql server", "angle": "structural"}]}
    ]}

Angles: ``semantic`` (best SBERT sentence pair WITHIN the field) · ``lexical`` (distinctive shared
terms, ranked by corpus IDF) · ``structural`` (shared SEDO tech / methodology / task concepts) ·
``domain`` (shared SEDO business-domain concepts). ``field`` uses the frontend FieldKey so the UI
needs no translation. A field with no meaningful overlap is omitted (the UI leaves it plain).
"""

from __future__ import annotations

import re

from app.ontology.sedo import get_sedo
from app.services.preprocessing import preprocess
from app.services.semantic_encoder import semantic_similarity
from app.utils.score_calculator import jaccard
from app.utils.text_cleaner import normalize_key

_STRUCT_LAYERS = {"TechnicalStack", "Methodology", "TaskType"}
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


def _tech_text(topic) -> str:
    return ", ".join(t.name for t in getattr(topic, "technologies", []) if getattr(t, "name", None))


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


def _shared_concept_surfaces(a_text: str, b_text: str) -> tuple[dict, dict]:
    """(struct, domain) surface forms of the SEDO concepts shared by both texts, present on each side."""
    sedo = get_sedo()
    shared = sedo.recognize(a_text) & sedo.recognize(b_text)
    a_norm = f" {normalize_key(a_text)} "
    b_norm = f" {normalize_key(b_text)} "
    struct = {"a": [], "b": []}
    domain = {"a": [], "b": []}
    for concept in shared:
        node = sedo.nodes[concept]
        if node["layer"] in _STRUCT_LAYERS:
            bucket = struct
        elif node["layer"] == "DomainEntity":
            bucket = domain
        else:
            continue
        for keyword in node["keywords"]:
            key = normalize_key(keyword)
            if not key:
                continue
            if f" {key} " in a_norm and keyword not in bucket["a"]:
                bucket["a"].append(keyword)
            if f" {key} " in b_norm and keyword not in bucket["b"]:
                bucket["b"].append(keyword)
    return struct, domain


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


def _text_field_alignment(field_key: str, a_text: str, b_text: str, lexical_model, *, allow_short: bool):
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

    # 2) shared SEDO concepts (domain then structural)
    struct, domain = _shared_concept_surfaces(a_text, b_text)
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


def _tech_alignment(query, match):
    """Technologies field: shared tech/method/task concepts (structural angle), scored by Jaccard."""
    a_text, b_text = _tech_text(query), _tech_text(match)
    if not a_text or not b_text:
        return None
    struct, _domain = _shared_concept_surfaces(a_text, b_text)
    a_spans = _dedupe([{"text": s, "angle": "structural"} for s in struct["a"]])
    b_spans = _dedupe([{"text": s, "angle": "structural"} for s in struct["b"]])
    if not a_spans or not b_spans:
        return None
    sedo = get_sedo()
    ca = {c for c in sedo.recognize(a_text) if sedo.nodes[c]["layer"] in _STRUCT_LAYERS}
    cb = {c for c in sedo.recognize(b_text) if sedo.nodes[c]["layer"] in _STRUCT_LAYERS}
    return {"field": "technologies", "angle": "structural", "score": round(jaccard(ca, cb), 4),
            "a": a_spans, "b": b_spans}


def compute_highlights(query, match, lexical_model=None) -> dict:
    """Field-aligned highlight map for one (query, match) pair; fields with no overlap are omitted."""
    fields: list[dict] = []
    for attr, field_key in _TEXT_FIELDS:
        alignment = _text_field_alignment(
            field_key, _field_text(query, attr), _field_text(match, attr),
            lexical_model, allow_short=(attr == "title"),
        )
        if alignment:
            fields.append(alignment)
    tech = _tech_alignment(query, match)
    if tech:
        fields.append(tech)
    return {"fields": fields}
