"""Module 1 — Title preprocessing & normalization (DASSF paper §V-C).

Turns a raw title into a structured, normalized representation:

    preprocess("Hotel management system using ReactJS and Node.js")
      → tokens  : {"hotel", "management", "react", "node", ...}   (folded, stop-words removed)
        tech    : {"react", "nodejs"}          (SEDO TechnicalStack concepts)
        methods : set()                         (SEDO Methodology concepts)
        domains : {"hotel"}                      (SEDO DomainEntity concepts)
        tasks   : {"crud_management"}            (SEDO TaskType concepts)

Pipeline steps (paper §V-C): tokenization → stop-word removal (EN + VI) → lightweight
lemmatization → NER + synonym folding. NER and folding are grounded in the SEDO ontology:
recognized surface forms (``reactjs``, ``react.js``, ``khách sạn``) collapse onto the canonical
concept id (``react``, ``hotel``), which is the concept-level equivalent of the paper's Word2Vec
synonym folding — deterministic and domain-specific, needing no extra model.

Notes / honest scope:
- "Lemmatization" here is a conservative rule-based singularizer (no heavy NLP dependency).
- Vietnamese "segmentation" is handled by SEDO's multi-word VI keywords (phrase dictionary),
  not a full word segmenter. Both are deliberately dependency-free; swap in spaCy / underthesea
  later without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ontology.sedo import get_sedo
from app.utils.text_cleaner import normalize_key

# Generic filler words that carry almost no discriminative signal in an SE thesis title.
# Kept deliberately conservative — domain/tech/task words (e.g. "management", "booking") are NOT here.
_STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "at", "by", "from", "into",
    "with", "using", "use", "used", "via", "based", "build", "building", "develop", "developing",
    "development", "design", "designing", "implement", "implementation", "support", "supporting",
    "integrated", "integrate", "system", "systems", "platform", "application", "applications",
    "app", "web", "website", "online", "project", "software", "solution", "new", "js",
}
_STOPWORDS_VI = {
    "xây", "dựng", "xây dựng", "hệ", "thống", "hệ thống", "ứng", "dụng", "ứng dụng", "nền", "tảng",
    "nền tảng", "phát", "triển", "phát triển", "sử", "dụng", "sử dụng", "cho", "và", "của", "trên",
    "bằng", "theo", "một", "các", "với", "website", "phần", "mềm", "phần mềm", "giải", "pháp",
}
_STOPWORDS = _STOPWORDS_EN | _STOPWORDS_VI

# Phrase-level folding on the normalized string (runs before tokenization). normalize_key turns
# "React.js" into "react js", so we collapse "<framework> js" back to the canonical single token —
# the concept-level equivalent of Word2Vec synonym folding for the lexical bag.
_PHRASE_SYNONYMS = {
    "react js": "react", "node js": "node", "vue js": "vue", "next js": "next",
    "express js": "express", "nest js": "nest", "asp net": "aspnet",
}
# Single-token surface synonyms (no dot variants).
_SYNONYMS = {
    "reactjs": "react", "reactts": "react",
    "nodejs": "node",
    "nextjs": "next",
    "vuejs": "vue",
    "ecommerce": "e-commerce",
}

_STRUCT_LAYERS = {"TechnicalStack", "Methodology", "TaskType"}


@dataclass
class Preprocessed:
    """Structured output of Module 1 (paper §V-C tuple + normalized tokens)."""

    tokens: set[str] = field(default_factory=set)   # folded surface tokens (stop-words removed)
    tech: set[str] = field(default_factory=set)     # SEDO TechnicalStack concept ids
    methods: set[str] = field(default_factory=set)  # SEDO Methodology concept ids
    domains: set[str] = field(default_factory=set)  # SEDO DomainEntity concept ids
    tasks: set[str] = field(default_factory=set)    # SEDO TaskType concept ids

    @property
    def structured_tuple(self) -> tuple[frozenset, frozenset, frozenset, frozenset]:
        """(TechStack, Method, DomainEntity, TaskType) — the tuple the paper's M1 emits."""
        return frozenset(self.tech), frozenset(self.methods), frozenset(self.domains), frozenset(self.tasks)

    @property
    def bag(self) -> list[str]:
        """Folded bag-of-terms for TF-IDF: surface tokens ∪ canonical concept ids."""
        return sorted(self.tokens | self.tech | self.methods | self.domains | self.tasks)


def _singularize(token: str) -> str:
    """Conservative rule-based lemmatizer: fold obvious English plurals, leave tricky cases alone."""
    if len(token) <= 3:
        return token
    if token.endswith(("is", "us", "ss", "os", "as", "ics", "ews", "ous")):
        return token  # analysis, business, news, analytics, various…
    if token.endswith("ies"):
        return token[:-3] + "y"  # libraries → library
    if token.endswith("s"):
        return token[:-1]  # systems → system, services → service, apps → app
    return token


def _clean_tokens(text: str) -> set[str]:
    """Tokenize → drop stop-words → fold surface synonyms → singularize."""
    key = normalize_key(text)
    if not key:
        return set()
    for phrase, canonical in _PHRASE_SYNONYMS.items():
        key = key.replace(phrase, canonical)
    tokens: set[str] = set()
    for raw in key.split():
        if len(raw) <= 1 or raw in _STOPWORDS:
            continue
        folded = _SYNONYMS.get(raw, raw)
        tokens.add(_singularize(folded))
    return tokens


def preprocess(text: str | None) -> Preprocessed:
    """Run Module 1 on one title/topic text and return the structured representation."""
    if not text or not text.strip():
        return Preprocessed()

    sedo = get_sedo()
    result = Preprocessed(tokens=_clean_tokens(text))
    for concept_id in sedo.recognize(text):
        layer = sedo.nodes[concept_id]["layer"]
        if layer == "TechnicalStack":
            result.tech.add(concept_id)
        elif layer == "Methodology":
            result.methods.add(concept_id)
        elif layer == "DomainEntity":
            result.domains.add(concept_id)
        elif layer == "TaskType":
            result.tasks.add(concept_id)
    return result


def concept_names(concept_ids: set[str]) -> list[str]:
    """Human-readable names for a set of SEDO concept ids (for attaching classification tags)."""
    sedo = get_sedo()
    return sorted(sedo.nodes[c]["name"] for c in concept_ids if c in sedo.nodes)


def analyze(text: str | None) -> list[str]:
    """sklearn-compatible analyzer: the folded bag-of-terms for one document."""
    return preprocess(text).bag
