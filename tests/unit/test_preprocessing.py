"""Tests for Module 1 — title preprocessing (app/services/preprocessing.py)."""

from app.services.preprocessing import (
    Preprocessed,
    _singularize,
    analyze,
    concept_names,
    preprocess,
)


# ── empty / trivial input ───────────────────────────────────────────────────────────
def test_empty_input_returns_empty():
    for value in (None, "", "   "):
        r = preprocess(value)
        assert r.tokens == set()
        assert r.tech == r.methods == r.domains == r.tasks == set()
        assert r.bag == []


# ── stop-word removal (EN + VI) ─────────────────────────────────────────────────────
def test_english_stopwords_removed():
    tokens = preprocess("Building a system using React").tokens
    for stop in ("building", "a", "system", "using"):
        assert stop not in tokens
    assert "react" in tokens


def test_vietnamese_stopwords_removed():
    tokens = preprocess("Xây dựng hệ thống quản lý khách sạn").tokens
    for stop in ("xây", "dựng", "hệ", "thống"):
        assert stop not in tokens
    # content words survive (normalize_key preserves Vietnamese diacritics)
    assert "khách" in tokens
    assert "sạn" in tokens


# ── lemmatization (conservative singularizer) ───────────────────────────────────────
def test_singularizer_folds_plurals():
    assert _singularize("systems") == "system"
    assert _singularize("services") == "service"
    assert _singularize("libraries") == "library"
    assert _singularize("apps") == "app"


def test_singularizer_leaves_tricky_words():
    for word in ("analysis", "business", "analytics", "news", "various"):
        assert _singularize(word) == word


def test_singularizer_short_words_untouched():
    assert _singularize("ios") == "ios"
    assert _singularize("api") == "api"


# ── synonym folding (surface + SEDO concept) ────────────────────────────────────────
def test_synonym_folding_collapses_react_variants():
    a = preprocess("ReactJS app")
    b = preprocess("React.js application")
    assert "react" in a.tech and "react" in b.tech
    assert "react" in a.tokens and "react" in b.tokens


# ── NER: concept extraction grouped by layer ────────────────────────────────────────
def test_concept_extraction_by_layer_english():
    r = preprocess("Hotel booking platform with Spring Boot using Scrum")
    assert "spring_boot" in r.tech
    assert "scrum" in r.methods
    assert "hotel" in r.domains
    assert "booking" in r.tasks


def test_concept_extraction_pharmacy_and_delivery():
    r = preprocess("pharmacy management system with delivery")
    assert "pharmacy" in r.domains
    assert "crud_management" in r.tasks       # "management system" → CRUD Management
    assert "courier_delivery" in r.domains    # "delivery" → Courier Delivery


def test_domain_and_task_are_separated():
    r = preprocess("movie recommendation system")
    assert "movie" in r.domains
    assert "recommendation" in r.tasks
    assert "movie" not in r.tasks


def test_unknown_technology_is_not_invented():
    # A stack SEDO does not model must not appear as a concept (paper §5.5 coverage gap).
    r = preprocess("system built with QuestPDF and ClosedXML")
    assert r.tech == set()


# ── structured tuple + bag ──────────────────────────────────────────────────────────
def test_structured_tuple_shape():
    tup = preprocess("Hotel management with React").structured_tuple
    assert len(tup) == 4
    assert all(isinstance(part, frozenset) for part in tup)


def test_bag_is_sorted_deduped_union():
    r = preprocess("Hotel management system with React")
    assert r.bag == sorted(set(r.bag))
    assert "hotel" in r.bag and "react" in r.bag and "crud_management" in r.bag


def test_analyze_matches_bag():
    text = "Restaurant booking with Vue.js"
    assert analyze(text) == preprocess(text).bag


# ── concept_names helper ────────────────────────────────────────────────────────────
def test_concept_names_maps_ids_to_display_names():
    assert concept_names({"hotel", "react"}) == ["Hotel", "React"]
    assert concept_names(set()) == []
    assert concept_names({"not-a-real-id"}) == []


def test_preprocessed_is_dataclass_default_empty():
    assert Preprocessed().bag == []
