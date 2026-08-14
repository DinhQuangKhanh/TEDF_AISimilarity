"""Tests for the lexical dimension — TF-IDF cosine (app/services/lexical.py)."""

import pytest

from app.services.lexical import LexicalScorer, build_idf_from_texts, build_lexical_scorer

_CORPUS = [
    "Hotel management system using React and Node.js",
    "Restaurant booking platform with Vue and Spring Boot",
    "Pharmacy inventory management with Node.js",
    "E-learning course platform with Django and PostgreSQL",
]


@pytest.fixture(scope="module")
def scorer() -> LexicalScorer:
    return LexicalScorer(_CORPUS)


def test_backend_is_tfidf_when_sklearn_present(scorer):
    assert scorer.backend == "tfidf-cosine"


def test_identical_titles_score_one(scorer):
    assert scorer.similarity(_CORPUS[0], _CORPUS[0]) == pytest.approx(1.0, abs=1e-6)


def test_disjoint_titles_score_zero(scorer):
    assert scorer.similarity("Hotel management React", "Django PostgreSQL course") == pytest.approx(0.0, abs=1e-6)


def test_empty_inputs_score_zero(scorer):
    assert scorer.similarity("", "Hotel") == 0.0
    assert scorer.similarity(None, None) == 0.0


def test_similarity_is_symmetric(scorer):
    a, b = _CORPUS[0], _CORPUS[2]
    assert scorer.similarity(a, b) == pytest.approx(scorer.similarity(b, a))


def test_similarity_in_unit_range(scorer):
    for a in _CORPUS:
        for b in _CORPUS:
            assert 0.0 <= scorer.similarity(a, b) <= 1.0


def test_partial_overlap_between_zero_and_one(scorer):
    # Both are hotel topics but different stack → shares "hotel", not everything.
    s = scorer.similarity("Hotel management using React", "Hotel booking using Vue")
    assert 0.0 < s < 1.0


def test_synonym_folding_boosts_similarity(scorer):
    # reactjs / react.js fold onto the same concept, so these should read as identical stacks.
    assert scorer.similarity("Hotel app with ReactJS", "Hotel application with React.js") == pytest.approx(1.0, abs=1e-6)


def test_build_lexical_scorer_from_objects():
    class T:
        def __init__(self, title):
            self.title = title

    model = build_lexical_scorer([T("Hotel management with React"), T("Pharmacy with Node")])
    assert model.backend == "tfidf-cosine"
    assert model.similarity("Hotel management with React", "Hotel management with React") == pytest.approx(1.0, abs=1e-6)


def test_weighted_jaccard_fallback_path():
    # Force the fallback branch (as if scikit-learn were unavailable) and confirm it still scores.
    fallback = LexicalScorer(_CORPUS)
    fallback._vectorizer = None
    fallback._idf = build_idf_from_texts(_CORPUS)
    value = fallback.similarity("hotel react node", "hotel vue node")
    assert 0.0 < value < 1.0
    assert fallback.similarity("hotel react", "hotel react") == pytest.approx(1.0)
