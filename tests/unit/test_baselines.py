"""Tests for baseline retrieval models (app/services/baselines.py)."""

import pytest

from app.services.baselines import BM25Index, TfidfCosine, basic_tokens

_CORPUS = [
    "Hotel management system with React and Node.js",
    "Restaurant booking platform with Vue and Spring Boot",
    "Pharmacy inventory management with Node.js",
    "E-learning course platform with Django",
]


# ── basic tokenizer ─────────────────────────────────────────────────────────────────
def test_basic_tokens_lowercases_and_strips_punctuation():
    assert basic_tokens("Hotel, Management!! React") == ["hotel", "management", "react"]


def test_basic_tokens_keeps_frequency():
    assert basic_tokens("react react node") == ["react", "react", "node"]


def test_basic_tokens_empty():
    assert basic_tokens("") == []
    assert basic_tokens(None) == []


# ── TF-IDF cosine (B1) ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def tfidf():
    return TfidfCosine(_CORPUS)


def test_tfidf_identical_is_one(tfidf):
    assert tfidf.similarity(_CORPUS[0], _CORPUS[0]) == pytest.approx(1.0, abs=1e-6)


def test_tfidf_no_shared_terms_is_zero(tfidf):
    assert tfidf.similarity("hotel react", "pharmacy django") == pytest.approx(0.0, abs=1e-6)


def test_tfidf_symmetric_and_in_range(tfidf):
    for a in _CORPUS:
        for b in _CORPUS:
            s = tfidf.similarity(a, b)
            assert 0.0 <= s <= 1.0
            assert s == pytest.approx(tfidf.similarity(b, a))


def test_tfidf_empty_inputs(tfidf):
    assert tfidf.similarity("", "hotel") == 0.0
    assert tfidf.similarity(None, None) == 0.0


# ── BM25 (B2) ───────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bm25():
    return BM25Index(_CORPUS)


def test_bm25_ranks_matching_doc_highest(bm25):
    query = "Hotel management React"
    scores = [bm25.score(query, i) for i in range(len(_CORPUS))]
    assert scores[0] == max(scores)  # doc 0 is the hotel/react topic


def test_bm25_idf_is_non_negative(bm25):
    assert all(v >= 0.0 for v in bm25.idf.values())


def test_bm25_unknown_query_scores_zero(bm25):
    assert bm25.score("quantum blockchain metaverse", 0) == 0.0


def test_bm25_empty_corpus_is_safe():
    empty = BM25Index([])
    assert empty.avgdl == 0.0
    assert empty.score("anything", 0) == 0.0  # guarded by avgdl, no IndexError


def test_bm25_rare_term_outweighs_common_term(bm25):
    # "django" appears in one doc (rare, high idf); "management" in several (common, low idf).
    assert bm25.idf["django"] > bm25.idf["management"]
