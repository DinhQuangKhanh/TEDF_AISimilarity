"""Lexical dimension — TF-IDF cosine over the corpus (DASSF paper §III-A, §V-E).

The paper's ``S_lexical`` is the cosine of two TF-IDF vectors. We fit a scikit-learn
``TfidfVectorizer`` once over the corpus, using Module 1 (``preprocessing.analyze``) as the
analyzer so the same stop-word removal, lemmatization and synonym folding feed the vectors.
Vectors are L2-normalized (the vectorizer default), so cosine similarity is a dot product.

If scikit-learn is unavailable the scorer degrades to the previous TF-IDF-weighted Jaccard
so the pipeline keeps running. ``backend`` reports which path is active.
"""

from __future__ import annotations

from app.core.logging import logger
from app.services.preprocessing import analyze
from app.utils.score_calculator import weighted_jaccard
from app.utils.text_cleaner import tokenize


class LexicalScorer:
    """Corpus-fitted TF-IDF cosine similarity for the lexical dimension."""

    def __init__(self, texts: list[str]):
        self._vectorizer = None
        self._cache: dict[str, object] = {}
        self._idf = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # analyzer=analyze → Module-1 folded bag-of-terms; sublinear_tf dampens repeats.
            self._vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True, norm="l2")
            self._vectorizer.fit(texts or [""])
            self.backend = "tfidf-cosine"
        except Exception:  # noqa: BLE001 - missing sklearn or a fit failure → fall back
            logger.warning("LexicalScorer: scikit-learn unavailable; using TF-IDF-weighted Jaccard.")
            self._idf = build_idf_from_texts(texts)
            self.backend = "weighted-jaccard-fallback"

    def _vector(self, text: str):
        vector = self._cache.get(text)
        if vector is None:
            vector = self._vectorizer.transform([text or ""])
            self._cache[text] = vector
        return vector

    def similarity(self, text_a: str | None, text_b: str | None) -> float:
        a = (text_a or "").strip()
        b = (text_b or "").strip()
        if not a or not b:
            return 0.0
        if self._vectorizer is None:  # fallback path
            return weighted_jaccard(tokenize(a), tokenize(b), self._idf)
        # L2-normalized rows → cosine == dot product.
        cosine = float(self._vector(a).multiply(self._vector(b)).sum())
        return max(0.0, min(1.0, cosine))


def build_idf_from_texts(texts: list[str]) -> dict[str, float]:
    """IDF over already-extracted texts, reusing the token pipeline (fallback helper)."""
    import math

    total = len(texts) or 1
    document_frequency: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in document_frequency.items()}


# Re-exported so callers can build a scorer straight from Thesis-like objects.
def build_lexical_scorer(theses) -> LexicalScorer:
    """Fit a LexicalScorer on the surface text (title) of each thesis-like object."""
    texts = [(getattr(t, "title", None) or "").strip() or _fallback_text(t) for t in theses]
    return LexicalScorer(texts)


def _fallback_text(thesis) -> str:
    parts = [getattr(thesis, f, None) for f in ("title", "description", "scope", "objectives", "expected_result")]
    return " ".join(p for p in parts if p)
