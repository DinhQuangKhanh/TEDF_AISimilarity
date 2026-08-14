"""Baseline retrieval models for the DASSF experiments (paper §VII-A, Table IV).

These are the *comparison* methods the paper benchmarks the full framework against:
  B1 — TF-IDF Cosine  : plain word-level TF-IDF, cosine similarity (Salton [11], Sparck Jones [12]).
  B2 — BM25           : Okapi BM25 probabilistic retrieval (Robertson & Zaragoza [10]).
(B3 SBERT and B4 DASSF-without-ontology reuse the main pipeline; see benchmark.py.)

Both are deliberately *plain* — a whitespace tokenizer with no ontology folding, so they act as
honest lower bounds that isolate the value added by SEDO and the multi-dimensional fusion.
"""

from __future__ import annotations

import math
from collections import Counter

from app.utils.text_cleaner import normalize_key


def basic_tokens(text: str | None) -> list[str]:
    """Plain tokenizer for the baselines: lowercase, split, keep tokens longer than one char.
    Returns a list (keeps term frequency), unlike the set-based ``tokenize``."""
    if not text:
        return []
    return [t for t in normalize_key(text).split() if len(t) > 1]


class TfidfCosine:
    """B1 — plain TF-IDF cosine. Fits scikit-learn's vectorizer on raw corpus titles."""

    def __init__(self, corpus_texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(analyzer=basic_tokens, norm="l2", sublinear_tf=True)
        self._vec.fit(corpus_texts or [""])
        self._cache: dict[str, object] = {}

    def _vector(self, text: str):
        vector = self._cache.get(text)
        if vector is None:
            vector = self._vec.transform([text or ""])
            self._cache[text] = vector
        return vector

    def similarity(self, text_a: str | None, text_b: str | None) -> float:
        a, b = (text_a or "").strip(), (text_b or "").strip()
        if not a or not b:
            return 0.0
        return max(0.0, min(1.0, float(self._vector(a).multiply(self._vector(b)).sum())))


class BM25Index:
    """B2 — Okapi BM25. Fit on the corpus; ``score(query, doc_index)`` ranks a candidate doc."""

    def __init__(self, corpus_texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [basic_tokens(t) for t in corpus_texts]
        self.n_docs = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        document_frequency: Counter = Counter()
        for doc in self.docs:
            document_frequency.update(set(doc))
        # BM25 idf with the +1 smoothing that keeps it non-negative.
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in document_frequency.items()
        }

    def score(self, query_text: str | None, doc_index: int) -> float:
        if not self.avgdl:
            return 0.0
        freqs = self.freqs[doc_index]
        doc_len = self.doc_len[doc_index]
        total = 0.0
        for term in basic_tokens(query_text):
            tf = freqs.get(term)
            if not tf:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += idf * (tf * (self.k1 + 1)) / denominator
        return total
