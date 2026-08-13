"""Sentence-level semantic similarity for the DASSF ``S_semantic`` dimension (paper §V-E).

Primary path: a multilingual Sentence-BERT model encodes each title into a fixed
embedding, and similarity is the cosine of the two embeddings — so "hotel reservation
system" and "booking platform for accommodations" score high even with no shared token.

Fallback path: if ``sentence-transformers`` is not installed (or the model cannot be
loaded), we degrade to token-set Jaccard so the service keeps running. This is clearly
NOT semantic — it is a stop-gap; install the dependency to get real semantics. The active
mode is exposed via :func:`backend_name` so callers/reports can state which path ran.

Install to enable the real encoder:
    pip install sentence-transformers
Model can be overridden with the SBERT_MODEL env var (default: a compact multilingual
model that covers both English and Vietnamese titles).
"""

from __future__ import annotations

import os
from functools import lru_cache

from app.core.logging import logger
from app.utils.text_cleaner import tokenize

_MODEL_NAME = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

_model = None
_load_attempted = False


def _get_model():
    """Lazily load the SBERT model once. Returns None when unavailable (→ fallback)."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer  # heavy import, kept lazy

        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Semantic encoder: loaded SBERT model '%s'", _MODEL_NAME)
    except Exception:  # noqa: BLE001 - any failure (missing dep, no weights) → fallback
        _model = None
        logger.warning(
            "Semantic encoder: sentence-transformers unavailable; falling back to token "
            "Jaccard for S_semantic. Install it to enable real semantics."
        )
    return _model


def model_available() -> bool:
    return _get_model() is not None


def backend_name() -> str:
    return "sbert" if model_available() else "jaccard-fallback"


@lru_cache(maxsize=8192)
def _embedding(text: str):
    """L2-normalized embedding of one text (cached, so a corpus title is encoded once)."""
    model = _get_model()
    vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return vector


def _jaccard_fallback(text_a: str, text_b: str) -> float:
    tokens_a, tokens_b = tokenize(text_a), tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def semantic_similarity(text_a: str | None, text_b: str | None) -> float:
    """Cosine similarity of two titles in [0, 1] (SBERT), or Jaccard when SBERT is absent."""
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0

    model = _get_model()
    if model is None:
        return _jaccard_fallback(a, b)

    # Embeddings are L2-normalized, so the dot product is the cosine. Unrelated titles can
    # land slightly negative; clamp to [0, 1] since a similarity score is non-negative.
    va, vb = _embedding(a), _embedding(b)
    cosine = float(sum(x * y for x, y in zip(va, vb)))
    return max(0.0, min(1.0, cosine))
