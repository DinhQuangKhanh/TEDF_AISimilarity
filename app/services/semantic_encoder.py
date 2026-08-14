"""Sentence-level semantic similarity for the DASSF ``S_semantic`` dimension (paper §V-E).

Primary path: a compact **English** MiniLM model runs through **ONNX Runtime** (via ``fastembed``) —
**no PyTorch** — so the whole encoder fits comfortably on a small (2 GB) server. Each text is encoded
into a fixed embedding and similarity is the cosine of the two embeddings, so "hotel reservation
system" and "booking platform for accommodations" score high even with no shared token.

Fallback path: if ``fastembed`` (ONNX) is not installed or the model cannot load, we degrade to
token-set Jaccard so the service keeps running. This is clearly NOT semantic — it is a stop-gap;
install the dependency to get real semantics. The active mode is exposed via :func:`backend_name`.

Why ONNX + an English model (not the multilingual SBERT we started with): duplicate checking runs on
**English content only**, and the multilingual ``sentence-transformers`` stack drags in PyTorch
(~1 GB resident). ``all-MiniLM-L6-v2`` on ONNX Runtime gives equivalent English quality for roughly a
tenth of the memory — the difference between fitting and OOM-ing a 2 GB box.

Install to enable the real encoder:
    pip install fastembed
Model can be overridden with the SBERT_MODEL env var
(default: ``sentence-transformers/all-MiniLM-L6-v2`` — English, 384-dim, ~90 MB ONNX weights).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from app.core.logging import logger
from app.utils.text_cleaner import tokenize

_MODEL_NAME = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None
_load_attempted = False


def _get_model():
    """Lazily load the ONNX embedding model once. Returns None when unavailable (→ fallback)."""
    global _model, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True
    try:
        from fastembed import TextEmbedding  # ONNX Runtime under the hood — no PyTorch

        _model = TextEmbedding(model_name=_MODEL_NAME)
        logger.info("Semantic encoder: loaded ONNX embedding model '%s'", _MODEL_NAME)
    except Exception:  # noqa: BLE001 - any failure (missing dep, no weights) → fallback
        _model = None
        logger.warning(
            "Semantic encoder: fastembed unavailable; falling back to token Jaccard for "
            "S_semantic. Install fastembed to enable real semantics."
        )
    return _model


def model_available() -> bool:
    return _get_model() is not None


def backend_name() -> str:
    return "minilm-onnx" if model_available() else "jaccard-fallback"


@lru_cache(maxsize=8192)
def _embedding(text: str) -> np.ndarray:
    """L2-normalized embedding of one text (cached, so a corpus title is encoded once)."""
    model = _get_model()
    # fastembed .embed() yields one numpy vector per input; take the single one we asked for.
    vector = np.asarray(next(iter(model.embed([text]))), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _jaccard_fallback(text_a: str, text_b: str) -> float:
    tokens_a, tokens_b = tokenize(text_a), tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def semantic_similarity(text_a: str | None, text_b: str | None) -> float:
    """Cosine similarity of two texts in [0, 1] (MiniLM/ONNX), or Jaccard when the model is absent."""
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return 0.0

    model = _get_model()
    if model is None:
        return _jaccard_fallback(a, b)

    # Embeddings are L2-normalized, so the dot product is the cosine. Unrelated texts can land
    # slightly negative; clamp to [0, 1] since a similarity score is non-negative.
    cosine = float(np.dot(_embedding(a), _embedding(b)))
    return max(0.0, min(1.0, cosine))
