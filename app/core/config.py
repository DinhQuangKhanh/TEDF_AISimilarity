import json
import os

from dotenv import load_dotenv

load_dotenv()

# backend/src, so we can locate data/tuned_weights.json regardless of CWD.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Dedup safety: whether the importer may ask an LLM to INVENT missing content fields
# (description/scope/objectives/expected_result). Default OFF — a duplicate-detection system
# must not compare on fabricated text. When off, missing fields stay empty and the row is
# flagged needs_review. See app/services/llm_normalizer_service.py.
ALLOW_LLM_CONTENT_FILL = os.getenv("ALLOW_LLM_CONTENT_FILL", "false").lower() == "true"

# SEDO / similarity settings.
# Default (False) matches the paper: an unrecognized concept is dropped and the
# structural/domain score falls (paper Sect. 5.5). Set True to fall back to token
# Jaccard when the ontology recognizes nothing on a dimension.
SEDO_FALLBACK_TOKENS = os.getenv("SEDO_FALLBACK_TOKENS", "false").lower() == "true"
# wpath k parameter (Zhu-Iglesias). Tune to match the paper's setup.
WPATH_K = float(os.getenv("WPATH_K", "0.8"))

# ── Structural-dimension tuning (P3) ────────────────────────────────────────────────
# Paper §V-E defines S_structural over Technical Stack ∪ Methodology only. Including TaskType
# (CRUD/Analytics/Tracking/Notification — near-universal in management systems) inflates the
# score. Default False = paper-aligned. Set True to restore the old behavior.
STRUCT_INCLUDE_TASKTYPE = os.getenv("STRUCT_INCLUDE_TASKTYPE", "false").lower() == "true"

# Department policy (P4): when scoring a new topic, compare only against the N most-recent
# semesters present in the corpus (Spring 2026 + Summer 2026 for the current data).
SIMILARITY_RECENT_SEMESTER_COUNT = int(os.getenv("SIMILARITY_RECENT_SEMESTER_COUNT", "2"))

# Weight each shared ontology concept by its corpus rarity (IDF), so ubiquitous concepts
# (React, ASP.NET, CRUD) count far less than rare ones (IoT, Blockchain, Unity). This fixes the
# "everyone uses React ⇒ everything looks structurally duplicated" base-rate problem. Default ON.
CONCEPT_IDF_WEIGHTING = os.getenv("CONCEPT_IDF_WEIGHTING", "true").lower() == "true"

# ── MDDM fusion weights (P3.3) ──────────────────────────────────────────────────────
# Resolution order:
#   1) explicit env (MDDM_SEMANTIC/LEXICAL/STRUCTURE/DOMAIN) — highest priority;
#   2) grid-searched weights from data/tuned_weights.json when MDDM_USE_TUNED=true (learned by
#      tools/tune_weights.py on the teacher benchmark);
#   3) the paper's optimum α=0.30, β=0.20, γ=0.30, δ=0.20.
# The result is renormalized to sum to 1 only if it does not already.
_PAPER_WEIGHTS = {"semantic": 0.30, "lexical": 0.20, "structure": 0.30, "domain": 0.20}
_WEIGHT_KEYS = ("semantic", "lexical", "structure", "domain")


def _load_tuned_weights() -> dict | None:
    path = os.path.join(_SRC_DIR, "data", "tuned_weights.json")
    try:
        with open(path, encoding="utf-8") as handle:
            weights = json.load(handle)["weights"]
        return {k: float(weights[k]) for k in _WEIGHT_KEYS}
    except Exception:  # noqa: BLE001 - missing/invalid file ⇒ fall back to paper
        return None


def _mddm_weights() -> dict:
    if any(os.getenv("MDDM_" + k.upper()) for k in _WEIGHT_KEYS):
        raw = {k: float(os.getenv("MDDM_" + k.upper(), str(_PAPER_WEIGHTS[k]))) for k in _WEIGHT_KEYS}
    elif os.getenv("MDDM_USE_TUNED", "false").lower() == "true":
        raw = _load_tuned_weights() or dict(_PAPER_WEIGHTS)
    else:
        raw = dict(_PAPER_WEIGHTS)
    total = sum(raw.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        raw = {k: v / total for k, v in raw.items()}
    return raw


MDDM_WEIGHTS = _mddm_weights()
