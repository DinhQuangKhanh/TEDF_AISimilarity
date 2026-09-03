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

# ── Structural dimension: which SEDO layers it reads ────────────────────────────────
# The paper defines S_structural over Technical Stack ∪ Methodology. But the expert ground truth
# (Ground_Truth.xlsx) labels duplication by BUSINESS FUNCTION and its rules FORBID using raw
# technology as the main basis ("cùng React + .NET nhưng khác nghiệp vụ vẫn là 0"). Measured on the
# 300 expert pairs (evaluate_ground_truth.py; ICTA_REVIEW_ANALYSIS §F.3/§F.6): a tech-based
# structural score is ANTI-correlated with the human label, whereas reading the TaskType layer
# ("what the system does") is the only structural definition the tuner keeps a positive weight for
# and the best for AI↔human agreement (CV-QWK 0.42 → 0.46). So structural reads **TaskType** by
# default — not Technical Stack. Override with a comma list, e.g.
# STRUCT_LAYERS="TechnicalStack,Methodology" to restore the old, paper/tech-centric behavior.
_VALID_SEDO_LAYERS = {"TechnicalStack", "Methodology", "TaskType", "DomainEntity"}


def _resolve_struct_layers() -> set[str]:
    raw = os.getenv("STRUCT_LAYERS", "TaskType")
    chosen = {p.strip() for p in raw.split(",") if p.strip() in _VALID_SEDO_LAYERS}
    return chosen or {"TaskType"}


STRUCT_LAYERS = _resolve_struct_layers()

# ── Capability structural dimension (ICTA §H / 1A) ──────────────────────────────────
# When true, the structural dimension is computed from CORE-FUNCTION overlap (booking, matching,
# payment, ocr, …) via app/services/capability.py instead of the coarse SEDO TaskType layer — the
# signal the expert ground truth actually judges by. Measured on the 300 expert pairs it lifts
# AI↔human CV-QWK 0.456 → 0.484 (HO 0.515 → 0.530), but the gain is within CV noise and the
# deterministic keyword extractor caps well below the human-tag ceiling (0.512), so it is
# **off by default** pending a higher-precision extractor. Set CAPABILITY_STRUCTURAL=true to enable.
CAPABILITY_STRUCTURAL = os.getenv("CAPABILITY_STRUCTURAL", "false").lower() == "true"
# Capabilities occurring in more than this fraction of the comparison pool are treated as ubiquitous
# "platform functions" (login/payment/notification/reporting) and dropped from the overlap — exactly
# the ground-truth rule "chức năng nền tảng dùng chung chỉ là bằng chứng phụ".
CAPABILITY_STOP_FRACTION = float(os.getenv("CAPABILITY_STOP_FRACTION", "0.2"))

# ── Mask technology before the semantic encode (ICTA §H / 2B) ───────────────────────
# The expert ground truth ignores technology; a text like "…using React, ASP.NET Core, SQL Server"
# inflates the semantic similarity of two same-stack topics. Stripping TechnicalStack terms before
# encoding lifts the semantic signal's correlation with the human label (Spearman 0.29 → 0.40) and
# raises AI↔human CV-QWK 0.456 → 0.484 with lower variance. On by default; set false to disable.
MASK_TECH_IN_SEMANTIC = os.getenv("MASK_TECH_IN_SEMANTIC", "true").lower() == "true"

# ── Four-level classification cut points (ICTA §H / 3A) ─────────────────────────────
# The paper's fixed cuts (0.40/0.65/0.85, Table 3) are badly miscalibrated to the deployed
# composite (which rarely exceeds 0.85): they give QWK 0.406 on the 300 expert pairs. Cut points
# CALIBRATED to maximize QWK on the ground truth (data/level_thresholds.json) give QWK 0.487
# (CV 0.406 → 0.484). Ascending lower-bounds (Moderate, High, Critical). Resolution:
#   1) env LEVEL_CUTS="0.40,0.55,0.65"; 2) data/level_thresholds.json when USE_CALIBRATED_LEVELS
#   (default true); 3) the paper's 0.40/0.65/0.85.
_PAPER_LEVEL_CUTS = (0.40, 0.65, 0.85)


def _load_level_cuts() -> tuple:
    env = os.getenv("LEVEL_CUTS")
    if env:
        try:
            parts = tuple(float(x) for x in env.split(","))
            if len(parts) == 3 and parts[0] <= parts[1] <= parts[2]:
                return parts
        except ValueError:
            pass
    if os.getenv("USE_CALIBRATED_LEVELS", "true").lower() == "true":
        path = os.path.join(_SRC_DIR, "data", "level_thresholds.json")
        try:
            with open(path, encoding="utf-8") as handle:
                cuts = tuple(float(x) for x in json.load(handle)["thresholds"])
            if len(cuts) == 3 and cuts[0] <= cuts[1] <= cuts[2]:
                return cuts
        except Exception:  # noqa: BLE001 - missing/invalid file ⇒ fall back to paper
            pass
    return _PAPER_LEVEL_CUTS


LEVEL_CUTS = _load_level_cuts()  # (Moderate, High, Critical) ascending lower-bounds

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


def _mddm_weights() -> tuple[dict, str]:
    """Return (weights, source) — ``source`` names which of the three rules above won, so the
    pipeline trace can say *where* the deployed weights came from."""
    if any(os.getenv("MDDM_" + k.upper()) for k in _WEIGHT_KEYS):
        raw = {k: float(os.getenv("MDDM_" + k.upper(), str(_PAPER_WEIGHTS[k]))) for k in _WEIGHT_KEYS}
        source = "env (MDDM_SEMANTIC/LEXICAL/STRUCTURE/DOMAIN)"
    elif os.getenv("MDDM_USE_TUNED", "false").lower() == "true":
        tuned = _load_tuned_weights()
        raw = tuned or dict(_PAPER_WEIGHTS)
        source = ("grid search — data/tuned_weights.json" if tuned
                  else "paper default (data/tuned_weights.json missing)")
    else:
        raw = dict(_PAPER_WEIGHTS)
        source = "paper default (α=.30, β=.20, γ=.30, δ=.20)"
    total = sum(raw.values())
    if total > 0 and abs(total - 1.0) > 1e-9:
        raw = {k: v / total for k, v in raw.items()}
    return raw, source


MDDM_WEIGHTS, MDDM_WEIGHTS_SOURCE = _mddm_weights()
