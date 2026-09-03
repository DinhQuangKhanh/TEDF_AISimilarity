"""Evaluate DASSF against the REAL expert ground truth (``Ground_Truth.xlsx``).

Why this file exists (see ``ICTA_REVIEW_ANALYSIS.md``)
=====================================================
The shipped harness ``evaluate.py`` scores the AI on ``data/FA26_topics_deduplicated.json``
— a *different, cruder* ground truth (one teacher, a scalar %, title-only queries). The
rigorous expert benchmark the reviewers demanded already exists in the repo as
``Ground_Truth.xlsx`` (mirrored to ``Ground_Truth_markdown.md``) but **no code ever reads it**.

This script closes that gap. It reads the real ground truth and fixes the four methodology
problems both ICTA reviewers raised:

  * R1.2 / R2.2  — "no results on the expert set". → We score the AI on the **300 expert
                   pairs** (5-field content, joined from the '53 De tai' sheet), not title-only.
  * R2.5         — "Fleiss' κ never reported". → We compute **inter-annotator agreement**
                   from GV1/GV2 (Cohen's κ, unweighted + quadratic-weighted). Note the artifact
                   has **2** annotators ⇒ Cohen's κ, not the paper's claimed 3-annotator Fleiss'.
  * R1.5 / R2.4  — "tune == test, no split, overfit". → Weights and the score→level thresholds
                   are calibrated on a **TRAIN split only**; every headline number is on the
                   held-out TEST split, plus **stratified k-fold CV** (mean ± std) for stability.
  * F.3          — the ground truth ranks a **domain-switch** variant only level 1. → We score
                   the 80 designed variant pairs by **change type** and show where the AI's
                   tech-driven structural signal disagrees with the human label.

Ordinal vs. percentage
----------------------
The human label is an **ordinal category 0-3**, never a percentage. The AI emits a composite
in [0,1]. We do NOT force the ground truth into a %. Instead we keep the human 0-3 labels and
learn a monotone score→{0,1,2,3} mapping on TRAIN, then measure agreement with **Quadratic
Weighted Kappa (QWK)** — the standard metric for an ordinal scale — alongside macro-F1,
adjacent accuracy and the binary duplicate AUC.

Run (no DB, no network beyond the one-time MiniLM weight download):
    python evaluate_ground_truth.py
Writes ``data/ground_truth_eval_report.json`` and prints a digest.
"""

from __future__ import annotations

import json
import math
import os
import re
from itertools import combinations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.core import config
from app.services.lexical import build_lexical_scorer
from app.services.semantic_encoder import backend_name
from app.utils import score_calculator as sc

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SRC_DIR, "data")
_REPORT = os.path.join(_DATA_DIR, "ground_truth_eval_report.json")

# The guide (sheet 'Huong dan') itself used this seed to sample the 300 pairs — reuse it so the
# split is reproducible and traceable to the dataset's own construction.
SEED = 20260824
DIMS = ("semantic", "lexical", "structure", "domain")
PAPER_WEIGHTS = (0.30, 0.20, 0.30, 0.20)  # α, β, γ, δ (paper Eq. 1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ground-truth loading (prefers Ground_Truth.xlsx; falls back to the md export)
# ─────────────────────────────────────────────────────────────────────────────
def _find_xlsx() -> str | None:
    for path in (
        os.path.join(_SRC_DIR, "Ground_Truth.xlsx"),
        os.path.join(_DATA_DIR, "Ground_Truth.xlsx"),
        os.path.expanduser("~/Downloads/Ground_Truth.xlsx"),
    ):
        if os.path.exists(path):
            return path
    return None


def _sheets_from_xlsx(path: str) -> dict[str, list[dict]]:
    import pandas as pd

    book = pd.read_excel(path, sheet_name=None, dtype=str)
    out: dict[str, list[dict]] = {}
    for name, frame in book.items():
        frame = frame.fillna("")
        out[name.strip()] = [
            {str(k).strip(): str(v).strip() for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    return out


def _sheets_from_markdown(path: str) -> dict[str, list[dict]]:
    """Parse the pipe-tables under each '## Sheet: <name>' heading. Verified safe: every data
    row in the labelled sheets has a constant column count (no literal '|' inside cells)."""
    lines = open(path, encoding="utf-8").read().split("\n")
    starts = [(i, m.group(1).strip()) for i, l in enumerate(lines)
              if (m := re.match(r"^## Sheet: (.+)$", l))]
    out: dict[str, list[dict]] = {}
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        table = [l for l in lines[start:end]
                 if l.startswith("|") and not re.match(r"^\|\s*-{2,}", l)]
        if not table:
            out[name] = []
            continue
        header = [c.strip() for c in table[0].strip().strip("|").split("|")]
        rows = []
        for line in table[1:]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            rows.append(dict(zip(header, cells)))
        out[name] = rows
    return out


def load_sheets() -> tuple[dict[str, list[dict]], str]:
    xlsx = _find_xlsx()
    if xlsx:
        try:
            return _sheets_from_xlsx(xlsx), f"xlsx:{os.path.basename(xlsx)}"
        except Exception:  # noqa: BLE001 - openpyxl/pandas hiccup → fall back to the md export
            pass
    md = os.path.join(_SRC_DIR, "Ground_Truth_markdown.md")
    return _sheets_from_markdown(md), "markdown:Ground_Truth_markdown.md"


def _col(row: dict, *candidates: str) -> str:
    """First present column among candidates (tolerant to header wording between xlsx/md)."""
    for c in candidates:
        for key in row:
            if key.lower() == c.lower():
                return row[key]
    # loose contains-match as a last resort
    for c in candidates:
        for key in row:
            if c.lower() in key.lower():
                return row[key]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Domain objects the DASSF scorer understands (duck-typed Thesis)
# ─────────────────────────────────────────────────────────────────────────────
class _Tag:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class EvalThesis:
    """Minimal stand-in for the SQLAlchemy Thesis so ``sc.calculate_scores`` runs without a DB."""

    def __init__(self, title, description="", scope="", objectives="", expected_result="",
                 technologies=()):
        self.title = title or ""
        self.description = description or ""
        self.scope = scope or ""
        self.objectives = objectives or ""
        self.expected_result = expected_result or ""
        self.technologies = [_Tag(t) for t in technologies]
        self.structures = []
        self.domains = []


def _split_tech(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,/;+]| and | & ", text)
    return [p.strip() for p in parts if p.strip()]


def build_topics(sheets) -> dict[str, EvalThesis]:
    """The 53 real topics (T01..T53), each with all five fields + technology tags."""
    topics: dict[str, EvalThesis] = {}
    for row in sheets.get("53 De tai", []):
        tid = _col(row, "ID").strip()
        if not re.match(r"^T\d+$", tid):
            continue
        topics[tid] = EvalThesis(
            title=_col(row, "Title", "Đề tài"),
            description=_col(row, "Description", "Mô tả"),
            objectives=_col(row, "Objective", "Objectives"),
            scope=_col(row, "Scope"),
            expected_result=_col(row, "Expected Result", "ExpectedResult"),
            technologies=_split_tech(_col(row, "Technology", "Stack")),
        )
    return topics


def build_variants(sheets) -> dict[str, dict]:
    """The 32 AI variants (V01-D/N/P/T ...), with content + base id + designed label + change type."""
    variants: dict[str, dict] = {}
    for row in sheets.get("Bien the AI", []):
        vid = _col(row, "Variant ID").strip()
        if not vid:
            continue
        variants[vid] = {
            "base_id": _col(row, "Base ID").strip(),
            "change_type": _col(row, "Loại thay đổi"),
            "design_label": _to_int(_col(row, "Nhãn thiết kế (0-3)", "Nhãn thiết kế")),
            "keep": _col(row, "Giữ/Bỏ"),
            "thesis": EvalThesis(
                title=_col(row, "Đề tài biến thể (AI sinh)", "Đề tài biến thể"),
                description=_col(row, "Mô tả biến thể"),
                objectives=_col(row, "Objective"),
                scope=_col(row, "Scope"),
                technologies=_split_tech(_col(row, "Stack")),
            ),
        }
    return variants


def _to_int(x):
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metrics (self-contained; ordinal 0..3)
# ─────────────────────────────────────────────────────────────────────────────
_K = 4  # categories 0,1,2,3


def _confusion(a, b):
    m = np.zeros((_K, _K), dtype=float)
    for x, y in zip(a, b):
        m[x, y] += 1
    return m


def kappa(a, b, weighted: str | None = None) -> float:
    """Cohen's κ over categories 0..3. weighted=None → unweighted; 'quadratic' → QWK."""
    a = list(a)
    b = list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    obs = _confusion(a, b)
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col) / n
    if weighted == "quadratic":
        w = np.zeros((_K, _K))
        for i in range(_K):
            for j in range(_K):
                w[i, j] = ((i - j) ** 2) / ((_K - 1) ** 2)
        num = (w * obs).sum()
        den = (w * exp).sum()
        return 1.0 - num / den if den else 1.0
    po = np.trace(obs) / n
    pe = np.trace(exp) / n
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def qwk(a, b) -> float:
    return kappa(a, b, weighted="quadratic")


def macro_f1(gold, pred) -> float:
    labels = sorted(set(gold))
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def adjacent_accuracy(gold, pred) -> float:
    n = len(gold)
    return sum(1 for g, p in zip(gold, pred) if abs(g - p) <= 1) / n if n else 0.0


def exact_accuracy(gold, pred) -> float:
    n = len(gold)
    return sum(1 for g, p in zip(gold, pred) if g == p) / n if n else 0.0


def binary_auc(scores, labels) -> float | None:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def binary_prf(gold, pred) -> dict:
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if (not g) and p)
    fn = sum(1 for g, p in zip(gold, pred) if g and (not p))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def _spearman(a, b) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration: weights (TRAIN) + score→ordinal thresholds (TRAIN)
# ─────────────────────────────────────────────────────────────────────────────
def _simplex(step=0.05):
    n = int(round(1 / step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                yield (a / n, b / n, c / n, d / n)


def composite(dims: np.ndarray, weights) -> np.ndarray:
    return dims @ np.asarray(weights, dtype=float)


def tune_weights(dims_train: np.ndarray, gold_train) -> tuple:
    """Pick MDDM weights that best CORRELATE (Spearman, threshold-free) with the ordinal gold on
    TRAIN. Threshold-free keeps weight selection independent of the later score→level cut points."""
    best_w, best_r = PAPER_WEIGHTS, -2.0
    for w in _simplex(0.05):
        r = _spearman(composite(dims_train, w), gold_train)
        if r is not None and r > best_r:
            best_r, best_w = r, w
    return best_w, round(best_r, 4)


def calibrate_thresholds(scores_train, gold_train, grid_step=0.05) -> tuple:
    """Three ordered cut points t1<t2<t3 mapping a composite → {0,1,2,3}, chosen to maximize QWK
    on TRAIN (tie-break macro-F1). Purely monotone, so it never re-orders the model's scores."""
    grid = [round(x, 4) for x in np.arange(grid_step, 1.0, grid_step)]
    best, best_key = (0.40, 0.65, 0.85), (-2.0, -2.0)
    for t in combinations(grid, 3):
        pred = [_apply_thresholds(s, t) for s in scores_train]
        key = (qwk(gold_train, pred), macro_f1(gold_train, pred))
        if key > best_key:
            best_key, best = key, t
    return best


def _apply_thresholds(score: float, thresholds) -> int:
    return int(sum(1 for t in thresholds if score >= t))


def best_binary_threshold(scores_train, labels_train) -> float:
    """Threshold on the composite that maximizes binary-F1 on TRAIN (duplicate = gold ≥ 2)."""
    best_t, best_f1 = 0.5, -1.0
    for t in [round(x, 3) for x in np.arange(0.05, 1.0, 0.02)]:
        pred = [s >= t for s in scores_train]
        f1 = binary_prf(labels_train, pred)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate one train/test split under a weight scheme
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_split(dims, gold_ord, bin_gold, tr, te, weights, tune=False) -> dict:
    dims_tr, dims_te = dims[tr], dims[te]
    gold_tr = [gold_ord[i] for i in tr]
    gold_te = [gold_ord[i] for i in te]
    binte = [bin_gold[i] for i in te]
    bintr = [bin_gold[i] for i in tr]

    used_w, spear_tr = (weights, None)
    if tune:
        used_w, spear_tr = tune_weights(dims_tr, gold_tr)

    s_tr = composite(dims_tr, used_w)
    s_te = composite(dims_te, used_w)

    thresholds = calibrate_thresholds(s_tr, gold_tr)
    pred_te = [_apply_thresholds(s, thresholds) for s in s_te]

    bt = best_binary_threshold(s_tr, bintr)
    bin_pred_te = [s >= bt for s in s_te]

    return {
        "weights": [round(x, 3) for x in used_w],
        "weight_spearman_train": spear_tr,
        "level_thresholds": [round(x, 3) for x in thresholds],
        "ordinal": {
            "qwk": round(qwk(gold_te, pred_te), 4),
            "macro_f1": round(macro_f1(gold_te, pred_te), 4),
            "adjacent_acc": round(adjacent_accuracy(gold_te, pred_te), 4),
            "exact_acc": round(exact_accuracy(gold_te, pred_te), 4),
            "spearman_vs_gold": (lambda r: round(r, 4) if r is not None else None)(
                _spearman(s_te, gold_te)),
        },
        "binary_duplicate": {
            "threshold": bt,
            "auc": (lambda a: round(a, 4) if a is not None else None)(binary_auc(s_te, binte)),
            **binary_prf(binte, bin_pred_te),
            "n_positive_test": int(sum(binte)),
        },
    }


def cross_validate(dims, gold_ord, bin_gold, coarse, weights, tune=False, k=5) -> dict:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    qwks, f1s, aucs = [], [], []
    for tr, te in skf.split(np.zeros(len(gold_ord)), coarse):
        r = evaluate_split(dims, gold_ord, bin_gold, tr, te, weights, tune)
        qwks.append(r["ordinal"]["qwk"])
        f1s.append(r["ordinal"]["macro_f1"])
        if r["binary_duplicate"]["auc"] is not None:
            aucs.append(r["binary_duplicate"]["auc"])
    ms = lambda xs: {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs)), 4)} if xs else None
    return {"folds": k, "qwk": ms(qwks), "macro_f1": ms(f1s), "binary_auc": ms(aucs)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    sheets, source = load_sheets()
    topics = build_topics(sheets)

    # ---- the 300 expert-labelled pairs ------------------------------------------------
    gv1, gv2, pair_dims_rows, kept = [], [], [], 0
    lexical_model = build_lexical_scorer(list(topics.values()))
    concept_idf = sc.build_concept_idf(list(topics.values()))
    capability_model = sc.build_capability_model(list(topics.values())) if config.CAPABILITY_STRUCTURAL else None

    scored_pairs = []
    for row in sheets.get("Cap gan nhan", []):
        a_id = _col(row, "ID A").strip()
        b_id = _col(row, "ID B").strip()
        g1 = _to_int(_col(row, "GV1 (0-3)", "GV1"))
        g2 = _to_int(_col(row, "GV2 (0-3)", "GV2"))
        if a_id not in topics or b_id not in topics or g1 is None or g2 is None:
            continue
        s = sc.calculate_scores(topics[a_id], topics[b_id], lexical_model, concept_idf, capability_model)
        gv1.append(g1)
        gv2.append(g2)
        pair_dims_rows.append([s["semantic_score"], s["lexical_score"],
                               s["structure_score"], s["domain_score"]])
        scored_pairs.append((a_id, b_id, g1, g2, s))
        kept += 1

    dims = np.asarray(pair_dims_rows, dtype=float)
    n = len(gv1)

    # Gold: consensus ordinal (round-half-up of the mean; the two raters are ≤1 apart everywhere)
    # and the guide's binary rule ("duplicate" = both raters ≥ 2).
    gold_ord = [int(math.floor((a + b) / 2 + 0.5)) for a, b in zip(gv1, gv2)]
    bin_gold = [1 if (a >= 2 and b >= 2) else 0 for a, b in zip(gv1, gv2)]
    coarse = [min(g, 2) for g in gold_ord]  # {0,1,≥2} — so k-fold can stratify (class 3 is tiny)

    # ---- inter-annotator agreement (answers R2.5) -------------------------------------
    human = {
        "n_pairs": n,
        "cohen_kappa": round(kappa(gv1, gv2), 4),
        "quadratic_weighted_kappa": round(qwk(gv1, gv2), 4),
        "exact_agreement": round(exact_accuracy(gv1, gv2), 4),
        "within_one_agreement": round(adjacent_accuracy(gv1, gv2), 4),
        "note": "2 annotators (GV1, GV2) ⇒ Cohen's κ, NOT the paper's claimed 3-annotator Fleiss' κ.",
    }
    gold_dist = {str(k): gold_ord.count(k) for k in range(_K)}

    # ---- single stratified hold-out + k-fold CV, paper vs train-tuned weights ---------
    tr, te = train_test_split(range(n), test_size=0.40, random_state=SEED, stratify=coarse)
    results = {
        "paper_weights": {
            "holdout": evaluate_split(dims, gold_ord, bin_gold, tr, te, PAPER_WEIGHTS, tune=False),
            "cv": cross_validate(dims, gold_ord, bin_gold, coarse, PAPER_WEIGHTS, tune=False),
        },
        "tuned_on_train": {
            "holdout": evaluate_split(dims, gold_ord, bin_gold, tr, te, PAPER_WEIGHTS, tune=True),
            "cv": cross_validate(dims, gold_ord, bin_gold, coarse, PAPER_WEIGHTS, tune=True),
        },
    }

    # ---- variant diagnostic: does the AI agree with the human by change-type? (F.3) ---
    variant = variant_analysis(sheets, topics, lexical_model, concept_idf)

    report = {
        "source": source,
        "semantic_backend": backend_name(),
        "structural_mode": "capability" if config.CAPABILITY_STRUCTURAL else "sedo-tasktype",
        "human_agreement": human,
        "gold_ordinal_distribution": gold_dist,
        "binary_positives": int(sum(bin_gold)),
        "ai_vs_human": results,
        "variant_by_change_type": variant,
        "reviewer_mapping": {
            "R1.2/R2.2": "AI now scored on the 300-pair expert set (5-field) — see ai_vs_human.",
            "R2.5": "Inter-annotator κ reported — see human_agreement (Cohen's, 2 raters).",
            "R1.5/R2.4": "Weights & level thresholds fit on TRAIN only; test + k-fold CV reported.",
            "F.3": "variant_by_change_type shows the domain-switch disagreement.",
        },
    }
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    _digest(report)


def variant_analysis(sheets, topics, lexical_model, concept_idf) -> dict:
    """Score the base↔variant pairs and aggregate by human change-type. The point: the design
    label (= human GV label) for a domain-switch is only 1, yet the AI's tech-driven structural
    score stays high — quantifying where the model fights the ground truth."""
    variants = build_variants(sheets)

    def resolve(token: str):
        token = token.strip()
        m = re.match(r"^(T\d+)", token)
        if m and m.group(1) in topics:
            return topics[m.group(1)]
        if token in variants:
            return variants[token]["thesis"]
        return None

    buckets: dict[str, dict] = {}
    for row in sheets.get("Cap bien the", []):
        a = resolve(_col(row, "Đối tượng A"))
        b = resolve(_col(row, "Đối tượng B"))
        rel = _col(row, "Quan hệ")
        design = _to_int(_col(row, "Nhãn thiết kế (0-3)", "Nhãn thiết kế"))
        if a is None or b is None or design is None or not rel.startswith("gốc"):
            continue  # only base↔variant rows carry a clean change-type label
        s = sc.calculate_scores(a, b, lexical_model, concept_idf)
        d = buckets.setdefault(rel, {"n": 0, "design_label": design, "sem": [], "lex": [],
                                     "str": [], "dom": [], "overall_paper": []})
        d["n"] += 1
        d["sem"].append(s["semantic_score"])
        d["lex"].append(s["lexical_score"])
        d["str"].append(s["structure_score"])
        d["dom"].append(s["domain_score"])
        d["overall_paper"].append(composite(
            np.array([[s["semantic_score"], s["lexical_score"],
                       s["structure_score"], s["domain_score"]]]), PAPER_WEIGHTS)[0])

    out = {}
    for rel, d in buckets.items():
        avg = lambda xs: round(float(np.mean(xs)), 3) if xs else None
        out[rel] = {
            "n": d["n"], "human_design_label_0_3": d["design_label"],
            "ai_structural": avg(d["str"]), "ai_domain": avg(d["dom"]),
            "ai_semantic": avg(d["sem"]), "ai_lexical": avg(d["lex"]),
            "ai_composite_paper_weights": avg(d["overall_paper"]),
        }
    return out


def _digest(r: dict) -> None:
    h = r["human_agreement"]
    print("=" * 78)
    print(f"DASSF vs REAL ground truth   ·   source = {r['source']}   ·   semantic = {r['semantic_backend']}"
          f"   ·   structural = {r['structural_mode']}")
    print("-" * 78)
    print(f"Inter-annotator (GV1 vs GV2, n={h['n_pairs']}):  Cohen's κ = {h['cohen_kappa']}  "
          f"| QWK = {h['quadratic_weighted_kappa']}  | exact = {h['exact_agreement']:.0%}  "
          f"| ±1 = {h['within_one_agreement']:.0%}")
    print(f"   {h['note']}")
    print(f"gold ordinal 0-3 distribution: {r['gold_ordinal_distribution']}   "
          f"binary positives (both≥2): {r['binary_positives']}")
    print("-" * 78)
    print("AI ↔ human agreement on the HELD-OUT test split (level thresholds fit on train):")
    print(f"{'scheme':>16}{'weights (α β γ δ)':>22}{'QWK':>7}{'macroF1':>9}{'adj':>6}{'binAUC':>8}")
    for name in ("paper_weights", "tuned_on_train"):
        ho = r["ai_vs_human"][name]["holdout"]
        o = ho["ordinal"]
        auc = ho["binary_duplicate"]["auc"]
        print(f"{name:>16}{str(ho['weights']):>22}{o['qwk']:>7.3f}{o['macro_f1']:>9.3f}"
              f"{o['adjacent_acc']:>6.2f}{(auc if auc is not None else float('nan')):>8.3f}")
    print("  (compare AI↔human QWK above to human↔human QWK = "
          f"{h['quadratic_weighted_kappa']} — the ceiling)")
    print("-" * 78)
    print("Stability — stratified 5-fold CV (mean ± std):")
    for name in ("paper_weights", "tuned_on_train"):
        cv = r["ai_vs_human"][name]["cv"]
        print(f"  {name:>16}: QWK {cv['qwk']['mean']:.3f}±{cv['qwk']['std']:.3f}  "
              f"macroF1 {cv['macro_f1']['mean']:.3f}±{cv['macro_f1']['std']:.3f}")
    print("-" * 78)
    print("Variant pairs by change-type  (human designed label vs the AI's tech-driven signals):")
    print(f"{'relation':>26}{'human':>7}{'ai_str':>8}{'ai_dom':>8}{'ai_comp':>9}")
    order = ["gốc ↔ viết lại", "gốc ↔ thu hẹp", "gốc ↔ chuyển lĩnh vực", "gốc ↔ bẫy"]
    vb = r["variant_by_change_type"]
    for rel in order + [k for k in vb if k not in order]:
        if rel not in vb:
            continue
        d = vb[rel]
        print(f"{rel:>26}{d['human_design_label_0_3']:>7}{d['ai_structural']:>8.2f}"
              f"{d['ai_domain']:>8.2f}{d['ai_composite_paper_weights']:>9.2f}")
    print("  → 'chuyển lĩnh vực' (domain-switch) is human-label 1, but watch ai_str / ai_comp stay high:")
    print("    the paper calls this the #1 duplicate; the humans barely do. (ICTA_REVIEW_ANALYSIS §F.3)")
    print("=" * 78)
    print(f"report → {_REPORT}")


if __name__ == "__main__":
    main()
