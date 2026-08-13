"""DASSF evaluation harness — scores the AI against the teacher's ground truth.

Ground truth: ``data/FA26_topics_deduplicated.json`` — for each Fall-2026 topic the teacher
recorded the single most-likely duplicate (``topicMostLikelyDuplicate``) among prior topics
and an expected duplication score (``duplicationResult``). This is the closest thing we have
to the paper's expert-annotated benchmark (paper §VI), so we treat it as the validation set
the paper tunes on (§V-E) and calibrates against.

Corpus (prior topics): the same Fall-2025 / Spring-2026 / Summer-2026 topics the web system
seeds, loaded straight from ``seed_thesis.py`` — no database needed.

What it reports:
  1. Ranking accuracy  — for each FA topic, is the teacher's target in our Top-1 / Top-3 / Top-5?
  2. Score agreement   — MAE and Pearson/Spearman between our composite and the teacher's score.
  3. SEDO coverage     — share of titles where the ontology recognizes ≥1 tech / ≥1 domain concept.
  4. Ablation          — Top-1 accuracy when each MDDM dimension is removed (paper Table V).
  5. Weight tuning     — grid-search over the simplex (step 0.05) for the best (α,β,γ,δ),
                         replacing the hard-coded guess with data-grounded weights.

Run (no DB required):
    python evaluate.py
Writes a machine-readable summary to ``data/eval_report.json`` and prints a digest.
"""

from __future__ import annotations

import json
import os
from collections import namedtuple

from app.services.semantic_encoder import backend_name
from app.utils import score_calculator as sc
from app.utils.text_cleaner import normalize_key
from seed_thesis import ENRICHED, FALL_2025, SPRING_2026, SUMMER_2026, _split_tech

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_FA_FILE = os.path.join(_DATA_DIR, "FA26_topics_deduplicated.json")
_REPORT_FILE = os.path.join(_DATA_DIR, "eval_report.json")

_DEFAULT_WEIGHTS = (0.30, 0.20, 0.30, 0.20)  # (semantic, lexical, structure, domain)
_DIMS = ("semantic", "lexical", "structure", "domain")

Tag = namedtuple("Tag", "name")


class EvalThesis:
    """Duck-typed stand-in for the SQLAlchemy Thesis, so the scorer runs without a DB."""

    def __init__(self, title, description=None, scope=None, objectives=None,
                 expected_result=None, technologies=(), aliases=()):
        self.title = title
        self.description = description
        self.scope = scope
        self.objectives = objectives
        self.expected_result = expected_result
        self.technologies = [Tag(t) for t in technologies]
        self.structures = []
        self.domains = []
        # Alternate title strings used only to match the teacher's target back to this topic.
        self.aliases = {a for a in ({title, *aliases}) if a}


def build_corpus() -> list[EvalThesis]:
    """Prior topics, mirroring seed_thesis.py (Fall-25 title-only; Spring/Summer enriched)."""
    corpus: list[EvalThesis] = []
    for title in FALL_2025:
        corpus.append(EvalThesis(title))
    for i, list_title in enumerate(SPRING_2026):
        detail = ENRICHED.get(f"SP_{i + 1:02d}")
        corpus.append(_from_detail(detail, list_title) if detail else EvalThesis(list_title))
    for g, list_title in enumerate(SUMMER_2026):
        detail = ENRICHED.get(f"SU_{g + 1:02d}")
        corpus.append(_from_detail(detail, list_title) if detail else EvalThesis(list_title))
    return corpus


def _from_detail(detail: dict, list_title: str) -> EvalThesis:
    return EvalThesis(
        title=detail["titleEn"],
        description=detail.get("description"),
        scope=detail.get("scope"),
        objectives=detail.get("objective"),
        expected_result=detail.get("expectedResult"),
        technologies=_split_tech(detail.get("technology")),
        aliases=(list_title,),
    )


def load_queries() -> list[dict]:
    with open(_FA_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    queries = []
    for code, row in data.items():
        title = (row.get("titleEn") or "").strip()
        target = (row.get("topicMostLikelyDuplicate") or "").strip()
        if not title or not target:
            continue
        queries.append({
            "code": code,
            "title": title,
            "gold_target": target,
            "gold_score": row.get("duplicationResult"),
        })
    return queries


# ── matching the teacher's free-text target back to a corpus topic ──────────────────
def _token_set(text: str) -> set[str]:
    return {t for t in normalize_key(text).split() if len(t) > 1}


def _title_jaccard(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def resolve_target(target: str, corpus: list[EvalThesis]) -> tuple[int | None, float]:
    """Best-matching corpus index for the teacher's target string (over all title aliases)."""
    best_idx, best_score = None, 0.0
    for idx, item in enumerate(corpus):
        score = max(_title_jaccard(target, alias) for alias in item.aliases)
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx, best_score


# ── scoring ─────────────────────────────────────────────────────────────────────────
def dims_matrix(queries, corpus, idf):
    """queries × corpus grid of (semantic, lexical, structure, domain) tuples."""
    matrix = []
    for q in queries:
        row = []
        for cand in corpus:
            s = sc.calculate_scores(q["_thesis"], cand, idf)
            row.append((s["semantic_score"], s["lexical_score"], s["structure_score"], s["domain_score"]))
        matrix.append(row)
    return matrix


def _overall(dims, weights):
    return sum(d * w for d, w in zip(dims, weights))


def ranking_metrics(queries, matrix, weights):
    """Top-1/3/5 accuracy over queries whose target resolved, plus score-agreement stats."""
    resolved = [q for q in queries if q["target_idx"] is not None]
    hits = {1: 0, 3: 0, 5: 0}
    ours, theirs = [], []
    for q in resolved:
        scored = sorted(
            range(len(matrix[q["_row"]])),
            key=lambda ci: _overall(matrix[q["_row"]][ci], weights),
            reverse=True,
        )
        rank = scored.index(q["target_idx"]) + 1
        for k in hits:
            if rank <= k:
                hits[k] += 1
        if q["gold_score"] is not None:
            ours.append(_overall(matrix[q["_row"]][q["target_idx"]], weights))
            theirs.append(float(q["gold_score"]))
    n = len(resolved)
    return {
        "n_resolved": n,
        "top1": hits[1] / n if n else 0.0,
        "top3": hits[3] / n if n else 0.0,
        "top5": hits[5] / n if n else 0.0,
        "score_mae": _mae(ours, theirs),
        "score_pearson": _pearson(ours, theirs),
        "score_spearman": _spearman(ours, theirs),
    }


# ── tiny stats helpers (avoid extra deps) ───────────────────────────────────────────
def _mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a) if a else None


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else None


def _spearman(a, b):
    if len(a) < 2:
        return None
    return _pearson(_ranks(a), _ranks(b))


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


# ── weight grid search over the simplex (step 0.05) ─────────────────────────────────
def simplex(step_pct=5):
    steps = 100 // step_pct
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            for c in range(steps + 1 - a - b):
                d = steps - a - b - c
                yield (a * step_pct / 100, b * step_pct / 100, c * step_pct / 100, d * step_pct / 100)


def tune_weights(queries, matrix):
    best, best_key = None, None
    for w in simplex(5):
        m = ranking_metrics(queries, matrix, w)
        # maximize Top-1, then Top-3, then Top-5, then minimize score MAE.
        key = (m["top1"], m["top3"], m["top5"], -(m["score_mae"] or 1.0))
        if best_key is None or key > best_key:
            best_key, best = key, (w, m)
    return best


def ablation(queries, matrix):
    """Drop each dimension from the default weights (renormalize) and re-score (paper Table V)."""
    base = ranking_metrics(queries, matrix, _DEFAULT_WEIGHTS)["top1"]
    rows = {}
    for i, name in enumerate(_DIMS):
        w = list(_DEFAULT_WEIGHTS)
        w[i] = 0.0
        total = sum(w)
        w = tuple(x / total for x in w) if total else tuple(w)
        top1 = ranking_metrics(queries, matrix, w)["top1"]
        rows[name] = {"top1": round(top1, 4), "delta_vs_full": round(top1 - base, 4)}
    return base, rows


def sedo_coverage(theses) -> dict:
    """Share of titles where SEDO recognizes ≥1 tech/method concept and ≥1 domain concept."""
    sedo = sc.get_sedo()
    struct_layers = {"TechnicalStack", "Methodology", "TaskType"}
    tech_hits = dom_hits = 0
    for t in theses:
        concepts = sedo.recognize(t.title or "")
        layers = {sedo.nodes[c]["layer"] for c in concepts}
        if layers & struct_layers:
            tech_hits += 1
        if "DomainEntity" in layers:
            dom_hits += 1
    n = len(theses) or 1
    return {"n": len(theses), "tech_or_method": round(tech_hits / n, 4), "domain": round(dom_hits / n, 4)}


def main() -> None:
    corpus = build_corpus()
    queries = load_queries()
    for row, q in enumerate(queries):
        q["_row"] = row
        q["_thesis"] = EvalThesis(q["title"])
        idx, match = resolve_target(q["gold_target"], corpus)
        q["target_idx"] = idx if match >= 0.5 else None
        q["target_match"] = round(match, 3)

    idf = sc.build_idf(corpus)
    matrix = dims_matrix(queries, corpus, idf)

    default_metrics = ranking_metrics(queries, matrix, _DEFAULT_WEIGHTS)
    best_weights, tuned_metrics = tune_weights(queries, matrix)
    base_top1, ablation_rows = ablation(queries, matrix)

    report = {
        "semantic_backend": backend_name(),
        "corpus_size": len(corpus),
        "queries_total": len(queries),
        "queries_resolved": default_metrics["n_resolved"],
        "sedo_coverage": {
            "corpus": sedo_coverage(corpus),
            "queries": sedo_coverage([q["_thesis"] for q in queries]),
        },
        "default_weights": {"weights": _DEFAULT_WEIGHTS, "metrics": _round(default_metrics)},
        "tuned_weights": {"weights": best_weights, "metrics": _round(tuned_metrics)},
        "ablation_from_default": {"full_top1": round(base_top1, 4), "drop": ablation_rows},
    }
    with open(_REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    _print_digest(report)


def _round(metrics: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def _print_digest(r: dict) -> None:
    print("=" * 68)
    print(f"DASSF evaluation  ·  semantic backend = {r['semantic_backend']}")
    print(f"corpus={r['corpus_size']} topics · queries={r['queries_total']} "
          f"(resolved to a gold target: {r['queries_resolved']})")
    cov = r["sedo_coverage"]
    print(f"SEDO coverage  corpus: tech {cov['corpus']['tech_or_method']:.0%} / "
          f"domain {cov['corpus']['domain']:.0%}   |   "
          f"queries: tech {cov['queries']['tech_or_method']:.0%} / domain {cov['queries']['domain']:.0%}")
    print("-" * 68)
    dm, tm = r["default_weights"]["metrics"], r["tuned_weights"]["metrics"]
    print(f"{'':22}{'Top-1':>8}{'Top-3':>8}{'Top-5':>8}{'MAE':>8}{'Pearson':>9}")
    print(f"default {str(r['default_weights']['weights']):>14}"
          f"{dm['top1']:>8.2f}{dm['top3']:>8.2f}{dm['top5']:>8.2f}"
          f"{(dm['score_mae'] or 0):>8.3f}{(dm['score_pearson'] or 0):>9.2f}")
    print(f"tuned   {str(tuple(round(x,2) for x in r['tuned_weights']['weights'])):>14}"
          f"{tm['top1']:>8.2f}{tm['top3']:>8.2f}{tm['top5']:>8.2f}"
          f"{(tm['score_mae'] or 0):>8.3f}{(tm['score_pearson'] or 0):>9.2f}")
    print("-" * 68)
    print("Ablation from default weights (Top-1 when a dimension is removed):")
    for name, row in r["ablation_from_default"]["drop"].items():
        print(f"  − {name:10}  Top-1 = {row['top1']:.2f}   Δ = {row['delta_vs_full']:+.2f}")
    print("=" * 68)
    print(f"report → {_REPORT_FILE}")


if __name__ == "__main__":
    main()
