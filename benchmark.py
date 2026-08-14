"""DASSF baseline comparison — reproduces the paper's Table IV method comparison.

The paper benchmarks the full framework against four baselines (§VII-A):
  B1 TF-IDF Cosine · B2 BM25 · B3 SBERT · B4 DASSF-without-ontology · Full DASSF.

We do not have the paper's 500 expert-annotated 4-level pairs, so we evaluate the same five
methods as a **retrieval task** on the teacher benchmark (data/FA26_topics_deduplicated.json):
for each Fall-2026 topic, how well does each method rank the teacher's chosen duplicate among all
prior topics? This is well-powered (one query per topic) and isolates the exact quantity the paper
cares about — whether ontology grounding + multi-dimensional fusion beat the plain baselines.

Metrics per method: Recall@1 / @5 / @10, MRR (mean reciprocal rank of the gold target), and mean
per-query AUC (probability the gold target outranks a random non-target).

Run (no DB required):
    python benchmark.py
Writes data/benchmark_report.json and prints a Table-IV-style digest.
"""

from __future__ import annotations

import json
import os

from app.services.baselines import BM25Index, TfidfCosine
from app.services.lexical import build_lexical_scorer
from app.services.semantic_encoder import backend_name
from app.utils import score_calculator as sc
from evaluate import EvalThesis, build_corpus, load_queries, resolve_target

_REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "benchmark_report.json")

# B4 collapses MDDM onto semantic+lexical only (γ=δ=0), renormalized (paper §VII-A).
_NO_ONTOLOGY = (0.6, 0.4, 0.0, 0.0)


def _title(thesis) -> str:
    return (getattr(thesis, "title", None) or "").strip()


def _mddm(dims, weights) -> float:
    return sum(d * w for d, w in zip(dims, weights))


def _score_functions(resolved, corpus, dims, tfidf, bm25):
    """A scoring closure ``f(query_index, candidate_index) -> float`` per method."""
    w = sc.WEIGHTS
    full = (w["semantic"], w["lexical"], w["structure"], w["domain"])
    return {
        "B1 TF-IDF Cosine": lambda qi, ci: tfidf.similarity(resolved[qi]["title"], _title(corpus[ci])),
        "B2 BM25": lambda qi, ci: bm25.score(resolved[qi]["title"], ci),
        "B3 SBERT": lambda qi, ci: dims[(qi, ci)][0],
        "B4 DASSF (no ontology)": lambda qi, ci: _mddm(dims[(qi, ci)], _NO_ONTOLOGY),
        "Full DASSF": lambda qi, ci: _mddm(dims[(qi, ci)], full),
    }


def _retrieval_metrics(resolved, n_corpus, score_fn) -> dict:
    recall = {1: 0, 5: 0, 10: 0}
    reciprocal_rank = 0.0
    auc_sum = 0.0
    n = len(resolved)
    for qi, q in enumerate(resolved):
        target = q["target_idx"]
        scores = [score_fn(qi, ci) for ci in range(n_corpus)]
        # rank of the gold target (1 = best); ties broken so the target is not unfairly advantaged.
        target_score = scores[target]
        better = sum(1 for ci, s in enumerate(scores) if ci != target and s > target_score)
        rank = better + 1
        for k in recall:
            if rank <= k:
                recall[k] += 1
        reciprocal_rank += 1.0 / rank
        others = [s for ci, s in enumerate(scores) if ci != target]
        below = sum(1 for s in others if s < target_score) + 0.5 * sum(1 for s in others if s == target_score)
        auc_sum += below / len(others) if others else 0.0
    return {
        "recall@1": round(recall[1] / n, 4),
        "recall@5": round(recall[5] / n, 4),
        "recall@10": round(recall[10] / n, 4),
        "mrr": round(reciprocal_rank / n, 4),
        "auc": round(auc_sum / n, 4),
    }


def main() -> None:
    corpus = build_corpus()
    queries = load_queries()
    for row, q in enumerate(queries):
        q["_row"] = row
        q["_thesis"] = EvalThesis(q["title"])
        idx, match = resolve_target(q["gold_target"], corpus)
        q["target_idx"] = idx if match >= 0.5 else None
    resolved = [q for q in queries if q["target_idx"] is not None]

    corpus_texts = [_title(c) for c in corpus]
    tfidf = TfidfCosine(corpus_texts)
    bm25 = BM25Index(corpus_texts)
    lexical_model = build_lexical_scorer(corpus)
    concept_idf = sc.build_concept_idf(corpus)

    # Precompute the four DASSF sub-scores once per (query, candidate); baselines reuse them.
    dims: dict[tuple[int, int], tuple] = {}
    for qi, q in enumerate(resolved):
        for ci, cand in enumerate(corpus):
            s = sc.calculate_scores(q["_thesis"], cand, lexical_model, concept_idf)
            dims[(qi, ci)] = (s["semantic_score"], s["lexical_score"], s["structure_score"], s["domain_score"])

    functions = _score_functions(resolved, corpus, dims, tfidf, bm25)
    results = {name: _retrieval_metrics(resolved, len(corpus), fn) for name, fn in functions.items()}

    report = {
        "semantic_backend": backend_name(),
        "corpus_size": len(corpus),
        "queries_evaluated": len(resolved),
        "task": "retrieval of the teacher's most-likely-duplicate among all prior topics",
        "methods": results,
    }
    with open(_REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    _print_digest(report)


def _print_digest(r: dict) -> None:
    print("=" * 78)
    print(f"DASSF baseline comparison (Table IV style)  ·  semantic backend = {r['semantic_backend']}")
    print(f"task: {r['task']}")
    print(f"corpus = {r['corpus_size']} prior topics · queries = {r['queries_evaluated']}")
    print("-" * 78)
    print(f"{'Method':26}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'AUC':>8}")
    for name, m in r["methods"].items():
        print(f"{name:26}{m['recall@1']:>8.2f}{m['recall@5']:>8.2f}{m['recall@10']:>8.2f}"
              f"{m['mrr']:>8.2f}{m['auc']:>8.2f}")
    print("=" * 78)
    print(f"report → {_REPORT_FILE}")


if __name__ == "__main__":
    main()
