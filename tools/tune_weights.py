"""Learn the MDDM weights by grid-searching on the teacher benchmark, and measure the
step-size ↔ time trade-off.

The four weights (α,β,γ,δ) are a GLOBAL property — they are tuned once against the labelled
benchmark (data/FA26_topics_deduplicated.json) and do not depend on any single new topic, so the
tuning is done here (offline) and cached to data/tuned_weights.json. Enable them in production with
MDDM_USE_TUNED=true. Re-running per duplication check would return the same weights (see the timings
below — the grid search itself is milliseconds; the cost is the one-time SBERT scoring of the
benchmark, which is independent of the query).

    python tools/tune_weights.py                 # compares steps, writes step-0.05 weights
    MDDM_TUNE_STEP=0.025 python tools/tune_weights.py   # write the step-0.025 result instead
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import level_metrics, prepare_benchmark, simplex_size, tune_weights  # noqa: E402

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_STEPS = (0.10, 0.05, 0.04, 0.025)


def main() -> None:
    t0 = time.perf_counter()
    queries, corpus, matrix = prepare_benchmark()
    n_corpus = len(corpus)
    prep = time.perf_counter() - t0
    print(f"[prep] SBERT + score {len(queries)}×{n_corpus} pairs = {prep:.1f}s  (ONE-TIME, "
          f"independent of grid step and of the query)\n")

    print(f"{'step':>7}{'grid pts':>10}{'time (s)':>10}{'macro-F1':>10}{'acc':>7}   best (α, β, γ, δ)")
    print("-" * 78)
    results = {}
    for step in _STEPS:
        n = round(1 / step)
        started = time.perf_counter()
        best = tune_weights(queries, matrix, n)
        elapsed = time.perf_counter() - started
        metrics = level_metrics(queries, matrix, best)
        results[step] = (best, metrics, elapsed)
        print(f"{step:>7}{simplex_size(n):>10}{elapsed:>10.3f}{metrics['macro_f1']:>10.2f}"
              f"{metrics['level_accuracy']:>7.2f}   {tuple(round(x, 3) for x in best)}")

    # Persist the weights for the chosen step (default 0.05) so production can use them.
    chosen = float(os.getenv("MDDM_TUNE_STEP", "0.05"))
    best, metrics, _ = results.get(chosen, results[0.05])
    payload = {
        "weights": {k: round(v, 4) for k, v in zip(("semantic", "lexical", "structure", "domain"), best)},
        "step": chosen,
        "macro_f1": metrics["macro_f1"],
        "level_accuracy": metrics["level_accuracy"],
        "source": "grid-search (level macro-F1) on data/FA26_topics_deduplicated.json",
    }
    out = os.path.join(_DATA_DIR, "tuned_weights.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\n→ wrote {out}")
    print(f"  weights (step {chosen}): {payload['weights']}")
    print("  Enable in production with:  MDDM_USE_TUNED=true")


if __name__ == "__main__":
    main()
