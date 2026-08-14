"""Tests for the benchmark metric helpers (benchmark.py) — pure, no SBERT/DB needed."""

from benchmark import _NO_ONTOLOGY, _mddm, _retrieval_metrics


def test_no_ontology_weights_are_a_valid_simplex_point():
    assert abs(sum(_NO_ONTOLOGY) - 1.0) < 1e-9
    # γ = δ = 0 → the ontology dimensions are zeroed out (paper §VII-A, B4).
    assert _NO_ONTOLOGY[2] == 0.0 and _NO_ONTOLOGY[3] == 0.0


def test_mddm_weighted_sum():
    assert _mddm((1.0, 1.0, 0.0, 0.0), _NO_ONTOLOGY) == 1.0
    assert _mddm((0.0, 0.0, 1.0, 1.0), _NO_ONTOLOGY) == 0.0  # ontology dims ignored by B4
    assert _mddm((1.0, 0.0, 0.0, 0.0), (0.3, 0.2, 0.3, 0.2)) == 0.3


def test_retrieval_metrics_perfect_ranking():
    resolved = [{"target_idx": 0}, {"target_idx": 1}]

    def score_fn(qi, ci):
        return 1.0 if ci == resolved[qi]["target_idx"] else 0.0

    m = _retrieval_metrics(resolved, 4, score_fn)
    assert m["recall@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["auc"] == 1.0


def test_retrieval_metrics_target_ranked_third():
    resolved = [{"target_idx": 2}]
    scores = [0.9, 0.8, 0.5, 0.1]  # target (idx 2) has two candidates above, one below

    m = _retrieval_metrics(resolved, 4, lambda qi, ci: scores[ci])
    assert m["recall@1"] == 0.0
    assert m["recall@5"] == 1.0          # rank 3 ≤ 5
    assert round(m["mrr"], 4) == round(1 / 3, 4)
    assert round(m["auc"], 4) == round(1 / 3, 4)  # 1 of 3 non-targets scores below the target


def test_retrieval_metrics_recall_at_10_boundary():
    resolved = [{"target_idx": 11}]
    scores = [1.0] * 11 + [0.5] + [0.0] * 8  # target at index 11 → rank 12 (11 above)

    m = _retrieval_metrics(resolved, 20, lambda qi, ci: scores[ci])
    assert m["recall@1"] == 0.0
    assert m["recall@10"] == 0.0  # rank 12 > 10
    assert round(m["mrr"], 4) == round(1 / 12, 4)


def test_retrieval_metrics_handles_ties_without_favoring_target():
    resolved = [{"target_idx": 0}]
    # everything ties → target must NOT be counted rank 1 for free; AUC = 0.5 (pure ties)
    m = _retrieval_metrics(resolved, 4, lambda qi, ci: 0.7)
    assert m["auc"] == 0.5
