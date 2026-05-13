import numpy as np

from trace_probes import (
    LeakyTraceOperator,
    build_trace_records,
    causal_trace_probe,
    fold_force_probe,
    p7_retrieval_regime_probe,
    trace_retention_probe,
)


def one_hot(i, d):
    x = np.zeros(d)
    x[i] = 1.0
    return x


def test_trace_retention_and_causal_probe():
    seq = [one_hot(2, 4), one_hot(0, 4), one_hot(1, 4), one_hot(3, 4)]
    op = LeakyTraceOperator(dim=4, decay=0.8, max_lag=10)
    retention = trace_retention_probe(seq, op, t=3, k=3)
    assert retention.metrics["lag_intensity"] > 0

    causal = causal_trace_probe(
        seq,
        lambda: LeakyTraceOperator(dim=4, decay=0.8, max_lag=10),
        t=3,
        k=3,
        replacement_state=one_hot(1, 4),
    )
    assert causal.metrics["projection_delta_l2"] > 0


def test_fold_force_probe_excludes_bookkeeping():
    seq = [one_hot(2, 4), one_hot(0, 4), one_hot(1, 4), one_hot(3, 4)]
    records, _ = build_trace_records(seq, LeakyTraceOperator(dim=4, decay=0.8, max_lag=10))
    result = fold_force_probe(records, query=one_hot(2, 4), target_id="r0", n_actions=4, k=1, target_action=2)
    assert result.metrics["bookkeeping_ignored"] is True
    assert result.metrics["fold_ablation_l2_h_cog"] > 0


def test_p7_probe_runs():
    sparse = p7_retrieval_regime_probe("sparse_confusable", n_values=(8, 16), n_queries=20, seed=1)
    rich = p7_retrieval_regime_probe("rich_distinctive", n_values=(8, 16), n_queries=20, seed=1)
    assert len(sparse.metrics["rows"]) == 2
    assert len(rich.metrics["rows"]) == 2
