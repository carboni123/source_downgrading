from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from trace_probes import (
    LeakyTraceOperator,
    build_trace_records,
    causal_trace_probe,
    fold_force_probe,
    p7_retrieval_regime_probe,
    trace_retention_probe,
)


def one_hot(i: int, d: int) -> np.ndarray:
    x = np.zeros(d)
    x[i] = 1.0
    return x


def main() -> None:
    dim = 4
    cue_action = 2
    sequence = [
        one_hot(cue_action, dim),
        np.array([0.20, 0.10, 0.00, 0.00]),
        np.array([0.00, 0.25, 0.00, 0.00]),
        np.array([0.15, 0.00, 0.00, 0.00]),
        np.array([0.10, 0.00, 0.00, 0.15]),
        np.array([0.00, 0.00, 0.00, 0.10]),
    ]

    def op_factory() -> LeakyTraceOperator:
        return LeakyTraceOperator(dim=dim, decay=0.8, max_lag=10)

    retention = trace_retention_probe(sequence, op_factory(), t=5, k=5)
    causal = causal_trace_probe(sequence, op_factory, t=5, k=5, replacement_state=one_hot(1, dim))
    records, _states = build_trace_records(sequence, op_factory())
    fold_force = fold_force_probe(records, query=one_hot(cue_action, dim), target_id="r0", n_actions=dim, k=1, target_action=cue_action)

    results = {
        "trace_retention": retention.metrics,
        "causal_trace": causal.metrics,
        "fold_force": fold_force.metrics,
        "p7_sparse_confusable": p7_retrieval_regime_probe("sparse_confusable", seed=9).metrics,
        "p7_compressed_preserving": p7_retrieval_regime_probe("compressed_preserving", seed=9).metrics,
        "p7_rich_distinctive": p7_retrieval_regime_probe("rich_distinctive", seed=9).metrics,
        "p7_aggressive_lossy": p7_retrieval_regime_probe("aggressive_lossy", seed=9).metrics,
    }

    out = Path(__file__).with_name("demo_results.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
