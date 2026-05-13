"""Generate controlled roadmap validation artifacts."""
from __future__ import annotations

import argparse

from fgm import run_controlled_roadmap_validations, write_roadmap_validation_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = write_roadmap_validation_outputs(args.output_dir, seed=args.seed)
    summary = run_controlled_roadmap_validations(seed=args.seed)
    print(f"summary={paths['summary']}")
    print(f"source_routing={paths['source_routing']}")
    print("source_route_accuracy", summary["source_routing"]["source_sensitive"]["route_accuracy"])
    print("inscription_utility_lift", summary["inscription_utility"]["utility_write"]["future_task_lift"])
    print("correction_transfer", summary["correction_chains"]["correction_chain"]["transfer_success"])
    print("residual_precision", summary["residual_attention"]["residual_posture_source"]["transition_effective_retrieval_precision"])
    print("self_index_binding", summary["self_index_binding"]["self_indexed"]["correct_binding_rate"])
    print("coupled_attention_shift", summary["coupled_field"]["source_aware"]["attention_shift_after_memory_ablation"])


if __name__ == "__main__":
    main()
