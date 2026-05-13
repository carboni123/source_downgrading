"""Generate multi-seed controlled replication artifacts."""
from __future__ import annotations

import argparse

from fgm import run_controlled_replication, write_controlled_replication_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    paths = write_controlled_replication_outputs(
        args.output_dir,
        seed_count=args.seed_count,
        start_seed=args.start_seed,
    )
    summary = run_controlled_replication(
        seed_count=args.seed_count,
        start_seed=args.start_seed,
    )
    acceptance = summary["acceptance"]
    print(f"summary={paths['summary']}")
    print(f"runs={paths['runs']}")
    print("seed_count", summary["seed_count"])
    print("minimum_effect_hold_rate", acceptance["minimum_effect_hold_rate"])
    print("all_controlled_replication_gates_met", acceptance["all_controlled_replication_gates_met"])


if __name__ == "__main__":
    main()
