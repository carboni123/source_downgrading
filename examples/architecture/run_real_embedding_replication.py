"""Generate real-embedding replication artifacts."""
from __future__ import annotations

import argparse

from fgm import write_real_embedding_replication_output


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    paths = write_real_embedding_replication_output(
        args.output_dir,
        model_name=args.model,
        seed_count=args.seed_count,
        start_seed=args.start_seed,
    )
    print(f"summary={paths['summary']}")
    print(f"runs={paths['runs']}")


if __name__ == "__main__":
    main()
