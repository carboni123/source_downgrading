"""Run deterministic source-aware rerank boundary regression."""
from __future__ import annotations

import argparse

from fgm import write_rerank_boundary_regression_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    path = write_rerank_boundary_regression_output(args.output_dir)
    print(f"rerank_boundary_regression={path}")


if __name__ == "__main__":
    main()
