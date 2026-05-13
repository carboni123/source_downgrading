"""Generate real-component validation artifact."""
from __future__ import annotations

import argparse

from fgm import write_real_component_validation_output


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
    args = parser.parse_args()

    path = write_real_component_validation_output(args.output_dir, model_name=args.model)
    print(f"real_component_summary={path}")


if __name__ == "__main__":
    main()
