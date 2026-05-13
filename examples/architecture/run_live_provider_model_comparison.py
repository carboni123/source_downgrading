"""Generate live provider/model comparison artifacts."""
from __future__ import annotations

import argparse

from fgm import parse_provider_model_target, write_live_provider_model_comparison_outputs


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
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Provider/model target as provider or provider:model. Repeat for multiple targets.",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--empty-response-retries", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--min-passed-targets", type=int, default=1)
    parser.add_argument("--artifact-prefix", default="live_provider_model_comparison")
    parser.add_argument("--summary-filename", default="live_provider_model_comparison_summary.json")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--include-audit-text", action="store_true")
    args = parser.parse_args()

    targets = [parse_provider_model_target(target) for target in args.target] if args.target else None
    paths = write_live_provider_model_comparison_outputs(
        args.output_dir,
        targets=targets,
        embedding_model=args.embedding_model,
        seed_count=args.seed_count,
        start_seed=args.start_seed,
        empty_response_retries=args.empty_response_retries,
        max_output_tokens=args.max_output_tokens,
        include_audit_text=args.include_audit_text,
        min_passed_targets=args.min_passed_targets,
        reuse_existing=args.reuse_existing,
        artifact_prefix=args.artifact_prefix,
        summary_filename=args.summary_filename,
    )
    print(f"summary={paths['summary']}")
    for name in sorted(path for path in paths if path != "summary"):
        print(f"{name}={paths[name]}")


if __name__ == "__main__":
    main()
