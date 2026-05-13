"""Generate live LLM replication artifacts with prompt/response audit logs."""
from __future__ import annotations

import argparse

from fgm import write_live_llm_replication_outputs


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
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--case-family", default="deploy_rollback")
    parser.add_argument("--summary-filename", default="live_llm_replication_summary.json")
    parser.add_argument("--audit-filename", default="live_llm_replication_audit.jsonl")
    parser.add_argument("--empty-response-retries", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--no-audit-text", action="store_true")
    args = parser.parse_args()

    paths = write_live_llm_replication_outputs(
        args.output_dir,
        provider=args.provider,
        model=args.model,
        embedding_model=args.embedding_model,
        seed_count=args.seed_count,
        start_seed=args.start_seed,
        case_family=args.case_family,
        include_audit_text=not args.no_audit_text,
        summary_filename=args.summary_filename,
        audit_filename=args.audit_filename,
        empty_response_retries=args.empty_response_retries,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"summary={paths['summary']}")
    print(f"audit={paths['audit']}")


if __name__ == "__main__":
    main()
