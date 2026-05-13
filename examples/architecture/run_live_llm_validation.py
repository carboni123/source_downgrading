"""Generate live LLM validation artifact."""
from __future__ import annotations

import argparse

from fgm import write_live_llm_validation_output


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
    parser.add_argument("--case-family", default="deploy_rollback")
    parser.add_argument("--empty-response-retries", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    args = parser.parse_args()

    path = write_live_llm_validation_output(
        args.output_dir,
        provider=args.provider,
        model=args.model,
        embedding_model=args.embedding_model,
        case_family=args.case_family,
        empty_response_retries=args.empty_response_retries,
        max_output_tokens=args.max_output_tokens,
    )
    print(f"live_llm_summary={path}")


if __name__ == "__main__":
    main()
