"""Analyze live LLM replication audit artifacts."""
from __future__ import annotations

import argparse

from fgm import write_live_replication_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--summary-filename", default="live_llm_replication_summary.json")
    parser.add_argument("--audit-filename", default="live_llm_replication_audit.jsonl")
    parser.add_argument("--diagnostics-filename", default="live_llm_replication_diagnostics.json")
    args = parser.parse_args()

    path = write_live_replication_diagnostics(
        args.output_dir,
        summary_filename=args.summary_filename,
        audit_filename=args.audit_filename,
        diagnostics_filename=args.diagnostics_filename,
    )
    print(f"diagnostics={path}")


if __name__ == "__main__":
    main()
