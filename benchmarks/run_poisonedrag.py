"""CLI entry point for the PoisonedRAG external-validation benchmark.

Usage:

    # Offline-style smoke run (12 built-in sample questions, all 4 arms).
    python benchmarks/run_poisonedrag.py --smoke

    # Full run from a JSONL of real PoisonedRAG questions.
    python benchmarks/run_poisonedrag.py \\
        --data benchmarks/data/poisonedrag/nq.jsonl \\
        --arms vector,trace_memory,bash,bash_nolabels \\
        --model gpt-4.1-mini \\
        --classifier-model gpt-4.1-mini

Hard mode is the default: every passage is labelled by the LLM source
classifier (no oracle labels). Expected token bill is dominated by
the classifier + bash arms; on the 12-question sample budget ~$0.10.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from benchmarks.poisonedrag.dataset import (
    load_poisoned_questions,
    sample_questions,
)
from benchmarks.poisonedrag.runner import ARM_FACTORIES, run_benchmark, write_report
from benchmarks.product_comparison.llm_client import get_openai_client


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "PoisonedRAG external validation benchmark for trace-memory. "
        "Runs an LLM source classifier on every passage (hard mode), "
        "then evaluates each memory arm's attack-success rate."
    ))
    parser.add_argument("--data", default=None,
                        help="Path to a JSONL of PoisonedQuestion records. "
                             "Omit to use the built-in 12-question sample.")
    parser.add_argument("--smoke", action="store_true",
                        help="Use the built-in 12-question sample. Implied "
                             "when --data is omitted.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of questions used from --data.")
    parser.add_argument(
        "--arms",
        default="vector,vector_with_labels,trace_memory,bash,bash_nolabels",
        help="Comma-separated subset of arms to run.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="OpenAI model for the answer arms.")
    parser.add_argument("--classifier-model", default=None,
                        help="OpenAI model for the source classifier "
                             "(defaults to --model).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Passage-shuffle seed (for reproducibility).")
    parser.add_argument("--out-dir", default="results/benchmarks",
                        help="Where to drop the JSON + markdown report.")
    args = parser.parse_args(argv)

    arm_names = [n.strip() for n in args.arms.split(",") if n.strip()]
    for n in arm_names:
        if n not in ARM_FACTORIES:
            parser.error(f"unknown arm {n!r}; choose from {list(ARM_FACTORIES)}")

    if args.data and not args.smoke:
        questions = load_poisoned_questions(Path(args.data), limit=args.limit)
        source = f"file:{args.data}"
    else:
        questions = sample_questions()
        if args.limit is not None:
            questions = questions[: args.limit]
        source = "built-in sample"
    print(f"loaded {len(questions)} questions from {source}")
    print(f"arms: {arm_names}")
    print(f"answer model: {args.model!r}  classifier model: "
          f"{(args.classifier_model or args.model)!r}")

    client = get_openai_client()
    classifier_model = args.classifier_model or args.model

    def _on_progress(arm_, idx_, total_, score_):
        marks = (
            f"correct={int(score_.contains_correct)} "
            f"target={int(score_.contains_target)} "
            f"refused={int(score_.refused)}"
        )
        if idx_ == 1 or idx_ % 10 == 0 or idx_ == total_:
            print(f"  [{arm_}] {idx_:4d}/{total_} {score_.question_id}: {marks}")

    t0 = time.perf_counter()
    result = run_benchmark(
        questions,
        client=client,
        arm_names=arm_names,
        answer_model=args.model,
        classifier_model=classifier_model,
        seed=args.seed,
        on_progress=_on_progress,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nfinished in {elapsed:.1f}s")
    print()

    for arm, agg in result.arm_aggregates.items():
        print(f"  {arm:>14s}  ASR={agg.attack_success_rate:.3f}  "
              f"clean_acc={agg.clean_accuracy:.3f}  "
              f"refused={agg.refusal_rate:.3f}  "
              f"tok={agg.total_tokens:>6d}  "
              f"tok/q={agg.tokens_per_question:>5.0f}  "
              f"api={agg.total_api_calls:>4d}  tool={agg.total_tool_calls:>4d}  "
              f"wall={agg.total_elapsed:.1f}s")

    cstats = result.classifier_stats
    print()
    print(f"  classifier: {cstats.n} passages, "
          f"in_tok={cstats.total_input_tokens}, out_tok={cstats.total_output_tokens}, "
          f"parse_err={cstats.n_parse_error}")
    for kind, row in result.classifier_confusion.items():
        print(f"    {kind}: {dict(row)}")

    out_dir = Path(args.out_dir)
    write_report(result, out_dir)
    print(f"\nwrote {out_dir / 'poisonedrag_results.json'}")
    print(f"wrote {out_dir / 'POISONEDRAG.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
