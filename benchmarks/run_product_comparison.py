"""Three-arm product-comparison benchmark.

Multi-turn sessions across three memory arms:

    bash         -- markdown files + LLM tool-use (glob/grep/read_file)
    vector       -- in-memory cosine top-k + LLM (no source labels)
    trace_memory -- closed loop: source-labelled reactivation envelope,
                    SSIR routing kwargs from a regex post-classifier,
                    add_derived inscription on every LLM-emitted answer.

Same dataset, same LLM, same grading. The headline metrics are
correctness, unsafe-contamination rate, and tokens-per-correct-answer.

Usage:

    python benchmarks/run_product_comparison.py --smoke         # 1 session per domain, all arms
    python benchmarks/run_product_comparison.py --full          # full dataset
    python benchmarks/run_product_comparison.py --arms vector,trace_memory
    python benchmarks/run_product_comparison.py --model gpt-4.1-mini

Requires OPENAI_API_KEY in env or .env. Cost on --smoke: roughly $0.05.
Cost on --full depends primarily on the bash arm's tool-use loop length;
budget ~$0.50 for a one-shot full sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.product_comparison.adversarial_reload import load_adversarial_dataset
from benchmarks.product_comparison.adversarial_reload_v2 import (
    load_adversarial_v2_dataset,
    v2_dataset_summary,
)
from benchmarks.product_comparison.arms import BashArm, TraceMemoryArm, VectorArm
from benchmarks.product_comparison.arms.base import MemoryArm
from benchmarks.product_comparison.dataset import Session, load_dataset
from benchmarks.product_comparison.grading import ArmAggregate
from benchmarks.product_comparison.judge import judge_run_payload
from benchmarks.product_comparison.llm_client import get_openai_client
from benchmarks.product_comparison.session import (
    SessionRunResult,
    aggregate,
    run_session,
)


ARM_FACTORIES: Dict[str, callable] = {
    "vector":         lambda client, model: VectorArm(client, model=model),
    "trace_memory":   lambda client, model: TraceMemoryArm(client, model=model),
    "bash":           lambda client, model: BashArm(client, model=model),
    "bash_nolabels":  lambda client, model: BashArm(
        client, model=model, include_source_labels=False,
    ),
}


def _smoke_subset(sessions: Sequence[Session]) -> List[Session]:
    """One session per domain — enough to validate the wiring + headline shape."""
    seen: set = set()
    chosen: List[Session] = []
    for session in sessions:
        if session.domain in seen:
            continue
        seen.add(session.domain)
        chosen.append(session)
    return chosen


def _format_aggregate(agg: ArmAggregate) -> str:
    parse_rate = (agg.n_parse_error / agg.n_questions) if agg.n_questions else 0.0
    return (
        f"  q={agg.n_questions:4d}  "
        f"strict={agg.correctness_rate:.3f}  "
        f"defens={agg.defensible_correctness_rate:.3f}  "
        f"unsafe={agg.unsafe_rate:.3f}  "
        f"contam_unsafe={agg.contam_unsafe_rate:.3f}  "
        f"clean_act={agg.clean_act_rate:.3f}  "
        f"parse_err={parse_rate:.3f}  "
        f"tok={agg.total_tokens:>7d}  "
        f"tok/sess={agg.tokens_per_session:>7.0f}  "
        f"tok/avoided={agg.tokens_per_unsafe_avoided:>8.0f}  "
        f"api={agg.total_api_calls:>4d}  "
        f"tool={agg.total_tool_calls:>4d}  "
        f"wall={agg.total_elapsed_seconds:.1f}s"
    )


def _write_results(
    out_dir: Path,
    *,
    arm_aggregates: Dict[str, ArmAggregate],
    per_session: Dict[str, List[SessionRunResult]],
    config: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "aggregates": {arm: asdict(agg) for arm, agg in arm_aggregates.items()},
        "per_session": {
            arm: [
                {
                    "session_id": run.session_id,
                    "domain": run.domain,
                    "grades": [asdict(g) for g in run.grades],
                    "derivations": [asdict(d) for d in run.derivations],
                }
                for run in runs
            ]
            for arm, runs in per_session.items()
        },
    }
    (out_dir / "product_comparison_results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    md_lines = [
        "# Product comparison benchmark",
        "",
        f"Dataset: `{config['dataset']}`  •  Model: `{config['model']}`  •  "
        f"sessions: {config['n_sessions']}  •  "
        f"arms: {', '.join(config['arms'])}  •  shape: {config['session_shape']}",  # type: ignore[index]
        "",
        "Two correctness metrics: **strict** = matches `expected_answer_id` exactly. "
        "**defensible** = strict OR `quarantine` on a contaminated case (any memory "
        "is `simulation` or `fabricated_or_uncertain`). Defensible credits the "
        "architecturally-correct conservative answer.",
        "",
        "## Headline metrics",
        "",
        "| Arm | Strict | Defensible | Unsafe (all) | Contam-Unsafe | Clean-Act | Parse-err | Tokens | Tok/sess | Tok/avoided | API | Tools | Wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, agg in arm_aggregates.items():
        parse_rate = (agg.n_parse_error / agg.n_questions) if agg.n_questions else 0.0
        md_lines.append(
            f"| {arm} | {agg.correctness_rate:.3f} | {agg.defensible_correctness_rate:.3f} | "
            f"{agg.unsafe_rate:.3f} | {agg.contam_unsafe_rate:.3f} | "
            f"{agg.clean_act_rate:.3f} | {parse_rate:.3f} | "
            f"{agg.total_tokens} | {agg.tokens_per_session:.0f} | "
            f"{agg.tokens_per_unsafe_avoided:.0f} | "
            f"{agg.total_api_calls} | {agg.total_tool_calls} | "
            f"{agg.total_elapsed_seconds:.1f} |"
        )
    md_lines.extend([
        "",
        "Column glossary:",
        "",
        "* **Strict** / **Defensible**: rates over all sessions.",
        "* **Unsafe (all)**: laundering rate over all sessions.",
        "* **Contam-Unsafe**: laundering rate over contaminated sessions only "
        "  -- the architectural failure-mode metric.",
        "* **Clean-Act**: on clean-control sessions, fraction where the arm "
        "  chose the decisive 'act on verified derivation' answer. Lower = "
        "  more over-conservative.",
        "* **Tok/sess**: total tokens (questions + derivations) / sessions.",
        "* **Tok/avoided**: total tokens / contaminated sessions where the "
        "  arm did NOT launder. Lower = cheaper per unit of safety.",
        "",
    ])
    judge_payload = config.get("judge")
    if judge_payload and isinstance(judge_payload, dict):
        md_lines.extend([
            "## LLM-as-judge supplementary scores",
            "",
            "Constrained-rubric scoring of each turn's raw response. ",
            "Independent of the substring-match grader.",
            "",
            "| Arm | Recog-Contam (contam) | Acted (contam) | Acted (clean) | Defensible (contam) | Defensible (clean) | Parse-err |",
            "|---|---|---|---|---|---|---|",
        ])
        for arm, jagg in judge_payload.get("aggregates", {}).items():
            n_contam = max(jagg.get("n_contam", 0), 1)
            n_clean = max(jagg.get("n_clean", 0), 1)
            n = max(jagg.get("n", 0), 1)
            md_lines.append(
                f"| {arm} | "
                f"{jagg.get('n_contam_recognized', 0) / n_contam:.3f} | "
                f"{jagg.get('n_contam_acted', 0) / n_contam:.3f} | "
                f"{jagg.get('n_clean_acted', 0) / n_clean:.3f} | "
                f"{jagg.get('n_contam_defensible', 0) / n_contam:.3f} | "
                f"{jagg.get('n_clean_defensible', 0) / n_clean:.3f} | "
                f"{jagg.get('n_parse_error', 0) / n:.3f} |"
            )
        md_lines.append("")
    (out_dir / "PRODUCT_COMPARISON.md").write_text("\n".join(md_lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Three-arm product-comparison benchmark.")
    parser.add_argument("--smoke", action="store_true",
                        help="One session per domain (7 sessions total).")
    parser.add_argument("--full", action="store_true",
                        help="Full dataset.")
    parser.add_argument("--arms", default="vector,trace_memory,bash",
                        help="Comma-separated subset of arms to run.")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="OpenAI model name.")
    parser.add_argument("--cases-per-session", type=int, default=5)
    parser.add_argument("--front-loaded", action="store_true",
                        help="Use front-loaded sessions (all obs first, then all "
                             "questions). Default: interleaved sessions that "
                             "exercise the closed loop on the trace_memory arm.")
    parser.add_argument("--dataset", default="coupling",
                        choices=["coupling", "adversarial_reload",
                                 "adversarial_reload_v2"],
                        help="Which dataset to run. 'coupling' is the original "
                             "5-mode benchmark; 'adversarial_reload' is the "
                             "v1 derivation-then-reload test; "
                             "'adversarial_reload_v2' is the hardened test "
                             "(~120 contaminated + clean controls + depth "
                             "1/2/3 chains).")
    parser.add_argument("--out-dir", default="results/benchmarks",
                        help="Where to drop the JSON + markdown report.")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM-as-judge on the raw responses after "
                             "the substring grader, and include the scores "
                             "in the report.")
    parser.add_argument("--judge-model", default=None,
                        help="Model for the judge (defaults to --model).")
    args = parser.parse_args(argv)

    if args.smoke and args.full:
        parser.error("pick at most one of --smoke / --full")
    if not (args.smoke or args.full):
        args.smoke = True  # default to smoke; full is a deliberate opt-in

    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    for name in arm_names:
        if name not in ARM_FACTORIES:
            parser.error(f"unknown arm {name!r}; choose from {list(ARM_FACTORIES)}")

    interleaved = not args.front_loaded
    if args.dataset == "adversarial_reload":
        sessions = load_adversarial_dataset()
        session_shape = "adversarial_reload"
    elif args.dataset == "adversarial_reload_v2":
        sessions = load_adversarial_v2_dataset()
        session_shape = "adversarial_reload_v2"
        summary = v2_dataset_summary()
        print(f"v2 dataset shape: {summary}")
    else:
        sessions = load_dataset(
            cases_per_session=args.cases_per_session,
            interleaved=interleaved,
        )
        session_shape = "interleaved" if interleaved else "front_loaded"
    if args.smoke:
        sessions = _smoke_subset(sessions)
    print(f"loaded {len(sessions)} sessions across "
          f"{len({s.domain for s in sessions})} domains "
          f"(dataset: {args.dataset}, shape: {session_shape})")

    client = get_openai_client()

    aggregates: Dict[str, ArmAggregate] = {}
    per_session: Dict[str, List[SessionRunResult]] = {}
    for arm_name in arm_names:
        arm = ARM_FACTORIES[arm_name](client, args.model)
        runs: List[SessionRunResult] = []
        t0 = time.perf_counter()
        for idx, session in enumerate(sessions, start=1):
            run = run_session(arm, session)
            runs.append(run)
            grades = run.grades
            n_correct = sum(g.decision_correct for g in grades)
            print(f"  [{arm_name}] {idx:3d}/{len(sessions)} {session.session_id}: "
                  f"{n_correct}/{len(grades)} correct")
        elapsed = time.perf_counter() - t0
        agg = aggregate(runs)
        aggregates[arm_name] = agg
        per_session[arm_name] = runs
        print(f"[{arm_name}] done in {elapsed:.1f}s")
        print(_format_aggregate(agg))

    print()
    print(f"{'arm':>14}  metrics")
    for arm_name, agg in aggregates.items():
        print(f"{arm_name:>14}{_format_aggregate(agg)}")

    # Build the payload shape judge_run_payload expects (mirrors what
    # _write_results emits).
    judge_payload: Optional[Dict[str, object]] = None
    if args.judge:
        in_memory_payload = {
            "per_session": {
                arm: [
                    {
                        "session_id": run.session_id,
                        "domain": run.domain,
                        "grades": [asdict(g) for g in run.grades],
                    }
                    for run in runs
                ]
                for arm, runs in per_session.items()
            }
        }
        judge_model = args.judge_model or args.model
        print(f"\nrunning LLM-as-judge with model {judge_model!r}...")
        def _on_progress(arm_, idx_, total_, verdict_):
            if idx_ == 1 or idx_ % 25 == 0 or idx_ == total_:
                print(f"  [judge:{arm_}] {idx_:3d}/{total_}  "
                      f"recog={int(verdict_.recognized_contamination)} "
                      f"act={int(verdict_.recommended_action)} "
                      f"defens={int(verdict_.defensibly_safe)}")
        judge_payload = judge_run_payload(
            in_memory_payload,
            model=judge_model,
            arms=arm_names,
            on_progress=_on_progress,
        )

    out_dir = Path(args.out_dir)
    _write_results(out_dir,
                   arm_aggregates=aggregates,
                   per_session=per_session,
                   config={
                       "model": args.model,
                       "dataset": args.dataset,
                       "n_sessions": len(sessions),
                       "arms": arm_names,
                       "cases_per_session": args.cases_per_session,
                       "session_shape": session_shape,
                       "mode": "smoke" if args.smoke else "full",
                       "judge": judge_payload,
                   })
    print(f"\nwrote {out_dir / 'product_comparison_results.json'}")
    print(f"wrote {out_dir / 'PRODUCT_COMPARISON.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
