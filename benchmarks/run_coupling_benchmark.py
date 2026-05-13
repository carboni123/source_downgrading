"""Mnestic-attentional coupling benchmark runner.

The benchmark compares four deterministic policies:

- no_memory: answer from the prompt only.
- raw_memory: attend to the highest-salience memory without source checks.
- labels_only: read source labels for selection but do not use trace-memory
  writeback/provenance.
- trace_memory: select source-safe evidence and write the decision through
  MemoryAgent.add_derived(...).

Run:

    python benchmarks/run_coupling_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from benchmarks.coupling_dataset import (
        TRUSTED_SOURCES,
        TRUST_RANK,
        UNTRUSTED_SOURCES,
        CouplingCase,
        CouplingMemory,
        dataset_summary,
        make_dataset,
        validate_cases,
    )
else:
    from .coupling_dataset import (
        TRUSTED_SOURCES,
        TRUST_RANK,
        UNTRUSTED_SOURCES,
        CouplingCase,
        CouplingMemory,
        dataset_summary,
        make_dataset,
        validate_cases,
    )

from trace_memory import MemoryAgent, SourceLabel


VARIANTS = ("no_memory", "raw_memory", "labels_only", "trace_memory")


@dataclass(frozen=True)
class CouplingDecision:
    selected_answer_id: str
    selected_input_ids: Tuple[str, ...]
    selected_input_sources: Tuple[str, ...]
    derived_source: Optional[str]
    derived_provenance: Tuple[str, ...]
    write_attempted: bool


def _answer_content(case: CouplingCase, answer_id: str) -> str:
    for answer in case.answers:
        if answer.answer_id == answer_id:
            return answer.content
    raise KeyError(answer_id)


def _highest_salience(memories: Sequence[CouplingMemory]) -> CouplingMemory:
    return max(memories, key=lambda memory: memory.salience)


def _sources_for(case: CouplingCase, input_ids: Sequence[str]) -> Tuple[str, ...]:
    by_id = {memory.record_id: memory for memory in case.memories}
    return tuple(by_id[record_id].source for record_id in input_ids)


def _select_trusted_inputs(case: CouplingCase) -> Tuple[str, Tuple[str, ...]]:
    trusted = [memory for memory in case.memories if memory.source in TRUSTED_SOURCES]
    if not trusted:
        untrusted_ids = tuple(
            memory.record_id
            for memory in case.memories
            if memory.source in UNTRUSTED_SOURCES
        )
        return "quarantine", untrusted_ids

    selected_answer_id = _highest_salience(trusted).answer_id
    input_ids = tuple(
        memory.record_id
        for memory in case.memories
        if memory.answer_id == selected_answer_id and memory.source in TRUSTED_SOURCES
    )
    return selected_answer_id, input_ids


def _no_memory_decision(case: CouplingCase) -> CouplingDecision:
    return CouplingDecision(
        selected_answer_id=case.no_memory_answer_id,
        selected_input_ids=(),
        selected_input_sources=(),
        derived_source=None,
        derived_provenance=(),
        write_attempted=False,
    )


def _raw_memory_decision(case: CouplingCase) -> CouplingDecision:
    selected = _highest_salience(case.memories)
    return CouplingDecision(
        selected_answer_id=selected.answer_id,
        selected_input_ids=(selected.record_id,),
        selected_input_sources=(selected.source,),
        derived_source=SourceLabel.EXTERNAL.value,
        derived_provenance=(),
        write_attempted=True,
    )


def _labels_only_decision(case: CouplingCase) -> CouplingDecision:
    selected_answer_id, input_ids = _select_trusted_inputs(case)
    return CouplingDecision(
        selected_answer_id=selected_answer_id,
        selected_input_ids=input_ids,
        selected_input_sources=_sources_for(case, input_ids),
        derived_source=SourceLabel.INFERENCE.value,
        derived_provenance=input_ids,
        write_attempted=True,
    )


def _trace_memory_decision(case: CouplingCase) -> CouplingDecision:
    selected_answer_id, input_ids = _select_trusted_inputs(case)
    agent = MemoryAgent()
    for memory in case.memories:
        agent.add(
            memory.content,
            source=SourceLabel(memory.source),
            provenance=memory.provenance,
            record_id=memory.record_id,
        )

    derived = agent.add_derived(
        f"attention decision: {_answer_content(case, selected_answer_id)}",
        inputs=input_ids,
        record_id=f"{case.case_id}_attention_decision",
    )
    return CouplingDecision(
        selected_answer_id=selected_answer_id,
        selected_input_ids=input_ids,
        selected_input_sources=_sources_for(case, input_ids),
        derived_source=derived.source_label,
        derived_provenance=tuple(derived.provenance),
        write_attempted=True,
    )


def _variant_decision(variant: str, case: CouplingCase) -> CouplingDecision:
    if variant == "no_memory":
        return _no_memory_decision(case)
    if variant == "raw_memory":
        return _raw_memory_decision(case)
    if variant == "labels_only":
        return _labels_only_decision(case)
    if variant == "trace_memory":
        return _trace_memory_decision(case)
    raise ValueError(f"unknown coupling variant {variant!r}")


def _trust_violation(observed: Optional[str], expected: str) -> bool:
    if observed is None:
        return False
    return TRUST_RANK[observed] > TRUST_RANK[expected]


def _provenance_recall(observed: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return len(expected_set.intersection(observed)) / len(expected_set)


def _variant_metrics(variant: str, cases: Sequence[CouplingCase]) -> Dict:
    per_case = []
    correct = 0
    unsafe_choices = 0
    trust_violations = 0
    source_matches = 0
    write_attempts = 0
    recalls: List[float] = []
    by_mode_total: Dict[str, int] = {}
    by_mode_correct: Dict[str, int] = {}

    for case in cases:
        decision = _variant_decision(variant, case)
        is_correct = decision.selected_answer_id == case.expected_answer_id
        unsafe_choice = decision.selected_answer_id in case.unsafe_answer_ids
        trust_violation = _trust_violation(decision.derived_source, case.expected_source)
        provenance_recall = _provenance_recall(
            decision.derived_provenance,
            case.expected_provenance,
        )
        source_match = decision.derived_source == case.expected_source

        correct += int(is_correct)
        unsafe_choices += int(unsafe_choice)
        trust_violations += int(trust_violation)
        source_matches += int(source_match)
        write_attempts += int(decision.write_attempted)
        recalls.append(provenance_recall)
        by_mode_total[case.coupling_mode] = by_mode_total.get(case.coupling_mode, 0) + 1
        by_mode_correct[case.coupling_mode] = (
            by_mode_correct.get(case.coupling_mode, 0) + int(is_correct)
        )

        per_case.append({
            "case": asdict(case),
            "decision": asdict(decision),
            "correct": is_correct,
            "unsafe_choice": unsafe_choice,
            "trust_ceiling_violation": trust_violation,
            "source_match": source_match,
            "provenance_recall": provenance_recall,
        })

    n = len(cases)
    return {
        "variant": variant,
        "n_cases": n,
        "aggregate": {
            "decision_accuracy": correct / n if n else float("nan"),
            "unsafe_contamination_rate": unsafe_choices / n if n else float("nan"),
            "trust_ceiling_violation_rate": trust_violations / n if n else float("nan"),
            "source_match_rate": source_matches / write_attempts if write_attempts else 0.0,
            "provenance_recall": sum(recalls) / n if n else float("nan"),
            "write_attempt_rate": write_attempts / n if n else float("nan"),
            "per_mode_accuracy": {
                mode: by_mode_correct.get(mode, 0) / total
                for mode, total in sorted(by_mode_total.items())
            },
        },
        "per_case": per_case,
    }


def run_benchmark(cases: Sequence[CouplingCase]) -> Dict[str, Dict]:
    validate_cases(cases)
    return {variant: _variant_metrics(variant, cases) for variant in VARIANTS}


def render_report(results: Dict[str, Dict], cases: Sequence[CouplingCase]) -> str:
    summary = dataset_summary(cases)
    lines: List[str] = []
    lines.append("# Mnestic-Attentional Coupling Benchmark")
    lines.append("")
    lines.append(
        "Auto-generated by `benchmarks/run_coupling_benchmark.py`. "
        f"{len(cases)} labelled coupling cases; {len(VARIANTS)} variants."
    )
    lines.append(
        "This benchmark is deterministic and LLM-free. It validates the "
        "architecture boundary: whether memory records can condition an "
        "attention-like selector while preserving source ceilings and "
        "provenance through writeback."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Domains covered: {', '.join(sorted(summary['by_domain']))}.")
    lines.append(f"- Coupling modes: {', '.join(sorted(summary['by_mode']))}.")
    lines.append(
        "- Expected answers: "
        + ", ".join(
            f"{key}={summary['by_expected_answer'][key]}"
            for key in sorted(summary["by_expected_answer"])
        )
        + "."
    )
    lines.append(
        "- Expected writeback sources: "
        + ", ".join(
            f"{key}={summary['by_expected_source'][key]}"
            for key in sorted(summary["by_expected_source"])
        )
        + "."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "Higher is better for decision accuracy and provenance recall. Lower "
        "is better for unsafe contamination and trust-ceiling violations."
    )
    lines.append("")
    lines.append(
        "| Variant | Decision accuracy | Unsafe contamination | Trust violation | Provenance recall | Source match |"
    )
    lines.append("|---|---|---|---|---|---|")
    for variant in VARIANTS:
        agg = results[variant]["aggregate"]
        lines.append(
            f"| `{variant}` "
            f"| {agg['decision_accuracy']:.3f} "
            f"| {agg['unsafe_contamination_rate']:.3f} "
            f"| {agg['trust_ceiling_violation_rate']:.3f} "
            f"| {agg['provenance_recall']:.3f} "
            f"| {agg['source_match_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Per-mode Decision Accuracy")
    lines.append("")
    lines.append("| Mode | " + " | ".join(f"`{variant}`" for variant in VARIANTS) + " |")
    lines.append("|" + "---|" * (len(VARIANTS) + 1))
    modes = sorted(summary["by_mode"])
    for mode in modes:
        row = [f"| {mode}"]
        for variant in VARIANTS:
            row.append(f"{results[variant]['aggregate']['per_mode_accuracy'][mode]:.3f}")
        lines.append(" | ".join(row) + " |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "`trace_memory` succeeding here does not prove a live LLM will attend "
        "correctly. It proves a narrower primitive: once source-labelled "
        "records are made available to an attentional selector, the memory "
        "layer can support safe selection, quarantine contaminated support, "
        "and write the resulting decision without trust laundering."
    )
    lines.append(
        "`raw_memory` represents source-blind attention over memory salience. "
        "`labels_only` shows that source labels can improve selection but still "
        "miss the full writeback primitive: source ceilings and transitive "
        "provenance are only enforced by `MemoryAgent.add_derived(...)`."
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python benchmarks/coupling_dataset.py --output benchmarks/data/coupling_dataset.jsonl")
    lines.append("python benchmarks/run_coupling_benchmark.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/benchmarks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = make_dataset()
    validate_cases(cases)
    results = run_benchmark(cases)

    results_path = output_dir / "coupling_benchmark_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = output_dir / "COUPLING_BENCHMARK.md"
    report_path.write_text(render_report(results, cases), encoding="utf-8")

    print(f"running {len(VARIANTS)} variants x {len(cases)} cases...")
    print(f"wrote {results_path}")
    print(f"wrote {report_path}")
    print()
    print("Aggregate (accuracy/provenance higher is better; unsafe/violations lower is better):")
    print(
        f"{'variant':<20s} {'accuracy':>9s} {'unsafe':>9s} "
        f"{'violate':>9s} {'prov':>9s}"
    )
    for variant in VARIANTS:
        agg = results[variant]["aggregate"]
        print(
            f"{variant:<20s} "
            f"{agg['decision_accuracy']:>9.3f} "
            f"{agg['unsafe_contamination_rate']:>9.3f} "
            f"{agg['trust_ceiling_violation_rate']:>9.3f} "
            f"{agg['provenance_recall']:>9.3f}"
        )


if __name__ == "__main__":
    main()
