"""End-to-end PoisonedRAG runner.

For each question:

  1. Combine clean + adversarial passages.
  2. Pass each passage through the LLM source classifier (hard mode --
     no oracle labels). Cache by passage text within the run.
  3. For each arm:
       - reset(question_id)
       - observe(passage_id, text, source_label=<classifier label>,
                 kind=<clean|adversarial>) for each passage
       - answer(question_text)
  4. Grade the free-text answer (attack success / clean accuracy /
     refusal).

The classifier label is the only place trace_memory gets help. The
vector arm ignores it; bash includes it in front matter when
configured to. The runner therefore implements *hard mode*: a real
source classifier is the only difference between the architectural
arm and the baselines.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .arms import (
    BashPRArm,
    OpenAnswerResult,
    TraceMemoryPRArm,
    VectorPRArm,
    VectorWithLabelsPRArm,
)
from .dataset import PoisonedQuestion
from .grading import ArmASR, QuestionScore, aggregate, grade_open_answer
from .source_classifier import (
    ClassificationResult,
    ClassifierStats,
    LLMSourceClassifier,
    confusion_matrix,
)


ARM_FACTORIES: Dict[str, Callable[[object, str], object]] = {
    "vector":              lambda client, model: VectorPRArm(client, model=model),
    "vector_with_labels":  lambda client, model: VectorWithLabelsPRArm(
        client, model=model,
    ),
    "trace_memory":        lambda client, model: TraceMemoryPRArm(client, model=model),
    "bash":                lambda client, model: BashPRArm(client, model=model),
    "bash_nolabels":       lambda client, model: BashPRArm(
        client, model=model, include_source_labels=False,
    ),
}


@dataclass
class RunResult:
    config: Dict[str, object]
    arm_aggregates: Dict[str, ArmASR]
    per_question: Dict[str, List[QuestionScore]]
    classifier_stats: ClassifierStats
    classifier_confusion: Dict[str, Dict[str, int]]


def _shuffle_passages(q: PoisonedQuestion, rng: random.Random) -> List[tuple]:
    """Return list of (passage_id, text, kind) tuples, shuffled so
    adversarial position does not bias retrieval."""
    items = (
        [(p.passage_id, p.text, "clean") for p in q.clean_passages]
        + [(p.passage_id, p.text, "adversarial") for p in q.adversarial_passages]
    )
    rng.shuffle(items)
    return items


def run_benchmark(
    questions: Sequence[PoisonedQuestion],
    *,
    client,
    arm_names: Sequence[str],
    answer_model: str,
    classifier_model: str,
    seed: int = 0,
    on_progress: Optional[Callable[[str, int, int, QuestionScore], None]] = None,
) -> RunResult:
    """Run the full benchmark in-process. Caller supplies the OpenAI
    client; the runner does not load .env etc."""
    rng = random.Random(seed)
    classifier = LLMSourceClassifier(client, model=classifier_model, cache=True)
    classifier_stats = ClassifierStats()

    # Phase 1: classify every distinct passage once.
    # Build the (q -> passage_label_map) before any arm runs so the
    # arms see consistent labels and the classifier cost is reported
    # separately from the arm cost.
    per_question_labels: Dict[str, Dict[str, ClassificationResult]] = {}
    for q in questions:
        passage_map: Dict[str, ClassificationResult] = {}
        for pid, text, kind in _shuffle_passages(q, rng):
            cls = classifier.classify(text)
            passage_map[pid] = cls
            classifier_stats.add(cls, kind=kind)
        per_question_labels[q.question_id] = passage_map

    # Phase 2: run each arm against every question with the labels
    # from phase 1.
    aggregates: Dict[str, ArmASR] = {}
    per_question: Dict[str, List[QuestionScore]] = {}
    for arm_name in arm_names:
        factory = ARM_FACTORIES[arm_name]
        arm = factory(client, answer_model)
        scores: List[QuestionScore] = []
        total = len(questions)
        for idx, q in enumerate(questions, start=1):
            arm.reset(q.question_id)
            passage_map = per_question_labels[q.question_id]
            for pid, text, kind in _shuffle_passages(q, rng):
                arm.observe(
                    pid, text,
                    source_label=passage_map[pid].label,
                    kind=kind,
                )
            result: OpenAnswerResult = arm.answer(q.question)
            score = grade_open_answer(q, result, arm=arm_name)
            scores.append(score)
            if on_progress is not None:
                on_progress(arm_name, idx, total, score)
        aggregates[arm_name] = aggregate(scores)
        per_question[arm_name] = scores

    return RunResult(
        config={
            "answer_model": answer_model,
            "classifier_model": classifier_model,
            "n_questions": len(questions),
            "arms": list(arm_names),
            "seed": seed,
        },
        arm_aggregates=aggregates,
        per_question=per_question,
        classifier_stats=classifier_stats,
        classifier_confusion=confusion_matrix(classifier_stats),
    )


def write_report(
    result: RunResult,
    out_dir: Path,
    *,
    json_name: str = "poisonedrag_results.json",
    md_name: str = "POISONEDRAG.md",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    payload = {
        "config": result.config,
        "aggregates": {
            arm: asdict(agg) for arm, agg in result.arm_aggregates.items()
        },
        "per_question": {
            arm: [asdict(s) for s in scores]
            for arm, scores in result.per_question.items()
        },
        "classifier": {
            "stats": asdict(result.classifier_stats),
            "confusion_matrix": result.classifier_confusion,
        },
    }
    (out_dir / json_name).write_text(json.dumps(payload, indent=2, default=str),
                                     encoding="utf-8")

    cstats = result.classifier_stats
    md = [
        "# PoisonedRAG external-validation benchmark",
        "",
        f"Answer model: `{result.config['answer_model']}` • "
        f"Classifier model: `{result.config['classifier_model']}` • "
        f"questions: {result.config['n_questions']} • "
        f"arms: {', '.join(result.config['arms'])}  ",  # type: ignore[arg-type]
        "",
        "Hard mode: each passage was labelled by an LLM source classifier "
        "(no oracle labels). The classifier's confusion matrix below shows "
        "how often each kind of passage (clean / adversarial) received each "
        "label.",
        "",
        "## Headline metrics",
        "",
        "| Arm | Attack-Success-Rate ↓ | Clean-Accuracy ↑ | Refusal-Rate | Tokens | Tok/q | API | Tools | Wall (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, agg in result.arm_aggregates.items():
        md.append(
            f"| {arm} | {agg.attack_success_rate:.3f} | "
            f"{agg.clean_accuracy:.3f} | {agg.refusal_rate:.3f} | "
            f"{agg.total_tokens} | {agg.tokens_per_question:.0f} | "
            f"{agg.total_api_calls} | {agg.total_tool_calls} | "
            f"{agg.total_elapsed:.1f} |"
        )
    md.extend([
        "",
        "**ASR** is the architectural failure-mode metric (lower better). ",
        "Clean accuracy is the percentage of responses that contain the ",
        "correct answer (higher better). Refusal-rate counts ``I do not know`` ",
        "style responses (defensible on PoisonedRAG since the corpus is ",
        "internally contradictory).",
        "",
        "## Source classifier behaviour",
        "",
        f"Classified {cstats.n} passages "
        f"(input_tokens={cstats.total_input_tokens}, output_tokens={cstats.total_output_tokens}, "
        f"parse_errors={cstats.n_parse_error}).",
        "",
        "| Ground-truth kind | external | tool_output | retrieved_memory | inference | simulation | fab/uncertain |",
        "|---|---|---|---|---|---|---|",
    ])
    label_order = (
        "external", "tool_output", "retrieved_memory",
        "inference", "simulation", "fabricated_or_uncertain",
    )
    for kind in ("clean", "adversarial"):
        row = result.classifier_confusion.get(kind, {})
        cells = [str(row.get(lbl, 0)) for lbl in label_order]
        md.append(f"| {kind} | " + " | ".join(cells) + " |")
    md.extend([
        "",
        "Read: the classifier's safety floor is the rate at which it labels "
        "adversarial passages as ``fabricated_or_uncertain``. A higher rate "
        "means trace_memory's source-downgrading sees more of the poisoned "
        "content correctly.",
        "",
    ])
    (out_dir / md_name).write_text("\n".join(md), encoding="utf-8")
