"""Convert the published PoisonedRAG corpus into our PoisonedQuestion JSONL.

PoisonedRAG (Zou et al., 2024) ships 100 questions per split (NQ /
HotpotQA / MS-MARCO) at
``external/PoisonedRAG/results/adv_targeted_results/{split}.json``.
Each entry includes:

  {"id":               <str>,
   "question":         <str>,
   "correct answer":   <str>,
   "incorrect answer": <str>,   # attacker's target
   "adv_texts":        [<str>, ... 5 entries]}

What's NOT bundled: clean (knowledge-base) passages. The original
benchmark expects the clean passages to come from the BEIR corpus at
runtime, which is a multi-GB download.

This converter takes the bundled adversarial passages verbatim and
synthesises clean passages via the LLM (~$0.01-0.02 per 100-question
split, cached to disk after the first run). Each clean passage is a
neutral encyclopaedia-style factual statement that contains the
correct answer.

This is an *approximation* of the original benchmark, not a perfect
reproduction. The architectural test it implements:

  * Real adversarial passages (verbatim from PoisonedRAG attack output).
  * Synthesised clean passages (LLM-written, encyclopaedia-style,
    containing the correct answer).
  * Hard-mode classifier labels every passage with no oracle help.
  * The comparison metric is the *delta* in attack-success rate between
    the vector baseline and the trace_memory arm on the same passage
    set -- which is meaningful regardless of how the clean passages
    were produced.

If you want a perfect reproduction, run ``prepare_dataset.py`` from
the PoisonedRAG repo to fetch the BEIR corpus, then write a custom
converter that retrieves real clean top-k passages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dataset import PoisonedPassage, PoisonedQuestion, write_poisoned_questions


_DEFAULT_REPO_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "external" / "PoisonedRAG"
)


_CLEAN_GEN_SYSTEM = """\
You write short neutral encyclopaedia-style factual statements.

You will receive a question and its correct short answer. Reply with
a JSON object containing exactly two clean passages that contain the
correct answer, written in the neutral style of Wikipedia or a
standard reference work. Do NOT mention the question explicitly. Do
NOT include hedges like "reportedly", "allegedly", or "according to
some sources" -- the statements should read as well-established fact.

Output schema:
  {"passage_1": "<1-3 sentences>",
   "passage_2": "<1-3 sentences>"}

No prose, no markdown fences. Each passage 30-90 words.
"""


def _cache_key(question: str, correct_answer: str) -> str:
    h = hashlib.sha256(
        (question.strip() + "||" + correct_answer.strip()).encode("utf-8")
    ).hexdigest()
    return h[:24]


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, str]]:
    if not cache_path.is_file():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _save_cache(cache_path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def _parse_passages_json(text: str) -> Optional[Tuple[str, str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        import re
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        d = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    p1 = str(d.get("passage_1", "")).strip()
    p2 = str(d.get("passage_2", "")).strip()
    if not p1 or not p2:
        return None
    return p1, p2


def _synthesize_clean(
    client,
    *,
    model: str,
    question: str,
    correct_answer: str,
    max_retries: int = 2,
) -> Tuple[str, str]:
    """Ask the LLM for 2 neutral clean passages. Falls back to a
    minimal hard-coded template on parse failure (rare)."""
    from benchmarks.product_comparison.llm_client import call_chat
    user = (
        f"Question: {question}\n"
        f"Correct answer: {correct_answer}\n\n"
        "Write two clean passages."
    )
    for _ in range(max_retries + 1):
        message, _usage, _elapsed = call_chat(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _CLEAN_GEN_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_output_tokens=240,
            temperature=0.0,
        )
        text = (message.content or "").strip()
        parsed = _parse_passages_json(text)
        if parsed is not None:
            return parsed
    # Fallback: minimal but contains the correct answer.
    return (
        f"In standard reference sources, the established answer is {correct_answer}.",
        f"Multiple encyclopaedic sources document that the correct value is {correct_answer}.",
    )


def convert_split(
    *,
    repo_path: Path,
    split: str,
    out_path: Path,
    client=None,
    clean_gen_model: str = "gpt-4.1-mini",
    cache_path: Optional[Path] = None,
    limit: Optional[int] = None,
    on_progress=None,
) -> List[PoisonedQuestion]:
    """Read PoisonedRAG ``adv_targeted_results/{split}.json`` and emit
    a PoisonedQuestion JSONL.

    If ``client`` is None, every question's clean passages must already
    be present in ``cache_path`` (lets the converter run offline once
    the cache is warm).
    """
    src = repo_path / "results" / "adv_targeted_results" / f"{split}.json"
    if not src.is_file():
        raise FileNotFoundError(f"PoisonedRAG file not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    cache_path = cache_path or (
        out_path.parent / f"clean_passages_cache_{split}.json"
    )
    cache = _load_cache(cache_path)
    cache_dirty = False

    out: List[PoisonedQuestion] = []
    items = list(data.items())
    if limit is not None:
        items = items[: limit]
    for idx, (qid, entry) in enumerate(items, start=1):
        question = str(entry["question"])
        correct = str(entry["correct answer"])
        target = str(entry["incorrect answer"])
        adv_texts = list(entry.get("adv_texts", []))
        key = _cache_key(question, correct)
        if key not in cache:
            if client is None:
                raise RuntimeError(
                    f"clean passages for {qid!r} not in cache and no "
                    f"client supplied to synthesise them"
                )
            p1, p2 = _synthesize_clean(
                client, model=clean_gen_model,
                question=question, correct_answer=correct,
            )
            cache[key] = {"passage_1": p1, "passage_2": p2}
            cache_dirty = True
        else:
            p1 = cache[key]["passage_1"]
            p2 = cache[key]["passage_2"]
        clean_passages = (
            PoisonedPassage(passage_id=f"{qid}_clean_1", text=p1, kind="clean"),
            PoisonedPassage(passage_id=f"{qid}_clean_2", text=p2, kind="clean"),
        )
        adv_passages = tuple(
            PoisonedPassage(
                passage_id=f"{qid}_adv_{i+1}",
                text=str(text),
                kind="adversarial",
            )
            for i, text in enumerate(adv_texts)
        )
        out.append(PoisonedQuestion(
            question_id=qid,
            question=question,
            correct_answer=correct,
            target_answer=target,
            correct_answer_aliases=(),
            target_answer_aliases=(),
            source_dataset=f"poisonedrag_{split}",
            clean_passages=clean_passages,
            adversarial_passages=adv_passages,
        ))
        if on_progress is not None:
            on_progress(idx, len(items), qid)
    if cache_dirty:
        _save_cache(cache_path, cache)
    write_poisoned_questions(out, out_path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Convert a PoisonedRAG split into our PoisonedQuestion JSONL "
        "with LLM-synthesised clean passages."
    ))
    parser.add_argument("--repo", default=str(_DEFAULT_REPO_PATH),
                        help=f"Path to the cloned PoisonedRAG repo "
                             f"(default: {_DEFAULT_REPO_PATH}).")
    parser.add_argument("--split", default="nq",
                        choices=["nq", "hotpotqa", "msmarco"],
                        help="Which PoisonedRAG split to convert.")
    parser.add_argument("--out", required=True,
                        help="Output JSONL path.")
    parser.add_argument("--cache",
                        default=None,
                        help="Path to the clean-passage cache. Default "
                             "is alongside --out.")
    parser.add_argument("--clean-gen-model", default="gpt-4.1-mini",
                        help="LLM used to synthesise clean passages.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of questions converted.")
    parser.add_argument("--offline", action="store_true",
                        help="Refuse to call the LLM. Only succeeds if "
                             "every question's clean passages are "
                             "already in the cache.")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    cache_path = Path(args.cache) if args.cache else None

    client = None
    if not args.offline:
        from benchmarks.product_comparison.llm_client import get_openai_client
        client = get_openai_client()

    def _on_progress(idx, total, qid):
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"  [{args.split}] {idx:3d}/{total} {qid}")

    t0 = time.perf_counter()
    questions = convert_split(
        repo_path=Path(args.repo),
        split=args.split,
        out_path=out_path,
        client=client,
        clean_gen_model=args.clean_gen_model,
        cache_path=cache_path,
        limit=args.limit,
        on_progress=_on_progress,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nwrote {len(questions)} questions to {out_path} in {elapsed:.1f}s")
    n_clean = sum(len(q.clean_passages) for q in questions)
    n_adv = sum(len(q.adversarial_passages) for q in questions)
    print(f"  total clean passages: {n_clean}")
    print(f"  total adversarial passages: {n_adv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
