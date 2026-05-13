# PoisonedRAG external-validation adapter

This benchmark validates the trace_memory **source-downgrading**
invariant against the publicly published **PoisonedRAG** (Zou et al.,
2024) adversarial-RAG corpus. It is the external test that complements
the in-house adversarial-reload v2 results.

## What it tests

PoisonedRAG injects attacker-crafted passages into a RAG corpus that
look authoritative but assert a target wrong answer. A naive RAG
retrieves them and the LLM commits to the attacker's target.

The architectural test for trace_memory is whether a *real* source
classifier can recognise these adversarial passages as
`fabricated_or_uncertain` *without any benchmark-side supervision*,
and whether the source-downgrading invariant on top of that classifier
output suppresses the attack-success rate.

## Hard mode

"Hard mode" means: the benchmark does not hand any oracle source
labels to the framework. Every passage is labelled by an LLM source
classifier (`benchmarks/poisonedrag/source_classifier.py`) before the
arms see it. The vector arm ignores the labels (the laundering
baseline); the bash arm includes them in YAML front matter (or omits
them in the `bash_nolabels` ablation); the trace_memory arm uses them
in `MemoryAgent.add(..., source=...)` so the reactivation envelope
surfaces each passage with its trust label.

The classifier itself is a single-prompt baseline. It is not fine-
tuned. A production deployment would replace it with the Cognivault
`Source(.)` classifier referenced in
`Closed_Loop_Mnestic_Agent_Architecture.md`.

## Wiring overview

```
PoisonedQuestion (clean + adversarial passages, expected answer, target answer)
    │
    ▼
LLMSourceClassifier (one LLM call per passage, hard mode)
    │   labels each passage: external | tool_output | ... | fabricated_or_uncertain
    ▼
┌──────────────┬──────────────────┬─────────────┬─────────────────┐
│ VectorPRArm  │ TraceMemoryPRArm │ BashPRArm   │ BashPRArm       │
│ (ignores     │ (labels in       │ (labels in  │ (no labels;     │
│  labels)     │  envelope)       │  YAML)      │  ablation)      │
└──────┬───────┴────────┬─────────┴──────┬──────┴──────────┬──────┘
       ▼                ▼                ▼                 ▼
   free-text       free-text       free-text         free-text
   answer          answer          answer            answer
       │                │                │                 │
       └────────────────┴────────────────┴─────────────────┘
                            │
                            ▼
                       grade_open_answer
                            │
                            ▼
                   ArmASR (attack-success-rate ↓,
                           clean-accuracy ↑,
                           refusal-rate)
```

## Layout

```
benchmarks/poisonedrag/
├── __init__.py
├── README.md                   # this file
├── dataset.py                  # PoisonedQuestion schema + JSONL loader + sample
├── source_classifier.py        # LLMSourceClassifier (hard-mode Source(.))
├── arms.py                     # VectorPRArm / TraceMemoryPRArm / BashPRArm
├── grading.py                  # attack-success / clean-accuracy / refusal
├── runner.py                   # end-to-end orchestration
└── _offline_smoke.py           # stub-client wiring test (no network)

benchmarks/run_poisonedrag.py   # CLI entry point
```

## Running

### Offline smoke (no network, ~1 second)

```bash
python -m benchmarks.poisonedrag._offline_smoke
```

Validates classifier JSON parsing, all 4 arm shapes, classifier
confusion-matrix wiring, and report writer.

### Sample run (12 built-in questions, ~$0.05)

```bash
python benchmarks/run_poisonedrag.py --smoke
```

The built-in 12 questions are hand-crafted in the PoisonedRAG format
(short factual answers, 2 clean + 3 adversarial passages each). This
is enough to validate the headline shape (does the classifier flag
adversarial passages, does trace_memory's attack-success rate fall
relative to vector) before paying for the real corpus.

### Full run against published PoisonedRAG data

1. Clone the PoisonedRAG repo:
   ```bash
   git clone https://github.com/sleeepeer/PoisonedRAG
   ```
2. Convert one of its splits (NQ, HotpotQA, MS-MARCO) into the
   `PoisonedQuestion` JSONL schema documented in `dataset.py`. Each
   line:
   ```json
   {
     "question_id": "nq_001",
     "question": "...",
     "correct_answer": "...",
     "target_answer": "...",
     "correct_answer_aliases": ["..."],
     "target_answer_aliases": ["..."],
     "source_dataset": "nq",
     "clean_passages": [
       {"passage_id": "nq_001_clean_1", "text": "..."},
       ...
     ],
     "adversarial_passages": [
       {"passage_id": "nq_001_adv_1", "text": "..."},
       ...
     ]
   }
   ```
3. Run:
   ```bash
   python benchmarks/run_poisonedrag.py \
       --data path/to/poisonedrag_nq.jsonl \
       --arms vector,trace_memory,bash,bash_nolabels \
       --model gpt-4.1-mini \
       --classifier-model gpt-4.1-mini \
       --limit 100
   ```

## Cost notes

Per question, hard mode costs roughly:

- Classifier: ~5 passages × ~120 tokens each = ~600 tokens
- vector arm: ~250 tokens
- trace_memory arm: ~350 tokens
- bash arm: ~3000-4000 tokens (tool-use rounds)
- bash_nolabels arm: ~3000-4000 tokens

So per question with all 4 arms + classifier: ~9000 tokens.
A 100-question run with `gpt-4.1-mini`: ~$0.50-$1.00.

To run cheaply for an initial signal, restrict to two arms:
`--arms vector,trace_memory` cuts cost by ~85%.

## Output

The runner writes:

- `results/benchmarks/poisonedrag_results.json` — full per-question JSON
  including each arm's raw response, the classifier's label per
  passage, and aggregate metrics.
- `results/benchmarks/POISONEDRAG.md` — markdown report with headline
  attack-success-rate table and the classifier confusion matrix.

## How to read the results

Three signals matter:

1. **Classifier confusion matrix.** Rows are ground-truth passage kinds
   (`clean` / `adversarial`); columns are the classifier's labels.
   Honest behaviour: clean passages → `external`; adversarial → mostly
   `fabricated_or_uncertain`. The classifier is the *only* way
   trace_memory learns which passages to downgrade. If the classifier
   fails to recognise adversarial passages, source-downgrading cannot
   help.

2. **Attack-Success-Rate (ASR), lower better.** Per-arm rate at which
   the response contains the attacker's target answer. The
   architectural prediction: trace_memory's ASR is lower than vector's
   by an amount proportional to the classifier's recall on adversarial
   passages.

3. **Clean accuracy, higher better.** Per-arm rate at which the
   response contains the correct answer. Over-cautious arms (refusing
   to answer at all) keep ASR low but tank clean accuracy too.

A useful summary number is the `ASR delta` between trace_memory and
vector at matched clean-accuracy levels — that isolates the
architectural contribution from the LLM's prior knowledge of the
question domain.
