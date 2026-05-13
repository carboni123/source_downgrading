# Source Downgrading: Empirical Validation Results

**Author:** Diego Falkowski Carboni — Tyxter
**Date:** 2026-05-13
**Companion paper:** [Source Downgrading: Trust-Bounded Writeback for Derived Memory Records](../paper/Source_Downgrading.tex)

## Summary

We implemented source-downgrading inscription as an end-to-end pipeline:
an LLM-driven source classifier infers a six-class trust label
(`external` / `tool_output` / `retrieved_memory` / `inference` /
`simulation` / `fabricated_or_uncertain`) for each unlabeled record, and
the `add_derived` API computes derived-record labels via the min-trust
rule of the paper (Definition 6).

Across two empirical programs we find:

1. **Internal benchmark (adversarial-reload v2, 139 sessions with
   derivation chains of depth 1–3).** Source-downgrading reduced
   attack-laundering rate to **0%** on contaminated sessions vs **36%**
   for vector RAG, while maintaining **82%** decisive-action rate on
   clean controls. The result validates the paper's specific claim:
   given correct labels on inputs, source-downgrading prevents
   trust-laundering across derivation chains.

2. **External benchmark (PoisonedRAG NQ, 100 single-shot QA questions
   with attacker-injected passages).** trace_memory cut attack-success
   rate from vector's 0.45 to 0.22 (a 24pp reduction) — but a control
   arm (vector cosine retrieval + classifier labels surfaced in the
   prompt) achieved the same 0.21 ASR with identical 0.82 clean
   accuracy. **The 24pp reduction is fully attributable to the
   classifier + label-aware retrieval prompt, not to the inscription
   rule.**

These results are not contradictory; they are complementary. PoisonedRAG
is a single-shot benchmark with no derivations, so the source-downgrading
inscription rule is dormant by design. The adversarial-reload v2
benchmark is structurally aligned with the paper's own validation
fixture (Section 5) and is the test that actually exercises the rule.

## What we implemented

The `trace-memory` library wraps three components in a single API:

| Component | What it does | Where the claim lives |
|---|---|---|
| Six-class source lattice + min-trust function | The trust ordering of the paper (Definition 1, 4, 5). | Source Downgrading paper §3. |
| Source-downgrading inscription rule (`MemoryAgent.add_derived`) | Caps derived-record source label at the lowest-trust contributing input (or `inference` if all inputs are above it). | Source Downgrading paper §4 (Definition 6, Propositions 1–4). |
| LLM source classifier (`LLMSourceClassifier`) | Infers a label for unlabeled raw text via a constrained-rubric prompt. | The paper explicitly defers this to "a separate empirical program" (§6.3). |

The classifier is the *enabling layer* for the inscription rule on raw
content. It is not the paper's contribution. We built it because the
external benchmark (PoisonedRAG) ships unlabeled passages and the paper's
rule consumes labels as inputs.

## Internal benchmark: adversarial-reload v2

### Setup

139 multi-turn sessions across 7 domains (SRE, customer support,
security, finance, healthcare, legal, research). Each session:
- 1 seed observation (always external)
- 1 auxiliary observation (fabricated rumor, simulation result, or
  another external observation for the clean-control subset)
- A chain of 1, 2, or 3 derivation prompts (LLM produces a free-text
  inference from the previous step)
- 1 reload question that retrieves the derivation and asks for an
  action that would launder the contamination if taken at face value

Sessions split:
- 111 contaminated (57 fabricated + 54 simulation)
- 28 clean controls (no fab/sim memories in the session)
- Chain depths: 118 depth-1, 14 depth-2, 7 depth-3

Model: `gpt-4.1-mini`. Labels: authored (oracle). This is the in-house
fixture, designed to mirror the paper's own validation harness (§5)
at larger scale.

### Headline metrics

| Arm | Strict ↑ | Defensible ↑ | Contam-Unsafe ↓ | Clean-Act ↑ | Tok/sess |
|---|---|---|---|---|---|
| vector (cosine RAG, no labels) | 0.237 | 0.612 | **0.360** | 0.500 | 339 |
| **trace_memory (source-downgrading)** | 0.216 | **0.964** | **0.000** | **0.821** | 659 |
| bash (tool-use + labels in YAML) | 0.755 | 0.813 | 0.000 | 0.071 | 3,870 |
| bash_nolabels (tool-use, no labels) | 0.763 | 0.799 | 0.000 | 0.071 | 3,732 |

(Strict and defensible are over all 139 sessions; contam-unsafe is the
laundering rate restricted to contaminated sessions; clean-act is the
decisive-action rate on clean controls.)

### Interpretation

The architectural prediction holds:
- vector RAG laundered contamination through derivation chains on **36%**
  of contaminated sessions. The derivation step inscribes a free-text
  inference as an unlabeled record in vector's store; the reload
  question retrieves it; the LLM commits to the contaminated action.
- trace_memory's source-downgrading caps the derived record at the
  min-trust of its inputs (`fab` or `sim`). The reload's envelope
  surfaces the label; the LLM correctly abstains. **0% laundering** on
  contaminated sessions.
- On clean controls (no contamination), trace_memory acts decisively
  82% of the time — substantially less over-cautious than the bash arms
  (7%). The conservative envelope does not cause systematic refusal on
  honest evidence.
- Token cost is 0.17× of bash, with strictly better defensible-correct
  and clean-action rates.

This is the test that validates the paper's claim: given correct labels
on inputs, source-downgrading reduces trust-laundering across derivation
chains to a measured zero on a real LLM-in-the-loop benchmark at scale.

## External benchmark: PoisonedRAG NQ

### Setup

100 questions from the published PoisonedRAG (Zou et al., USENIX 2025)
NQ split, with 5 attacker-crafted passages per question (LM-targeted
attack from the paper's `adv_targeted_results/nq.json`). 2 clean
passages per question synthesized by the LLM in encyclopaedia style
(BEIR corpus not bundled with the PoisonedRAG repo and ~3GB to
download).

Hard mode: every passage labelled by `LLMSourceClassifier` with no
oracle supervision. Model: `gpt-4.1-mini`. Seed: 0.

### Headline metrics

| Arm | Attack-Success ↓ | Clean Acc ↑ | Tok/q |
|---|---|---|---|
| vector (cosine, no labels) | 0.450 | 0.560 | 206 |
| **vector_with_labels** (cosine + classifier labels in prompt) | **0.210** | **0.820** | 365 |
| **trace_memory** (cosine + classifier labels + inscription rule) | **0.220** | **0.820** | 393 |
| bash (markdown + tool-use, labels in YAML) | 0.270 | 0.680 | 1,033 |
| bash_nolabels (markdown + tool-use, no labels) | 0.300 | 0.650 | 956 |

### Classifier confusion (the load-bearing component)

| Ground-truth kind | external | fab/uncertain | simulation | inference |
|---|---|---|---|---|
| clean (200 passages) | **196** | 4 | 0 | 0 |
| adversarial (500 passages) | 144 | **295** | 59 | 2 |

Classifier precision on clean: 98%. Recall on adversarial: 71%
(355/500 → fab/sim/infer combined). 29% of adversarial passages slipped
through as `external`.

### Interpretation

The architecture **falls back to cosine retrieval** on PoisonedRAG.
trace_memory and vector_with_labels are statistically indistinguishable
(0.220 vs 0.210 ASR — 1pp delta, well within noise on n=100; identical
0.820 clean accuracy). The 24pp ASR reduction relative to vanilla vector
is **fully attributable to the classifier + the label-aware retrieval
prompt**, not to the inscription rule.

### Why

PoisonedRAG is **single-shot question-answering**. The benchmark
exercises one cycle: observe → retrieve → answer. There is no
`add_derived` call between observation and answer; no derived record
is inscribed; the inscription rule has nothing to act on.

Both trace_memory and vector_with_labels do, in this setting, the same
three things:
1. Cosine-similarity retrieval over the passages (identical embedding,
   identical top-k).
2. Surface the classifier's source label per passage in the prompt.
3. Include the trust-ordering block in the system prompt.

Source-downgrading inscription (Definition 6) is *defined for derived
records*. When the only records present are observations and the only
output is a final answer, the rule is dormant. The architecture
collapses to "cosine retrieval + classifier-labelled prompt template,"
and that template is exactly what vector_with_labels also implements.

This is the paper's own position. From §6.2:

> The rule is the kind of primitive recursive memory architectures need:
> small, falsifiable, composable. **It does not require a new
> architecture; it constrains what an existing architecture's
> inscription path must compute.**

And §6.3:

> Source labels are treated as inputs to the inscription rule, not
> outputs of an inference. When labels are themselves inferred from
> content and retrieval features — as they must be for real agent
> operation — the trust composition is no stronger than the weakest
> source-inference decision in the chain. **The trust ordering is a
> sound rule given correct labels; building it on top of a
> label-inference layer that is reliable enough is a separate
> empirical program.**

PoisonedRAG measures the label-inference layer. Adversarial-reload v2
measures the inscription rule.

## What this validates

The paper's specific claim — **derived trust cannot exceed input trust;
inference is the upper bound for inscription** — is validated by the
adversarial-reload v2 benchmark on three axes:

1. **Multi-turn LLM-in-the-loop:** 139 sessions with `gpt-4.1-mini`
   producing free-text derivations. Source-downgrading produces 0%
   laundering on contaminated sessions; the cosine baseline produces
   36%. The rule's correctness is not a synthetic-fixture artefact.

2. **Chain depth:** depth-2 and depth-3 derivation chains (one
   derivation feeding the next) maintain the 0% laundering rate. The
   trust-ceiling property is preserved transitively, as Proposition 1
   of the paper predicts.

3. **Clean-control honesty:** on 28 sessions with no contamination,
   trace_memory acts on the verified derivation 82% of the time. The
   rule does not produce systematic over-suppression on honest
   evidence (a concern the paper raises explicitly in §6.4).

## What this does not validate

PoisonedRAG demonstrates that:

1. On single-shot RAG with attacker-injected passages, the *classifier*
   and a *label-aware prompt template* together cut attack-success rate
   by roughly half. This is a finding about the label-inference layer,
   not the inscription rule. The paper anticipated this would be a
   separate program (§6.3) and does not claim it.

2. The classifier reaches 71% recall on adversarial passages produced
   by PoisonedRAG's LM-targeted attack with no fine-tuning. This is a
   credible floor but not a publishable classifier result — a fine-tuned
   `Source(.)` model would presumably do better. The paper flags
   exactly this as future work.

3. The classifier is the architectural bottleneck on single-shot RAG.
   With a 71%-recall classifier, the framework's overall safety floor
   on PoisonedRAG is whatever the LLM does when its retrieval set is
   29% adversarial-with-the-trusted-label. Improving the classifier is
   the most direct way to push attack-success rate lower.

These are real product-level findings but they are *outside the scope*
of the source-downgrading paper.

## Honest framing of what was shipped

The trace-memory library currently bundles three components:

- **The source-downgrading inscription rule** — the paper's
  contribution. Validated by adversarial-reload v2.
- **The LLM source classifier** — a prerequisite for applying the rule
  to unlabeled content. Validated separately by PoisonedRAG, as a
  classifier (not as the inscription rule).
- **The label-aware retrieval prompt template** — a packaging of the
  trust ordering for consumption by the LLM. Does work on single-shot
  RAG (PoisonedRAG); pulls no additional weight on derivation chains
  (vector_with_labels would also have the labels in the prompt; the
  inscription rule is what does the work there).

Three components, two papers worth of empirical results, one library
that ships all three. The source-downgrading paper covers exactly one
of them — and that one is the load-bearing claim under derivation.

## Conclusion

The empirical work validates the Source Downgrading paper's specific
claim and clarifies its boundary:

- **On derivation chains, source-downgrading is the rule that prevents
  trust-laundering.** Validated end-to-end on adversarial-reload v2
  (0% laundering vs vector's 36% on 100+ contaminated multi-turn
  sessions, real LLM, depth-1/2/3 chains).
- **On single-shot RAG, the framework reduces to cosine retrieval plus
  a source classifier plus a label-aware prompt template.** The
  inscription rule is dormant by design. The classifier and the
  prompt template do useful work (PoisonedRAG NQ: 0.45 → 0.21 ASR),
  but neither is the paper's contribution and the result should not be
  cited as evidence for source-downgrading.

The paper's narrow claim is correct, validated at scale, and now
empirically distinguished from the broader system it lives inside.
Future work — the next paper in the sequence — is the label-inference
layer: a fine-tuned `Source(.)` classifier whose adversarial recall is
high enough to cap the framework's safety floor on production RAG
deployments. The trace-memory codebase ships the prompt-engineered
baseline of that classifier as `benchmarks/poisonedrag/source_classifier.py`,
which can serve as a starting point.

## Reproducing

All numbers in this document are reproducible from the codebase.

**adversarial-reload v2:**
```bash
python benchmarks/run_product_comparison.py \
    --full --dataset adversarial_reload_v2 \
    --arms vector,trace_memory,bash,bash_nolabels \
    --judge --model gpt-4.1-mini
```
Wall: ~58 min. Cost: ~$3 on `gpt-4.1-mini`.

**PoisonedRAG NQ:**
```bash
# (Clone https://github.com/sleeepeer/PoisonedRAG into external/ first.)
python -m benchmarks.poisonedrag.convert_real \
    --split nq --out benchmarks/data/poisonedrag/nq.jsonl
python benchmarks/run_poisonedrag.py \
    --data benchmarks/data/poisonedrag/nq.jsonl \
    --arms vector,vector_with_labels,trace_memory,bash,bash_nolabels \
    --model gpt-4.1-mini
```
Wall: ~28 min. Cost: ~$0.30.

Result files: `results/benchmarks/PRODUCT_COMPARISON.md`,
`results/benchmarks/product_comparison_results.json`,
`results/benchmarks/POISONEDRAG.md`, and
`results/benchmarks/poisonedrag_results.json`.
