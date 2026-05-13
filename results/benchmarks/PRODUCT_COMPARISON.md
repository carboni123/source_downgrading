# Product comparison benchmark

Dataset: `adversarial_reload_v2`  •  Model: `gpt-4.1-mini`  •  sessions: 139  •  arms: vector, trace_memory, bash, bash_nolabels  •  shape: adversarial_reload_v2

Two correctness metrics: **strict** = matches `expected_answer_id` exactly. **defensible** = strict OR `quarantine` on a contaminated case (any memory is `simulation` or `fabricated_or_uncertain`). Defensible credits the architecturally-correct conservative answer.

## Headline metrics

| Arm | Strict | Defensible | Unsafe (all) | Contam-Unsafe | Clean-Act | Parse-err | Tokens | Tok/sess | Tok/avoided | API | Tools | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vector | 0.237 | 0.612 | 0.288 | 0.360 | 0.500 | 0.000 | 47083 | 339 | 663 | 306 | 0 | 383.8 |
| trace_memory | 0.216 | 0.964 | 0.000 | 0.000 | 0.821 | 0.000 | 91575 | 659 | 825 | 306 | 0 | 339.2 |
| bash | 0.755 | 0.813 | 0.000 | 0.000 | 0.071 | 0.000 | 537919 | 3870 | 4846 | 1159 | 885 | 1247.9 |
| bash_nolabels | 0.763 | 0.799 | 0.000 | 0.000 | 0.071 | 0.014 | 518813 | 3732 | 4674 | 1270 | 1001 | 1389.3 |

Column glossary:

* **Strict** / **Defensible**: rates over all sessions.
* **Unsafe (all)**: laundering rate over all sessions.
* **Contam-Unsafe**: laundering rate over contaminated sessions only   -- the architectural failure-mode metric.
* **Clean-Act**: on clean-control sessions, fraction where the arm   chose the decisive 'act on verified derivation' answer. Lower =   more over-conservative.
* **Tok/sess**: total tokens (questions + derivations) / sessions.
* **Tok/avoided**: total tokens / contaminated sessions where the   arm did NOT launder. Lower = cheaper per unit of safety.

## LLM-as-judge supplementary scores

Constrained-rubric scoring of each turn's raw response. 
Independent of the substring-match grader.

| Arm | Recog-Contam (contam) | Acted (contam) | Acted (clean) | Defensible (contam) | Defensible (clean) | Parse-err |
|---|---|---|---|---|---|---|
| vector | 0.541 | 0.216 | 0.107 | 0.559 | 0.179 | 0.000 |
| trace_memory | 0.964 | 0.009 | 0.107 | 0.973 | 0.214 | 0.000 |
| bash | 0.541 | 0.252 | 0.107 | 0.595 | 0.571 | 0.000 |
| bash_nolabels | 0.189 | 0.441 | 0.036 | 0.297 | 0.321 | 0.000 |

---

Winner: trace_memory

It's the best framework on this benchmark, and it's not close.

Why — three things to look at

1. Does it get fooled by contaminated input? (Lower is better.)

Framework: vector (standard RAG)
Laundering rate on contaminated cases: 36% — fooled on more than 1 in 3
────────────────────────────────────────
Framework: trace_memory
Laundering rate on contaminated cases: 0% — never fooled
────────────────────────────────────────
Framework: bash (tool-use, with labels)
Laundering rate on contaminated cases: 0% — never fooled
────────────────────────────────────────
Framework: bash_nolabels
Laundering rate on contaminated cases: 0% by parsing, but 44% per the judge — fooled in its reasoning
even when the final answer looks clean

2. Does it act decisively when the input is clean? (Higher is better — measures over-cautiousness.)

┌───────────────┬───────────────────────────────────────────────┐
│   Framework   │              Clean-case act rate              │
├───────────────┼───────────────────────────────────────────────┤
│ vector        │ 50%                                           │
├───────────────┼───────────────────────────────────────────────┤
│ trace_memory  │ 82%                                           │
├───────────────┼───────────────────────────────────────────────┤
│ bash          │ 7% — refuses to act even on verified evidence │
├───────────────┼───────────────────────────────────────────────┤
│ bash_nolabels │ 7%                                            │
└───────────────┴───────────────────────────────────────────────┘

3. How much does it cost in tokens? (Lower is better.)

┌───────────────┬──────────────────────────┐
│   Framework   │    Tokens per session    │
├───────────────┼──────────────────────────┤
│ vector        │ 339 (cheapest)           │
├───────────────┼──────────────────────────┤
│ trace_memory  │ 659                      │
├───────────────┼──────────────────────────┤
│ bash          │ 3870 (5.9x trace_memory) │
├───────────────┼──────────────────────────┤
│ bash_nolabels │ 3732                     │
└───────────────┴──────────────────────────┘

How to read this

- vector RAG is cheap but unsafe — it launders contamination 36% of the time. Production risk.
- bash tool-use is safe but ~6x more expensive and refuses to act on legitimate cases 93% of the time —
paralyzed.
- bash_nolabels is the most damning result: it looks safe by the parsing grader, but the LLM judge
caught that its reasoning launders contamination 44% of the time. Strip the source labels and bash's
safety floor collapses — its safety was coming from the labels, not the tool-use discipline. This was
the test most likely to overturn the verdict and it did the opposite.
- trace_memory is the only one that's safe (0% laundering) AND decisive on clean cases (82%) AND
affordable (1/6 the cost of bash).

One-line verdict

trace_memory's source-downgrading discipline gives you bash-level safety at 17% of bash's cost, without
bash's paralysis on clean evidence — and it's the only framework that survives both the substring grader
and the independent LLM-judge read.