# Paper Revisions Plan — Source Downgrading

**Paper:** `paper/Source_Downgrading.tex` in the consolidated artifact repo.
**Triggered by:** [`source_downgrading_review.md`](./source_downgrading_review.md) — multi-hour deep-research peer review run 2026-05-13.
**Reviewer verdict:** *Publishable only with substantial revisions; not publishable as-is for a top-tier venue.*
**Central recommendation:** Reframe the paper from *"missing primitive"* to *"application and validation of a known IFC integrity-meet pattern, with an inference-typing rule, specialised to recursive agent memory."*

This document maps each reviewer point to a concrete change. It is the
work plan for v2 of the paper.

---

## Methodological note on the review's numbering

The review's §2 verdict table references "Definitions 1–8 and
Propositions 1–5". The current paper actually contains:

- 6 `definition` environments
- 2 `proposition` environments
- 3 `property` environments
- 1 `prediction` environment

The "1–8 / 1–5" wording came from the **prompt** Claude wrote (the
prompt mis-stated the numbering); the paper itself is internally
consistent. The reviewer correctly flagged this as a publication
hazard, but the affected verdict rows (Definition 8, Propositions 3/4/5)
are reviewing objects that **do not exist in the paper**. They should
be re-mapped to the existing Properties 1/2/3 and Prediction 1.

When applying the per-claim verdicts below, **use the paper's actual
numbering**:

| Reviewer label | Paper's actual object |
|---|---|
| Definition 5/6 | Definition 6 (source-downgrading inscription) |
| Definition 6/7 | Definition 7 (local laundering rate) — currently numbered 6 in `.tex`, verify |
| Definition 7/8 | Definition 8 (truth-grounded laundering rate) — verify numbering |
| Proposition 3 | Property 1 (failure of pure provenance propagation) |
| Proposition 4 | Property 2 (naive inscription launders by default) |
| Proposition 5 | Property 3 (cascade invisibility) |

**Action:** Audit theorem-environment numbering in the `.tex`. The
review revealed the numbering may already drift between `definition`
and `property` blocks (the reviewer counted differently than the prompt
expected). Re-number sequentially within a single counter shared
across Definitions, Propositions, Properties, Predictions, OR keep
separate counters but ensure every cross-reference resolves to the
intended object. Either is fine; consistency is the requirement.

---

## Critical revisions (block top-tier submission)

These must land before sending the paper to a top-tier venue.

### C1. Reframe the abstract and §1 introduction

**Reviewer point:** §1 of the review, §9 (strongest counter-argument).

> *"The phrase 'isolates the missing primitive' overclaims. The min-trust
> rule is a special case of Denning-style lattice integrity meet. The
> contribution is the application + validation, not the algebra."*

**Specific changes:**
- Abstract: remove "missing primitive" language. Replace with
  language like:
  > *"We adapt the lattice integrity-meet rule from information-flow
  > control (Denning 1976; Myers & Liskov 1997) to recursive agent
  > memory, add an inference-typing rule that caps derivations even
  > when all inputs are externally observed, identify a concrete
  > failure mode (inference laundering) in current LLM-agent memory
  > systems, and validate that the rule prevents it on a
  > deterministic fixture and a 139-session LLM-in-the-loop
  > benchmark."*
- §1 paragraph 3: drop "isolates that missing rule and validates it."
  Replace with: "instantiates an IFC-style integrity lattice for
  recursive agent memory and validates it on the laundering failure
  mode."
- §1 contributions list, item 1 ("a six-class source lattice...") —
  keep, but reframe item 2 as: *"a source-downgrading inscription
  primitive that combines lattice integrity meet (standard) with an
  inference-typing rule (specific to derivation semantics)."*
- §1 contribution item 3 stays as-is — provenance propagation alone
  insufficient is a fine empirical observation.
- §1 contribution item 5 (empirical validation) — add explicit
  reference to the 139-session v2 benchmark.

**Effort:** ~3 hours of LaTeX editing. No new experiments.

### C2. Add IFC / provenance prior-work positioning in §2

**Reviewer point:** §4.2 (IFC analogy) and §4.3 (provenance semirings)
of the review.

> *"This is a special case of a known framework... The phrase 'source
> downgrading' also collides with 'downgrading' in IFC, where
> downgrading/declassification are established terms."*

**Specific changes:**
- Add a new §2.X subsection: *"Information-flow control and integrity
  lattices."* Cite Denning 1976, Myers & Liskov 1997, Sabelfeld & Myers
  2003. State plainly: the algebraic core of source-downgrading is the
  integrity-meet rule from IFC, specialised to a six-class source
  lattice for agent memory. State what this paper adds beyond IFC:
  (a) the inference-typing rule on derivations (derivation changes
  source class even for high-integrity inputs); (b) the agent-memory
  application + concrete failure-mode characterisation
  (inference laundering); (c) the cascade-invisibility audit
  methodology.
- Add a new §2.X subsection: *"Database provenance."* Cite Green,
  Karvounarakis & Tannen 2007; Cheney, Chiticariu & Tan 2009. State:
  $\Prov$ propagation is a simple set-valued provenance annotation; the
  novelty is in binding lineage to an inscription policy in an agent
  memory API, not in the lineage algebra.
- **Rename caveat:** consider whether to keep "source downgrading" as
  the name given the IFC-collision with declassification/downgrading.
  Two options: (i) keep the name and add a footnote distinguishing it
  from IFC declassification; (ii) rename to *"derived-source meet"*
  or *"trust-bounded inscription"*. Recommendation: keep the current
  name (it's already published under that label in our codebase) and
  add the footnote.

**Effort:** ~4-6 hours of writing + literature checks.

### C3. Move the PoisonedRAG boundary into the paper itself

**Reviewer point:** §5.4 of the review.

> *"This result should be in the paper, not only in
> RESEARCH_CONCLUSIONS.md. Without it, readers may incorrectly infer
> that source-downgrading is a RAG poisoning defense. It is not."*

**Specific changes:**
- Add a new §5.X subsection: *"What the rule is not: single-shot RAG
  vs derivation chains."* Report the PoisonedRAG NQ result (vector
  0.45 → trace_memory 0.22 → vector_with_labels 0.21 ASR) and state
  that the 24pp ASR reduction is attributable to the LLM source
  classifier + label-aware retrieval prompt, **not** the inscription
  rule. Explain: PoisonedRAG is single-shot QA with no `add_derived`
  call, so the rule is dormant by design.
- Update §1 abstract to acknowledge the scope explicitly:
  > *"Source-downgrading constrains inscription, not retrieval; it is
  > not a RAG-poisoning defence."*
- Add to §6.4 (Limits): the deferral of `Source(.)` plus the
  single-shot-vs-derivation boundary means the rule alone does not
  defend production single-shot RAG. A deployable agent memory system
  needs both the rule (for derivations) and a source-label classifier
  (for raw text ingestion).

**Effort:** ~2 hours of writing. The benchmark data already exists.

### C4. Tighten Proposition 2 and the cascade-invisibility statement

**Reviewer point:** §3.2 and §6.1 of the review.

> *"Proposition 2 needs a slightly stronger invariant... the proof must
> say that reachability is computed by graph traversal and that
> traversal is cycle-safe."*

> *"the claimed exact undercount [in cascade invisibility] needs
> stricter conditions on chain shape and audit definition."*

**Specific changes:**
- Proposition 2 proof: add explicit statement that $\Prov$ is modelled
  as a finite directed graph (possibly acyclic by construction at
  inscription time), that "transitive closure" is graph reachability
  under a visited-set traversal, and that the induction is over chain
  length with the base case being the contributing record itself. As
  written the proof is correct but skips the graph model.
- Property 3 (cascade invisibility): the closed-form statement
  *"the undercount equals the number of chained derivations whose
  contributing records' labels were themselves laundered to external
  on prior steps"* is too broad. Either:
  - (i) Restrict to: linear chains where each derived record has
    exactly one contributor, each laundering produces `external`,
    and the local metric tests only `non-ext → ext`. Add these as
    explicit hypotheses. Then the closed-form holds.
  - (ii) Replace with the existential statement: *"there exist
    fixtures in which $\mathrm{LR}_{\text{local}}$ strictly undercounts
    $\mathrm{LR}_{\text{truth}}$"*. Drop the equality claim.
  - Recommendation: do (i). The closed form is more useful as
    guidance for production audits, and the restrictions are reasonable
    for the paper's fixture.

**Effort:** ~2 hours.

### C5. Narrow the trust-ordering claim around `react > tool`

**Reviewer point:** §3.3 of the review.

> *"There are realistic cases where `react > tool` is operationally
> correct... The paper uses source class as a proxy for reliability,
> but reliability is not determined by source class alone."*

**Specific changes (pick one):**
- **Option A (preferred, minimal):** In §3.2, explicitly narrow the
  ordering's claim to *"a default engineering policy for the tested
  implementation, not a universal trust order."* Add a paragraph
  explaining that for cases where a reactivated record carries
  verified provenance and a tool output is from an adversarial
  channel, the operational ordering may invert; an implementation can
  override the default by attaching per-record reliability metadata
  consulted alongside the source class.
- **Option B (ambitious):** Refactor the lattice to a product:
  $(\text{origin}, \text{verification})$ where origin is the seven
  classes and verification is a small qualifier (`verified`,
  `unverified`, `signed`). This is a larger paper change and likely
  pushes scope into a follow-on; defer to v2 of v2.

**Recommendation:** Option A for this revision. Option B for a future
paper if the multi-dimensional model becomes load-bearing.

**Effort:** Option A ~2 hours. Option B ~1-2 days + benchmark updates.

---

## High priority (strong reviewer recommendations)

These substantially strengthen §5 (empirical validation) and address
concrete reviewer asks. Recommended before re-submission, not strictly
blocking.

### H1. Run the five adversarial tests from review §7

**Reviewer point:** §7 of the review — five concrete tests with input
record structures and expected outputs.

**The tests:**
1. **Reconvergent branching DAG.** One branch clean, one contaminated;
   the rule must keep the contaminated label visible at the
   reconvergence point.
2. **Self-referential provenance through aliasing.** A record whose
   provenance graph contains a cycle. Either the implementation rejects
   at write/import time, or traversal must use a visited set.
3. **Contradictory same-rank inputs.** Two tool outputs disagreeing
   on a value. The rule produces a source-correct label (`infer`) but
   says nothing about epistemic warrant. Document the gap.
4. **Cross-agent trust transfer.** Agent A's `sim` record exported
   through a tool/API to Agent B. Without signed source envelopes,
   the label is lost. Document the failure mode.
5. **Adversarial source classifier prompt-injection.** A passage that
   contains text designed to manipulate the classifier. Measure the
   classifier's robustness and the downstream ASR.

**Specific changes:**
- Implement each test in `benchmarks/adversarial_extensions/`
  (new module). Reuse `MemoryAgent.add_derived` and the v2
  product-comparison harness.
- Report results in a new §5.X subsection: *"Adversarial extensions."*
- For test 5, the classifier-injection test, write up the result as
  part of the §6 limitations or as a new appendix on the
  `Source(.)` classifier program.

**Effort:** ~1-2 days of harness work + LaTeX writeup. The harness
infrastructure already exists (v2 dataset, multi-arm runner, judge).

### H2. Add a lattice-IFC baseline to the comparison

**Reviewer point:** §5.5 of the review.

> *"A stronger baseline would include lattice-taint/IFC meet
> propagation implemented directly as a baseline, not only naive and
> provenance-only policies."*

**Specific changes:**
- Implement a `lattice_taint_baseline` policy in
  `src/fgm/laundering.py` (alongside
  `naive_inscribe` and `provenance_propagating`) that applies
  pure min-trust meet WITHOUT the inference ceiling.
- Re-run the paper's 5-case fixture and the v2 benchmark with this
  new baseline.
- Expected result: the lattice-IFC baseline matches source-downgrading
  on `LR_local`, `Prov. recall`, `LR_truth` for the existing
  contamination cases (fab/sim inputs), but FAILS on cases where all
  inputs are `ext` (because pure min-trust returns `ext`, violating the
  inference-typing rule).
- This is the test that **isolates the inference-ceiling
  contribution**. If the lattice-IFC baseline matches source-downgrading
  on all metrics, the inference ceiling adds no measurable value and
  should be dropped from the paper's central claim. If it differs on
  the `ext+ext → infer` cases, source-downgrading's contribution
  beyond standard IFC is empirically demonstrated.

**Effort:** ~half a day.

### H3. Document the classifier-recall vs ASR relationship explicitly

**Reviewer point:** §8.2 of the review — derived the model
`ASR(r) = r·q_detected + (1-r)·q_missed` and concluded production
classifiers need adversarial recall in the high-80s to mid-90s.

**Specific changes:**
- Add a new §6.5 subsection (or appendix): *"Classifier-recall as the
  architectural bottleneck."* Reproduce the derivation from review §8.2.
  Plot ASR vs classifier recall under two model assumptions
  (`q_detected = 0` and `q_detected = 0.05`).
- State the production target: adversarial recall ≥ 0.86 for ASR ≤
  0.10; ≥ 0.93 for ASR ≤ 0.05.
- Position this as setting up the next paper in the sequence: a
  fine-tuned `Source(.)` classifier with measured adversarial recall.

**Effort:** ~half a day.

### H4. Add branching-DAG cases to the v2 benchmark

**Reviewer point:** §5.3 of the review.

> *"The benchmark appears mostly linear rather than branching/
> reconvergent. ... v2 is sufficient to validate the narrow mechanical
> claim for linear contaminated derivation chains with oracle labels."*

**Specific changes:**
- Extend `adversarial_reload_v2.py` schema to support branching
  derivation chains. Currently `derivation_prompts` is a flat tuple;
  extend to support a DAG via a list of `(prompt, input_record_ids)`
  pairs.
- Add ~10-20 sessions per domain with branching shapes (one observation
  → two derivations → reconverge at a third). Aim for ~70 new
  branching sessions across 7 domains.
- Re-run the v2 benchmark. Report results in §5.X.

**Effort:** ~1-2 days (schema change + new scenarios + run).

---

## Medium priority (publication strength)

### M1. Handle empty / pathological input sets explicitly

**Reviewer point:** §3.5 of the review.

**Specific changes:**
- Definition 6 (source-downgrading inscription): add a paragraph on
  edge cases.
  - Empty contributing set: implementation MUST reject or label `fab`.
  - Cyclic provenance: implementation MUST detect at write/import time
    and either reject or break the cycle.
  - Same-rank contradictions: implementation MUST surface a contradiction
    flag at write time (separate from the source label).
  - Hybrid source labels: out of scope for this paper; split records at
    the caller's level.
- Update the harness to test these cases.

**Effort:** ~half a day.

### M2. State the cross-agent label translation problem as future work

**Reviewer point:** §3.5 and §7 (adversarial test 4) of the review.

**Specific changes:**
- Add to §6.3 (Limits): cross-agent source-label transfer is not
  protected by the rule alone. Multi-agent systems need signed
  provenance envelopes or label translation tables. Cite as a
  follow-on paper in the publication sequence.

**Effort:** ~1 hour.

### M3. Acknowledge production-audit-protocol gap explicitly

**Reviewer point:** §6.3 of the review.

> *"Truth-grounded metrics are practical in fixtures and partially
> practical in production... Production audits need approximate truth
> anchors."*

**Specific changes:**
- In Prediction 1 (audit protocol requirement), add an explicit list
  of production approximations: immutable append-only provenance logs,
  canary contaminants, sampled oracle review, independent shadow
  source classifiers, signed source envelopes, claim-level support
  verification. State that the choice depends on deployment threat
  model.

**Effort:** ~1 hour.

---

## Low priority / next-paper

Items the reviewer flagged but that belong in follow-on work.

### L1. Multi-dimensional source labels (origin × verification × generation-mode × recency)

The reviewer's Option B from §3.3. Cleaner ontology but a larger change.
Defer to a future paper; mention in §6 as future direction.

### L2. Fine-tuned `Source(.)` classifier paper

§6.3 of the paper already defers this; the review's §8 derives a
quantitative target (high-80s/mid-90s adversarial recall). The next
paper in the sequence should build that classifier with an explicit
adversarial-recall benchmark.

### L3. Production audit protocol design

A separate methodology paper on how to run paired self-vs-truth audits
in deployed agents. The current paper points at the need; the actual
protocol design is a separate program.

### L4. Larger v2 benchmark (~500 sessions, balanced depth distribution)

The current v2 has 139 sessions with 118/14/7 depth distribution. A
follow-on paper could scale to 500+ with balanced depth (~150 each
depth-1/2/3) and add branching DAGs. Useful but not blocking.

---

## Suggested execution order

1. **Days 1-2:** C1 + C2 + C3 (LaTeX-only revisions, no new
   experiments). Result: paper is honestly framed and properly
   positioned vs IFC/provenance literature. This alone moves the
   reviewer's verdict from "publishable with substantial revisions" to
   "publishable with revisions."
2. **Day 3:** C4 + C5 (formal tightening, narrow ordering claim).
3. **Days 4-5:** H2 (lattice-IFC baseline) + H3 (classifier-recall
   appendix). Result: the empirical case isolates source-downgrading's
   contribution from standard IFC; the next-paper target is explicit.
4. **Days 6-8:** H1 (five adversarial tests) + H4 (branching DAG
   cases). Result: the empirical breadth addresses the reviewer's
   strongest §5 critique.
5. **Day 9:** M1 + M2 + M3 (pathological inputs, cross-agent gap,
   production-audit gap).
6. **Day 10:** Final pass, re-read, ensure all reviewer points are
   either addressed in-text or explicitly deferred.

Total: ~2 weeks of focused work for a credible top-tier resubmission.

---

## Tracking

When changes are implemented, update this document with a status
column. Each section above can be marked:
- ✅ done in `Source_Downgrading_v2.tex`
- 🟡 partial (specify which sub-point)
- ❌ deferred (with one-line reason)

Once all Critical items are done, re-submit the revised paper +
revision letter quoting the reviewer's points back at each addressed
section.

### Current status — 2026-05-13

| Item | Status | Notes |
|---|---|---|
| C1 | ✅ done in `paper/Source_Downgrading.tex` | Abstract and introduction reframed from "missing primitive" to IFC integrity-meet application + validation; 139-session benchmark and PoisonedRAG boundary added. |
| C2 | ✅ done in `paper/Source_Downgrading.tex` | Added IFC/integrity-lattice and database-provenance subsections with Denning, Myers/Liskov, Sabelfeld/Myers, Green et al., and Cheney et al.; kept name with IFC-collision caveat. |
| C3 | ✅ done in `paper/Source_Downgrading.tex` | Added single-shot RAG / PoisonedRAG negative-control subsection and limits language distinguishing retrieval from derived writeback. |
| C4 | ✅ done in `paper/Source_Downgrading.tex` | Strengthened transitive-provenance proof with finite graph traversal / visited-set language; restricted cascade-invisibility equality to linear externalizing chains. |
| C5 | ✅ done in `paper/Source_Downgrading.tex` | Narrowed the trust ordering to a default engineering policy and documented the `react > tool` inversion case. |
| H1 | ❌ deferred | Adversarial extension tests not implemented yet. |
| H2 | ❌ deferred | Lattice-IFC baseline not implemented yet. |
| H3 | 🟡 partial | PoisonedRAG classifier bottleneck is acknowledged; full ASR-vs-recall derivation/plot not added. |
| H4 | ❌ deferred | Branching-DAG benchmark schema/cases not implemented yet. |
| M1 | 🟡 partial | Paper now documents empty inputs, cycles, contradictions, hybrid labels, and semantic-support assumptions; harness tests not added yet. |
| M2 | ✅ done in `paper/Source_Downgrading.tex` | Cross-agent label-translation gap added to limits. |
| M3 | ✅ done in `paper/Source_Downgrading.tex` | Prediction 1 now lists production truth-anchor approximations. |

Build check from `paper/`: `pdflatex -interaction=nonstopmode -jobname=Source_Downgrading_verify Source_Downgrading.tex`
passes and writes `Source_Downgrading_verify.pdf` (25 pages). The temp
job name is used because `Source_Downgrading.pdf` may be locked by a
viewer on Windows. Remaining LaTeX noise is one underfull hbox in the
Proposition 1 paragraph.

Follow-up adjustment: framework-heavy integration language was removed
again to keep the paper usable as a standalone ground-truth/scaffolding
artifact. The retained additions are the load-bearing ones for this
paper's objective: source downgrading is framed as an
inscription-integrity invariant, the API boundary says
`add_derived(...)` must compute labels rather than accept caller labels,
the source-inference uncertainty caveat remains, and the discussion
keeps the `Source != Credibility != Confidence` distinction.
