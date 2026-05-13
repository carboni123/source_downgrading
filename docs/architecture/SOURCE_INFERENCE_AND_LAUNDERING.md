# Source Inference and Inference-Laundering Validation

Originally developed on branch: `source-inference-and-laundering`
Parent branch: `validation-primitives` (Codex's primitive-roadmap execution)
Current state: merged into `master` via the validation framework merge.

## Why this branch exists

The validation-primitives branch grounds the primitive validation roadmap with
source-sensitive routing, operation-memory, correction chains, residual
attention, and self-index binding under the assumption that *source labels
are supplied by the generator*. Two open questions blocked treating those
results as foundational:

1. **`Source(.)` is given, not earned.** Every existing test passes the source
   label as a constructor argument; the policy reads it back. The harder
   sub-problem — recovering source from content/context — has no harness.
2. **Inference laundering is not tested.** The roadmap lists
   `inference_laundering_rate` as a Phase 1 metric and lists the experiment
   `retrieved memory -> inferred claim -> later recall`, but no implementation
   covers it.

Both gaps are flagship items in the MAFC v4 revision spec
(`../followup_random_day_2/MAFC_v4_revision_spec.md`). This branch adds the
two missing harnesses without touching Codex's existing work.

## What was added

### `src/fgm/laundering.py`

Drives the real `FGMAgent.add` and `FGMAgent.query` API across five
deterministic cases. Three inscription policies are compared:

- `naive_inscribe` -- the laundering baseline. Defaults of `agent.add(content)`:
  source falls back to `external`, no provenance is propagated.
- `provenance_propagating` -- sets `source=inference` and copies the union of
  contributing records' provenance plus their record ids.
- `source_downgrading` -- caps source at the min-trust of contributing inputs
  (inference is the upper bound), and still propagates provenance.

Trust ordering: `fabricated_or_uncertain < simulation < inference <
retrieved_memory < tool_output < external`.

Metrics:

- `inference_laundering_rate` -- locally detectable laundering (uses the
  stored labels of contributing inputs).
- `derived_trust_ceiling_violation_rate` -- truth-grounded laundering (uses
  the fixture's `expected_max_trust`).
- `provenance_chain_recall` -- whether the expected provenance origins are
  reachable from each derived record.
- `false_externalization_after_inference` -- whether a later query about a
  derived claim is routed to a trusted write target when it should be
  quarantined.
- `transitive_provenance_depth_mean` -- mean walk-back depth on derived records.

### `src/fgm/source_inference.py`

Tests `Source(.)` as an inference function. Policies take only
`(content, query_context, retrieval_margin, recency_rank)` and emit a
predicted source class. Four policies:

- `uniform_external` -- always predict `external` (the laundering baseline).
- `lexical_rules` -- token-based, reusing Codex's existing marker sets
  (`_FABRICATED_QUERY_MARKERS`, `_SIMULATION_QUERY_MARKERS`,
  `_EXTERNAL_QUERY_MARKERS`) plus new sets for inference, retrieved memory,
  and tool output.
- `feature_threshold` -- thresholds retrieval margin and recency only.
- `combined` -- lexical first, then feature threshold for the lexically silent
  residue.

Fixture: 30 cases across 6 source classes with 5 deliberately ambiguous cases
that lack lexical markers and sit in mid-range feature space.

### Examples and tests

- `examples/run_laundering_validation.py`
- `examples/run_source_inference_validation.py`
- `tests/architecture/test_laundering.py` -- 9 cases including the cascade-invisibility
  finding (see below).
- `tests/architecture/test_source_inference.py` -- 7 cases including dominance gates per
  class.

All 192 tests pass under `pytest -m "not live"` (168 existing + 24 new).

## Results

### Laundering harness (seed 0, 5 cases, 7 derived records)

```text
policy                  laundering  prov_recall  false_ext_after  ceiling_viol  prov_depth
naive_inscribe          0.43        0.00         0.40             1.00          0.00
provenance_propagating  0.00        1.00         0.40             0.57          2.29
source_downgrading      0.00        1.00         0.00             0.00          2.29
```

Three substantive findings:

1. **Cascade invisibility.** `naive_inscribe`'s local laundering rate (0.43) is
   much lower than its truth-grounded ceiling-violation rate (1.00). Once the
   policy relabels a derived record as `external`, subsequent derivations
   from it see "all external inputs" and the framework cannot detect the
   continued laundering from its own perspective. The two metrics together
   make this property observable and testable.

2. **Provenance propagation is necessary but not sufficient.**
   `provenance_propagating` perfectly preserves provenance chains
   (depth 2.29 across the fixture) but still violates the trust ceiling in
   57% of derived records, because labeling everything `inference`
   over-trusts derivations from simulation or fabricated inputs. It also
   still triggers false externalization on later queries (0.40), since
   downstream routing is fooled by the inflated `inference` label.

3. **Source downgrading is the actual fix.** Capping the derived source at
   the min-trust input (with `inference` as the ceiling) plus propagating
   provenance yields zero laundering, zero ceiling violations, and zero false
   externalization on later queries. This is the policy MAFC v4 should
   adopt as the default `R[.]` rule for the `infer` route.

### Source-inference harness (30 cases, 5 ambiguous)

```text
policy             accuracy  false_ext  ambiguous
uniform_external   0.17      1.00       0.40
lexical_rules      0.87      0.12       0.20
feature_threshold  0.50      0.20       0.80
combined           0.93      0.04       0.60
```

Per-class breakdown for `combined`:

```text
external              0.8  (1 confused with tool_output)
tool_output           0.8  (1 confused with external -- the ambiguous case)
retrieved_memory      1.0
inference             1.0
simulation            1.0
fabricated_or_uncertain 1.0
```

Three substantive findings:

1. **Source inference is empirically possible at high accuracy on
   structured cases.** `combined` reaches 0.93 overall vs `uniform_external`'s
   0.17 -- a 0.76 delta. This is grounded evidence that `Source(.)` can be
   built as more than an interface assumption.

2. **Feature thresholding alone is insufficient.** `feature_threshold`
   reaches 0.50 accuracy but completely misses three classes
   (`tool_output -> external`, `simulation -> retrieved_memory`,
   `fabricated -> retrieved_memory`). Margin and recency cannot separate
   these classes because they overlap in feature space. The MAFC v4 spec's
   open issue around `Source(.)` should explicitly note that retrieval
   features are diagnostically incomplete; lexical or learned content
   features are required.

3. **Lexical rules dominate marker classes; ambiguous cases need features.**
   `lexical_rules` hits 1.0 on the four lexically explicit classes (fab,
   sim, infer, tool) but drops on `retrieved_memory` (0.6) because the
   ambiguous retrieved cases lack markers. The `combined` policy recovers
   these via the feature fallback. This shows the right composition
   pattern: high-precision lexical signals first, feature thresholding as
   the residue handler.

## What this changes about MAFC v4

The v4 revision spec called out two open issues that this branch can now
close:

- **`Source(.)` is no longer purely an interface assumption.** The combined
  policy demonstrates that source inference on plausibly realistic content
  patterns is feasible at ~0.93 accuracy. The spec should add a footnote
  citing this harness and noting the breakdown structure (lexical
  high-precision + feature fallback).

- **Inference laundering is no longer a flagship prediction without a test.**
  The laundering harness gives MAFC v4 a deterministic, replayable
  falsifier. The empirical claim becomes: source-downgrading inscription
  produces zero laundering and zero false externalization on the controlled
  fixture, while naive inscription violates the trust ceiling 100% of the
  time. This is the kind of grounded result the spec needs to stand on.

The cascade-invisibility property (laundering rate < ceiling violation rate
for the naive policy) is a third finding that should be added to the v4
failure-modes section as a sub-failure of inference laundering: even the
framework's own self-audit can be fooled once a label has been laundered
once, so detection requires either truth-grounded ablations or rigorous
provenance tracking from the start.

## How to reproduce

```bash
cd C:\Users\DiegoPC\Documents\GitHub\source_downgrading
pip install -e .
python -m pytest tests/architecture/test_laundering.py tests/architecture/test_source_inference.py -q
python examples/architecture/run_laundering_validation.py --output-dir results/architecture/architecture
python examples/architecture/run_source_inference_validation.py --output-dir results/architecture/architecture
```

Artifacts:

```text
results/architecture/laundering_validation_summary.json
results/architecture/laundering_validation.jsonl
results/architecture/source_inference_validation_summary.json
results/architecture/source_inference_validation.jsonl
```

The targeted source-inference and laundering tests pass with 24/24 tests. The
full non-live suite passes with 200 passed and 4 deselected after the merge.
The harnesses are deterministic under fixed seed.
