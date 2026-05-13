# Source Downgrading Standalone Paper

Author: Diego Falkowski Carboni <diego@tyxter.dev> - Tyxter

## What this repo is

This workspace isolates the Source Downgrading paper and its validation
materials from the broader SIMReC/SIMFC framework. Treat the paper as a
standalone systems/security-style contribution.

The core thesis is:

> Source downgrading is an inscription-integrity invariant for derived
> memory records. Given correctly labeled contributors, a derived record
> may not carry source integrity higher than its weakest contributor,
> with inference as the upper bound for derived inscription.

## Primary artifact

- Main paper: `paper/Source_Downgrading.tex`
- Build output: `paper/Source_Downgrading.pdf`
- Validation/code evidence:
  - `src/`
  - `tests/`
  - `benchmarks/`
  - `results/`

## Framing rules

Keep the paper narrow and grounded.

Do:
- Frame source downgrading as a writeback-time integrity invariant.
- Emphasize the falsifiable claim: given correct input labels, derived
  records cannot acquire illegitimate source integrity.
- Keep the IFC/provenance positioning: the algebraic meet pattern is
  known; the contribution is application, inference ceiling, API binding,
  failure-mode characterization, and validation.
- Preserve the boundary between source, credibility, and confidence:
  source is how a record was produced; credibility is revisable belief in
  the content; confidence is an estimated quality/calibration signal.
- Treat source inference as an upstream, fallible empirical layer.
- Use PoisonedRAG as a negative control showing that source downgrading
  is not a single-shot RAG-poisoning defense.

Do not:
- Present source downgrading as a new memory type.
- Present it as a full memory-reliability solution.
- Claim it solves retrieval poisoning, hallucination detection,
  contradiction resolution, cross-agent label stripping, or belief
  revision.
- Add major references to MAFC, correction chains, routing papers, or
  other framework components unless the claim is directly load-bearing
  for the source-downgrading invariant.
- Use the paper as a place to explain the whole SIMReC/SIMFC framework.

## Load-bearing claims to keep

- Source downgrading is an inscription-integrity invariant.
- Derived trust cannot exceed input trust.
- Inference is the upper bound for derived inscription.
- Provenance must be propagated transitively.
- Caller-supplied labels are the laundering attack surface; an
  `add_derived(content, contributing_record_ids)` API should compute
  derived labels from contributor labels.
- Source, credibility, and confidence are distinct. A confirmed inference
  remains an inference; its credibility may rise, but its source label
  should not be promoted to external.

## Evidence and validation posture

Use existing results conservatively:

- Deterministic laundering fixture: validates the mechanical invariant.
- Multi-seed embedding-noise sweep: shows the label invariant is not a
  retrieval-order artifact.
- Adversarial-reload v2 benchmark: validates derived-writeback behavior
  under LLM-generated derivation chains with authored/oracle source
  labels.
- PoisonedRAG: negative control. It measures source classifier and
  label-aware prompt behavior in single-shot RAG; the inscription rule is
  dormant because no derived record is written.

## Build commands

Compile the paper with MiKTeX:

```powershell
Push-Location paper
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode Source_Downgrading.tex
Pop-Location
```

If the PDF is locked by a viewer, compile with a temporary job name:

```powershell
Push-Location paper
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -jobname=Source_Downgrading_verify Source_Downgrading.tex
Pop-Location
```

Run `pdflatex` twice when citations or cross-references change.

## Editing guidance

- Keep edits scoped to the standalone paper.
- Prefer precise claims over broader framework language.
- Keep benchmark claims tied to the exact condition tested.
- When adding limitations, distinguish:
  - label inference failures,
  - derived-writeback failures,
  - retrieval-time failures,
  - belief/credibility failures,
  - provenance transport failures.
- Avoid adding new terminology unless it removes ambiguity in the
  source-downgrading invariant.
