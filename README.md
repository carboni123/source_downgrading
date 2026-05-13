# Source Downgrading Artifact

This repository is the standalone source of truth for the Source Downgrading paper, its validation code, benchmark harnesses, committed result files, and publication-readiness notes.

Core claim:

> Source downgrading is an inscription-integrity invariant for derived memory records. Given correctly labeled contributors, a derived record may not carry source integrity higher than its weakest contributor, with inference as the upper bound for derived inscription.

## Repository Layout

| Path | Purpose |
|---|---|
| `paper/Source_Downgrading.tex` | Publication manuscript. |
| `paper/Source_Downgrading.pdf` | Final compiled manuscript, produced by the publication gate. |
| `src/trace_memory/` | Public memory API and source-downgrading implementation. |
| `src/fgm/`, `src/trace_probes/` | Validation primitives used by the deterministic laundering and architecture evidence. |
| `tests/trace_memory/` | Non-live API, storage, source-boundary, and benchmark tests. |
| `tests/architecture/` | Non-live validation primitive tests and live-marked provider tests. |
| `benchmarks/` | Benchmark runners, datasets, and benchmark support packages. |
| `benchmarks/data/` | Serialized benchmark input datasets. |
| `results/benchmarks/` | Committed benchmark outputs and generated benchmark reports. |
| `results/architecture/` | Committed validation result files from the architecture harness. |
| `CLAIMS_MATRIX.md` | Claim-by-claim evidence audit for the paper. |
| `REPRODUCIBILITY.md` | Commands, observed outputs, and live/API rerun boundaries. |
| `PUBLICATION_CHECKLIST.md` | Submission-readiness checklist and remaining risks. |
| `docs/` | Inventory, historical docs, review notes, and legacy architecture ledgers. |

The former `trace-memory/` and `trace-memory-architecture/` trees have been merged into this layout. Historical package metadata is preserved under `docs/historical/`.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Optional live or heavy dependencies:

```powershell
python -m pip install -e ".[llm,embeddings,source-classifier]"
```

## Non-Live Validation

Run the combined non-live test suite:

```powershell
python -m pytest tests -m "not live"
```

Run the deterministic source-downgrading fixture tests:

```powershell
python -m pytest tests/architecture/test_laundering.py tests/trace_memory/test_fr4_add_derived_no_laundering.py tests/trace_memory/test_benchmark.py -m "not live"
```

Reproduce deterministic benchmark reports:

```powershell
python benchmarks/laundering_dataset.py --output benchmarks/data/laundering_dataset.jsonl
python benchmarks/run_laundering_benchmark.py --output-dir results/benchmarks
python benchmarks/source_boundary_dataset.py --output benchmarks/data/source_boundary_dataset.jsonl
python benchmarks/run_source_boundary_benchmark.py --output-dir results/benchmarks
python benchmarks/coupling_dataset.py --output benchmarks/data/coupling_dataset.jsonl
python benchmarks/run_coupling_benchmark.py --output-dir results/benchmarks
```

Live/API benchmarks are not rerun by default. Their committed outputs are under `results/benchmarks/` and `results/architecture/`, and their status is recorded in `REPRODUCIBILITY.md`.

## Paper Build

Compile with MiKTeX from the `paper/` directory:

```powershell
Push-Location paper
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode Source_Downgrading.tex
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode Source_Downgrading.tex
Pop-Location
```

If a viewer locks the PDF, use a temporary job name for verification:

```powershell
Push-Location paper
& "C:\Users\DiegoPC\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode -jobname=Source_Downgrading_verify Source_Downgrading.tex
Pop-Location
```

## Scope

Source downgrading is not a new memory type, not a full memory-reliability solution, not a retrieval-poisoning defense, not hallucination detection, and not belief revision. It is a writeback-time source-integrity rule for derived records. Source, credibility, and confidence remain separate quantities.
