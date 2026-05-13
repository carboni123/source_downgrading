# Repository Inventory

This inventory was created before the consolidation moves. The full pre-move file manifest with SHA-256 hashes is in `docs/pre_consolidation_file_manifest.csv` and contains 350 visible source/evidence files.

## Pre-Consolidation Inputs

| Area | Files | Role |
|---|---:|---|
| root | 2 | `AGENTS.md` and `Source_Downgrading.tex`. |
| `trace-memory/` | 184 | Public API package, tests, benchmark runners, benchmark data, committed benchmark results, docs, and examples. |
| `trace-memory-architecture/` | 164 | Validation primitives, architecture tests, examples, docs, and committed validation result files. |

## Consolidation Map

| Original path | Consolidated path |
|---|---|
| `Source_Downgrading.tex` | `paper/Source_Downgrading.tex` |
| `trace-memory/src/trace_memory/` | `src/trace_memory/` |
| `trace-memory-architecture/src/fgm/` | `src/fgm/` |
| `trace-memory-architecture/src/trace_probes/` | `src/trace_probes/` |
| `trace-memory/tests/` | `tests/trace_memory/` |
| `trace-memory-architecture/tests/` | `tests/architecture/` |
| `trace-memory/benchmarks/*.py`, benchmark packages | `benchmarks/` |
| `trace-memory/benchmarks/*_dataset.jsonl` | `benchmarks/data/` |
| `trace-memory/benchmarks/poisonedrag_data/` | `benchmarks/data/poisonedrag/` |
| `trace-memory/benchmarks/*_results.json`, `*.md`, `*.log`, `*_audit.jsonl` | `results/benchmarks/` |
| `trace-memory-architecture/results/` | `results/architecture/` |
| `trace-memory/examples/` | `examples/trace_memory/` |
| `trace-memory-architecture/examples/` | `examples/architecture/` |
| `trace-memory/README.md` and research notes | `docs/` |
| `trace-memory-architecture/*.md` | `docs/architecture/` |
| original `pyproject.toml` files | `docs/historical/` |

## Redundant Local Metadata Removed

After the visible source/evidence files were moved, the old roots contained only generated or local metadata: nested `.git` directories, `.pytest_cache`, `.ruff_cache`, `__pycache__`, and egg-info. Those were removed from the publication artifact. The local `.env` was moved to the repository root and is ignored by `.gitignore`.

