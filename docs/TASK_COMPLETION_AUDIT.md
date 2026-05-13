# Task Completion Audit

Verification date: 2026-05-13.

| Requirement | Evidence |
| --- | --- |
| Read and follow `AGENTS.md`. | Paper edits preserve standalone framing, source/credibility/confidence distinction, and PoisonedRAG negative-control boundary. |
| Consolidate `trace-memory/`, `trace-memory-architecture/`, paper files, benchmark outputs, and docs. | Clean root layout: `paper/`, `src/`, `tests/`, `benchmarks/`, `results/`, `examples/`, `docs/`. Pre-move inventory: `docs/pre_consolidation_file_manifest.csv`; mapping: `docs/REPOSITORY_INVENTORY.md`. |
| Avoid destructive deletion before inventory. | Inventory was created before old roots were removed after migration. Historical package metadata moved to `docs/historical/`; committed result/data files preserved under `results/` and `benchmarks/data/`. |
| Update imports, paths, README, paper Code Availability, and benchmark docs. | `pyproject.toml`, `README.md`, `AGENTS.md`, `paper/Source_Downgrading.tex`, benchmark runners, architecture examples, and active docs point to consolidated paths. |
| Extract substantive paper claims into `CLAIMS_MATRIX.md`. | `CLAIMS_MATRIX.md` contains 21 claims classified as formal/theoretical, implementation/API, deterministic conformance, LLM benchmark, negative-control/limitation, or future work. |
| Mark unsupported claims or revise wording. | Paper overclaims were softened; PoisonedRAG classifier count corrected; unsupported broad claims are now limitations/future work. |
| Run non-live tests. | `REPRODUCIBILITY.md`: `python -m pytest tests -m "not live"` -> 374 passed, 4 deselected. |
| Run deterministic/source-downgrading fixture tests. | `REPRODUCIBILITY.md`: focused pytest command -> 45 passed. |
| Verify deterministic conformance fixture, multi-seed sweep, adversarial-reload v2, and PoisonedRAG outputs. | `REPRODUCIBILITY.md` records JSON validation output from `results/architecture/*.json` and `results/benchmarks/*.json`. |
| Do not rerun expensive live/API benchmarks unless acceptable. | `REPRODUCIBILITY.md` explicitly marks product-comparison and PoisonedRAG live/API benchmarks as not rerun and gives reproduction commands. |
| Proofread and keep paper narrow. | `paper/Source_Downgrading.tex` keeps the invariant scoped to derived writeback; non-goals and limitations remain explicit. |
| Preserve `Source != Credibility != Confidence`. | Paper and checklist preserve the boundary; no source promotion is claimed from increased credibility. |
| Ensure benchmark claims match evidence exactly. | `CLAIMS_MATRIX.md` and `REPRODUCIBILITY.md` align paper claims with committed JSON rates; PoisonedRAG low-trust count is 356. |
| Compile paper twice and fix broken refs/citations/serious boxes. | `paper/Source_Downgrading.pdf` built twice. Final log has no undefined citations/references and no overfull boxes; one mild underfull hbox remains. |
| Create publication checklist. | `PUBLICATION_CHECKLIST.md`. |
| Final PDF path. | `paper/Source_Downgrading.pdf`. |
