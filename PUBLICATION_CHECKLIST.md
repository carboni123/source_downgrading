# Publication Checklist

Status as of 2026-05-13.

| Area | Status | Notes |
| --- | --- | --- |
| Abstract accuracy | Pass | Abstract now frames source downgrading as an inscription-integrity invariant for systems that compose and reinscribe derived records. It does not present the rule as a full memory-reliability solution. |
| Novelty framing | Pass | Paper states that the meet pattern is known from IFC/provenance work; the contribution is the binding of source-integrity meet, derivation-operation ceiling, provenance preservation, and a memory writeback API into an enforceable inscription primitive. |
| Related work | Pass with residual risk | Related-work language was softened so existing RAG/agent-memory systems are not claimed to be intrinsically unsafe. Submission risk remains if reviewers expect deeper comparison against additional memory-agent systems. |
| Claim/evidence coverage | Pass | `CLAIMS_MATRIX.md` maps each substantive claim to formal sections, implementation paths, tests, result files, reproduction commands, or future-work/limitation status. |
| Reproducibility | Pass | `REPRODUCIBILITY.md` records commands and outputs. Non-live tests passed. Live/API benchmark outputs were validated from committed result files and clearly marked as not rerun. |
| Limitations | Pass | Limitations distinguish source-inference failures, incomplete contributor witnesses, derived-writeback failures, retrieval-time failures, belief/credibility failures, and provenance transport failures. PoisonedRAG is framed as a negative control. |
| Code availability | Pass | Paper Code Availability points at consolidated `paper/`, `src/`, `tests/`, `benchmarks/`, `results/`, and `docs/` paths. Root `README.md` documents the repo layout and validation commands. |
| Build artifacts | Pass | `paper/Source_Downgrading.pdf` builds with MiKTeX `pdflatex` after two passes. Final log has no undefined references/citations and no overfull boxes. |
| Source/credibility/confidence boundary | Pass | Paper preserves the distinction: source is production origin, credibility is revisable belief, confidence is a quality/calibration signal. Confirmed inference remains inference. |
| PoisonedRAG interpretation | Pass | Paper explicitly states that PoisonedRAG does not exercise `add_derived` and therefore does not validate the inscription rule. Classifier count corrected to 356 low-trust adversarial passages. |
| Journal submission risks | Open | Main risks: small conformance fixtures, live/API results not rerun in this pass, additional LLM ablations would sharpen attribution, source inference is upstream and fallible, oracle labels in adversarial-reload v2, omitted contributors, and transport-boundary failure for stripped cross-agent labels. |

## Submission Gate

- [x] Consolidated repository structure exists.
- [x] Main paper lives in `paper/Source_Downgrading.tex`.
- [x] Final PDF exists at `paper/Source_Downgrading.pdf`.
- [x] Non-live tests pass.
- [x] Focused source-downgrading fixture tests pass.
- [x] Committed benchmark result files validate headline claims.
- [x] Live/API benchmark rerun status is explicit.
- [x] Claims matrix exists.
- [x] Reproducibility log exists.
- [x] Remaining risks are documented rather than hidden.
