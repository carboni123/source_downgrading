# Trace Memory Architecture

Validation probes and proof-of-concept agent memory system for the trace-theoretic framework formalized in:

> **Trace Before Memory: A Causal-Retentive Operator for Recursive Mnestic Systems**
> Diego Falkowski Carboni (2025)

This repository contains two integrated components:

1. **`trace_probes`** — Minimal executable probes that validate the revised trace/memory definitions from the paper
2. **`fgm`** (Fold-Gated Memory) — A six-layer agent memory system implementing the full architecture

Both build on the companion [SIMFC calculus](https://doi.org/10.5281/zenodo.20042034) and its [reference implementation](https://github.com/carboni123/simrec-reference).

## Architecture

```
prior attended state
  -> retained trace residue (Layer 1: trace)
  -> causal intervention sensitivity
  -> stabilized storage (Layer 2: storage)
  -> bounded top-k retrieval with margin tracking (Layer 3: addressability)
  -> fold into non-bookkeeping transition (Layer 4: folding)
  -> record of fold operation (Layer 5: operation-memory)
  -> confusability-triggered compression (Layer 6: compression)
```

### Trace Probes (`src/trace_probes/`)

Five probes mapping paper definitions to executable tests:

| Probe | Paper section | What it tests |
|-------|---------------|---------------|
| `trace_retention_probe` | Def. 1 (Causal trace) | Lag projection retains residue of prior state |
| `causal_trace_probe` | Def. 1 (Intervention condition) | Perturbing prior state changes trace component |
| `addressability_probe` | Def. 4 (Cognitive memory) | Top-k retrieval selects stored record |
| `fold_force_probe` | Def. 5 (Fold-force) | Folded memory changes non-bookkeeping transition |
| `p7_retrieval_regime_probe` | Prop. 1 (Conditional interference) | Phase boundary across four retrieval regimes |

### Fold-Gated Memory (`src/fgm/`)

| Layer | Class | Role |
|-------|-------|------|
| 1 Trace | (implicit) | LLM context window — short-lived, decaying |
| 2 Storage | `MemoryStore` | Persistent vector store with per-record fold-force tracking |
| 3 Addressability | `MarginRetriever` | Top-k retrieval with retrieval margin and confusability monitoring |
| 4 Folding | `FoldGate` | Dual-pass transition measurement — only memories that change decisions pass |
| 5 Operations | `OperationMemory` | Records of prior fold operations — enables recursive self-correction |
| 6 Compression | `Compressor` | Prunes zero-fold records, merges duplicates, preserves retrieval margins |

`FGMAgent` ties all layers into a single query cycle: retrieve -> fold -> measure -> gate -> record -> auto-compress.

### Primitive Validation Instrumentation

The FGM layer now exposes the first roadmap primitives needed for bottom-up validation:

- Source/provenance fields on `MemoryRecord` (`source_label`, `source_confidence`, `provenance`)
- Provenance source labels, active-source labels, and route scores on `FoldResult`
- Source-aware `OperationRecord` metadata for fold operations
- Transparent route constants for `null`, `trace`, `durable_memory`, `operation_memory`, `correction_chain`, and `quarantine`
- `ValidationRecord` JSONL helpers in `fgm.validation`, so primitive experiments can be scored from logs instead of in-memory objects
- Replayable route/source metrics and baseline policies (`always_write`, `never_write`, `source_blind`)
- Synthetic inscription-utility policy comparison in `fgm.inscription`, with fixed future-utility labels
- Controlled correction-chain policy comparison in `fgm.correction`, with fixed prior/evidence/update/revised/delta labels
- Residual-attention retrieval comparison in `fgm.residual`, with semantic, recency, posture, and source-aware posture baselines
- Engineered self-index binding comparison in `fgm.self_index`, with user/project/role/permission truth labels
- Toy coupled-field intervention probes in `fgm.coupling`, measuring memory-to-attention and attention-to-write shifts
- Source-inference policy comparison in `fgm.source_inference`, with uniform-external, lexical, feature-threshold, and combined baselines
- Inference-laundering policy comparison in `fgm.laundering`, with naive, provenance-propagating, and source-downgrading inscription baselines
- Consolidated fixed-seed artifact generation via `examples/run_validation_roadmap.py`
- Multi-seed controlled replication via `examples/run_controlled_replication.py`
- Real-component smoke replication with sentence-transformer embeddings via `examples/run_real_component_validation.py`
- Real-embedding query-variant replication via `examples/run_real_embedding_replication.py`
- Live LLM validation gate with OpenAI/Anthropic provider support and skip/cost artifacts via `examples/run_live_llm_validation.py`
- Live LLM prompt-suite replication with prompt/response audit logs via `examples/run_live_llm_replication.py`
- Live replication boundary diagnostics via `examples/analyze_live_replication.py`, including provider-valid route semantics
- Provider/model comparison matrix artifacts via `examples/run_live_provider_model_comparison.py`
- Additional fixed-truth task families via `--case-family` on the live replication runner
- Deterministic source-aware rerank boundary regression via `examples/run_rerank_boundary_regression.py`
- Claim-scale OpenAI N=20 live replication artifacts via filename arguments on `examples/run_live_llm_replication.py`

The current route scorer is intentionally deterministic and simple. It is a validation scaffold, not the final learned inscription-routing model.
The live LLM command defaults to OpenAI and records `skipped` with a zero-cost ledger when `OPENAI_API_KEY` is not configured.
Set `OPENAI_LIVE_MODEL` or pass `--model` to override the default model.

```bash
python examples/run_validation_roadmap.py --output-dir results --seed 0
python examples/run_source_inference_validation.py --output-dir results
python examples/run_laundering_validation.py --output-dir results
python examples/run_controlled_replication.py --output-dir results --seed-count 50
python examples/run_real_component_validation.py --output-dir results
python examples/run_real_embedding_replication.py --output-dir results --seed-count 20
python examples/run_live_llm_validation.py --output-dir results
python examples/run_live_llm_replication.py --output-dir results --seed-count 5
python examples/analyze_live_replication.py --output-dir results
python examples/run_rerank_boundary_regression.py --output-dir results
python examples/run_live_llm_replication.py --output-dir results --seed-count 20 --summary-filename live_llm_replication_n20_summary.json --audit-filename live_llm_replication_n20_audit.jsonl
python examples/analyze_live_replication.py --output-dir results --summary-filename live_llm_replication_n20_summary.json --audit-filename live_llm_replication_n20_audit.jsonl --diagnostics-filename live_llm_replication_n20_diagnostics.json
python examples/run_live_llm_replication.py --output-dir results --seed-count 5 --empty-response-retries 1 --summary-filename live_llm_replication_retry_summary.json --audit-filename live_llm_replication_retry_audit.jsonl
python examples/run_live_llm_replication.py --output-dir results --seed-count 20 --empty-response-retries 1 --summary-filename live_llm_replication_n20_retry_summary.json --audit-filename live_llm_replication_n20_retry_audit.jsonl
python examples/run_live_provider_model_comparison.py --output-dir results --target openai:gpt-5-mini-2025-08-07 --target openai:gpt-5-nano --target openai:gpt-4.1-mini --target anthropic --seed-count 5 --empty-response-retries 1 --min-passed-targets 3 --reuse-existing
python examples/run_live_provider_model_comparison.py --output-dir results --target openai:gpt-5-nano --seed-count 5 --empty-response-retries 1 --max-output-tokens 1000 --artifact-prefix live_provider_model_comparison_nano_max1000 --summary-filename live_provider_model_comparison_nano_max1000_summary.json
python examples/run_live_llm_replication.py --provider openai --model gpt-4.1-mini --output-dir results --seed-count 5 --case-family billing_refund --empty-response-retries 1 --max-output-tokens 300 --summary-filename live_task_family_billing_refund_gpt41_summary.json --audit-filename live_task_family_billing_refund_gpt41_audit.jsonl --no-audit-text
python examples/run_live_llm_replication.py --provider openai --model gpt-4.1-mini --output-dir results --seed-count 5 --case-family security_rotation --empty-response-retries 1 --max-output-tokens 300 --summary-filename live_task_family_security_rotation_gpt41_summary.json --audit-filename live_task_family_security_rotation_gpt41_audit.jsonl --no-audit-text
python examples/run_live_llm_validation.py --provider anthropic --output-dir results
```

## Install

```bash
pip install -e .                    # core (numpy only)
pip install -e ".[embeddings]"      # + sentence-transformers for real embedding tests
pip install -e ".[llm]"             # + anthropic SDK for live LLM tests
pip install -e ".[dev]"             # + pytest
pip install -e ".[all]"             # everything
```

## Run tests

```bash
python -m pytest tests/ -v                    # all offline tests
python -m pytest tests/ -v -m live            # live LLM tests (requires provider API key)
```

## Examples

```bash
python examples/trace_demo.py                 # trace probes demo
python examples/trace_full_validation.py      # all 6 predictions validated
python examples/fgm_demo.py                   # fold-gated memory lifecycle
```

## Minimal usage

```python
import numpy as np
from trace_probes import LeakyTraceOperator, trace_retention_probe

sequence = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
op = LeakyTraceOperator(dim=3, decay=0.8, max_lag=10)
result = trace_retention_probe(sequence, op, t=2, k=2)
print(result.metrics)
```

```python
from fgm import FGMAgent, hash_embed, default_transition

agent = FGMAgent(dim=64, transition_fn=default_transition, embed_fn=hash_embed)
result = agent.query("What was decided about X?", embed=hash_embed("X decision"))
print(result.fold_force, result.gated)
```

## Key findings

- **P7 phase boundary validated at N=1000**: rich_distinctive maintains 1.000 hit rate; sparse_confusable degrades to ~0.03
- **Fold-force metric**: L2 divergence doesn't discriminate with LLM transitions; answer-quality fold-force (cosine toward memory content) discriminates at 5.9x
- **h_cog necessity**: Non-bookkeeping variable restriction proven functionally necessary for signal/noise discrimination
- **Compression trigger**: Proactive confusability-triggered compression confirmed as architecturally necessary; reactive per-record pruning is insufficient

## Related publications

- **SIMFC paper**: [DOI 10.5281/zenodo.20042034](https://doi.org/10.5281/zenodo.20042034)
- **simrec-reference**: [github.com/carboni123/simrec-reference](https://github.com/carboni123/simrec-reference)
- **Trace Formalization paper**: [DOI 10.5281/zenodo.20043070](https://doi.org/10.5281/zenodo.20043070)

## License

MIT
