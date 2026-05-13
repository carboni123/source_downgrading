# trace-memory

A Python memory layer for LLM agents with operationally validated trust composition.

**Status:** v0.6.0a0 (Phase 6 alpha — empirical benchmarks). See [`PRD.md`](PRD.md) for the full product requirements.

## What this is

`trace-memory` is a memory layer that promises a small set of behaviours, all backed by a validation result file:

- Every stored record carries a source label.
- Derived records cannot launder their inputs' trust (source-downgrading inscription).
- Retrieval returns operationally-meaningful memory (records that would change the agent's decision via fold-force), not just similar text.
- Belief revisions preserve full lineage as correction nodes.
- Audit returns paired self-vs-truth metrics that detect cascade-invisibility.

The validated primitives now live in this consolidated artifact under `src/fgm/`, with evidence in `results/architecture/` and the historical ledger in `docs/architecture/VALIDATED_PRIMITIVES_LEDGER.md`. This library is the developer-facing facade over that work.

## What this is not

- Not a RAG framework.
- Not a vector database.
- Not a chat-memory wrapper.
- Not a managed service.

## Install

Pre-alpha; not yet on PyPI. Local dev install uses the consolidated repository root:

```bash
pip install -e ".[dev]"
```

Python 3.10+, single dependency: `numpy>=1.24`.

## Five-minute quickstart

Anti-laundering in ten lines. Run this and inspect the audit.

```python
from trace_memory import MemoryAgent, SourceLabel

agent = MemoryAgent()

# Plant two records: a real observation and a fabricated rumor.
obs = agent.add("server returned 500 at 14:02 UTC", source=SourceLabel.EXTERNAL)
rumor = agent.add("rumor: outage was caused by ops team", source=SourceLabel.FABRICATED_OR_UNCERTAIN)

# Derive a new claim from both. trace-memory computes the source label
# from contributing inputs -- you cannot pass source= here.
derived = agent.add_derived("the ops team caused the outage", inputs=[obs, rumor])
print(derived.source_label)   # 'fabricated_or_uncertain' -- capped at the lowest-trust input
print(derived.provenance)     # contains both obs.record_id and rumor.record_id

# A self-only audit warns that truth labels are absent; it cannot declare clean.
print(agent.audit_laundering())
```

If you try to bypass the API and inscribe the derivation as an external observation, the public API will not let you. `agent.add(content)` without a `source=` argument raises `MissingSourceError`; `agent.add_derived(content, source=..., inputs=...)` raises `TypeError` (no `source` keyword).

## The complete Phase 1 surface

### 1. Construct an agent

```python
from trace_memory import MemoryAgent

agent = MemoryAgent(
    dim=64,                 # embedding dim
    retrieval_k=3,          # default top-k for query()
    fold_threshold=0.01,    # records below this fold-force are gated out
)
```

Optional plug-ins: pass `embed_fn=...` for a custom embedding function (defaults to deterministic hash bag-of-words), `transition_fn=...` for a custom `Φ(state, input, fold)` transition operator.

### 2. Ingest records with explicit source

```python
from trace_memory import SourceLabel

agent.add(
    "external observation: deploy migration succeeded",
    source=SourceLabel.EXTERNAL,
    provenance=("deploy_log_42",),
    source_confidence=1.0,
)

# All seven source classes are available:
SourceLabel.EXTERNAL
SourceLabel.TOOL_OUTPUT
SourceLabel.RETRIEVED_MEMORY
SourceLabel.INFERENCE
SourceLabel.SIMULATION
SourceLabel.FABRICATED_OR_UNCERTAIN
SourceLabel.OPERATION_RECORD
```

`SourceLabel` inherits from `str`, so `SourceLabel.EXTERNAL == "external"` and you can interpolate it directly into strings.

### 3. Derive new records safely

```python
r1 = agent.add("E1", source=SourceLabel.EXTERNAL, provenance=("sensor_a",))
r2 = agent.add("S1", source=SourceLabel.SIMULATION, provenance=("sim_branch",))

# The derived source is computed: cap at min-trust of inputs, with
# INFERENCE as the ceiling. Caller cannot assert a higher source.
derived = agent.add_derived(
    "derived from r1 and r2",
    inputs=[r1, r2],   # MemoryRecord objects or record-id strings
)

derived.source_label  # 'simulation' (capped by r2)
derived.provenance    # contains 'sensor_a', 'sim_branch', r1.record_id, r2.record_id
```

The trust ordering is `fabricated_or_uncertain < simulation < inference < retrieved_memory < tool_output < external`. Inference is the upper bound for any derivation; external inputs cannot produce a derived record labeled external.

### 4. Query and inspect routing

```python
result = agent.query("did the deploy succeed?")

result.retrieved              # list[RetrievalHit]
result.fold_force             # float -- magnitude of change in transition
result.gated                  # bool -- whether the fold passed the threshold
result.selected_route         # 'trace' | 'durable_memory' | 'operation_memory' |
                              # 'correction_chain' | 'quarantine' | 'null'
result.route_scores           # dict[str, float] -- score per route
result.source_labels          # list[str] -- source labels of retrieved records
```

The routing layer quarantines any retrieval set containing an untrusted source (simulation, fabricated). This is defensive by design — if you want trusted routes on honest queries, use `retrieval_k=1` or filter the store by source before query.

### 5. Record belief revisions

```python
node = agent.revise_belief(
    prior_belief="high CPU means we need more web servers",
    evidence="after adding servers CPU stayed high; profiler found O(n^2) handler",
    update_operation="replace capacity hypothesis with code-path hypothesis",
    revised_belief="fix handler complexity before scaling infrastructure",
    delta="root_cause:web_capacity->handler_complexity",
    provenance=("profiling_run_17",),
    confidence=0.93,
)

# All revisions are listable:
agent.correction_nodes()  # list[CorrectionNode]
```

Correction nodes preserve the full lineage of every belief update. They are persisted as `OPERATION_RECORD`-source records in the underlying store and surfaced via `agent.correction_nodes()`.

### 6. Audit for laundering

```python
# Self-referential audit only -- carries a cascade-invisibility warning
# because once a label has been laundered, the self-metric undercounts.
audit = agent.audit_laundering()
audit.local_laundering_rate     # float
audit.truth_grounded_rate       # None (no truth supplied)
audit.cascade_invisibility_warning  # True
audit.is_clean                  # False -- cannot be declared clean without truth

# Paired audit with truth ceilings supplied (e.g., in tests):
truth = {"D_pure": SourceLabel.INFERENCE, "D_sim": SourceLabel.SIMULATION}
audit = agent.audit_laundering(truth_ceilings=truth)
audit.local_laundering_rate     # float
audit.truth_grounded_rate       # float -- truth-grounded ceiling violations
audit.gap                       # truth - local
audit.is_clean                  # True iff both rates are zero
```

`is_clean` requires the truth-grounded rate to be defined and zero. The self-rate alone is treated as a lower bound, never as a measurement, because of cascade invisibility (see `paper/Source_Downgrading.tex` §5 for the property and its empirical demonstration).

## Full demo

`examples/anti_laundering_demo.py` runs all five laundering-fixture scenarios end-to-end with routing verification and a counter-demo showing how the truth-grounded audit catches an attempt to bypass the API.

```bash
python examples/anti_laundering_demo.py
```

Expected output: every derivation lands on its truth-supplied source label; honest queries route to `operation_memory`; contaminated queries route to `quarantine`; clean audit (0.0/0.0).

## What this library does NOT promise

Reproduced from `docs/architecture/VALIDATED_PRIMITIVES_LEDGER.md`:

- No retrieval guarantees beyond ~10k records (untested at scale).
- No compression / forgetting (vault grows monotonically in v0.1).
- No performance guarantees on natural-prose `Source(·)` inference (Phase 2 work).
- No open-ended long-horizon stability claims (the deterministic laundering benchmark covers source-downgrading chains up to depth 5, but not unconstrained agent stability).
- No active recall (the library will not surface memories without an explicit query).
- No cryptographic tenant isolation (self-index metadata only, Phase 2).
- No adversarial robustness guarantees (untested under adversarial inputs).

See `PRD.md` §7 for the constraint table with pointers to specific ledger entries.

## Phase 2 additions (v0.2.0a0)

Three additional primitives are now available:

### Self-index binding (FR-8)

Engineered tenant isolation via metadata. Records carry a `SelfIndex(user_id, project_id, role, permission_scope, standing_commitment)`; retrieval filters by the agent's active index.

```python
from trace_memory import MemoryAgent, SelfIndex, SourceLabel

alice = MemoryAgent(self_index=SelfIndex(user_id="alice", project_id="X"))
alice.add("alice's note", source=SourceLabel.EXTERNAL)

# Switching the active index changes what's retrievable.
alice.active_self_index = SelfIndex(user_id="bob", project_id="X")
result = alice.query("note")  # alice's record is now excluded
```

Records with no SelfIndex remain globally visible. Records with a SelfIndex are visible only when the agent's active index matches on user, project, role, and permission_scope. `standing_commitment` is content, not a filter. **This is NOT cryptographic tenant isolation** — see PRD §7 limits.

### Utility-based inscription (FR-5)

Opt-in budget-constrained writes. Configure with `UtilityWritePolicy(budget=N)` and use `add_candidate(...)` to queue writes; `flush_inscriptions()` commits only the top-`budget` by predicted utility.

```python
from trace_memory import MemoryAgent, SourceLabel, UtilityWritePolicy

agent = MemoryAgent(inscription_policy=UtilityWritePolicy(budget=3))

agent.add_candidate("useful_1", source=SourceLabel.EXTERNAL, predicted_utility=0.95)
agent.add_candidate("useful_2", source=SourceLabel.EXTERNAL, predicted_utility=0.90)
agent.add_candidate("distractor", source=SourceLabel.EXTERNAL, predicted_utility=0.10)
agent.add_candidate("useful_3", source=SourceLabel.EXTERNAL, predicted_utility=0.85)

committed = agent.flush_inscriptions()  # top 3 by utility; distractor dropped
```

Immediate-commit `add(...)` is unaffected by the policy. The two paths coexist.

### Source(·) inference (FR-6)

Recover source labels from content + retrieval features. The combined policy reaches 0.93 accuracy on a structured 30-case fixture. The DistilBERT path measured in `source_classifier/` reaches 1.0 raw accuracy, 0.0 false externalization, and 0.0 trust upgrades on the generated natural-prose test split. On the authored challenge set the raw checkpoint reaches 0.983 accuracy with 0.0 false externalization and 0.0 trust upgrades, so `policy="trained_transformer"` applies explicit boundary guards around the model; the guarded policy reaches 1.0 accuracy and zero false externalization/trust upgrades on both generated and challenge evaluations.

```python
from trace_memory import MemoryAgent, infer_source

# Standalone helper:
label = infer_source(
    "hypothetical: if traffic doubled",
    retrieval_margin=0.15,
    recency_rank=2,
)
# label == SourceLabel.SIMULATION

# Agent convenience (emits warning):
agent = MemoryAgent()
record = agent.add_with_inferred_source("fabricated rumor: ...")
# record.source_label == 'fabricated_or_uncertain'
```

Available policies: `"uniform_external"` (baseline), `"lexical_rules"`, `"feature_threshold"`, `"combined"` (default), `"trained_transformer_raw"`, guarded `"trained_transformer"` gated by `TRACE_MEMORY_SOURCE_CLASSIFIER_MODEL` plus the optional `source-classifier` dependency group, and `"llm"` (v0.8.0a0 — see below).

### LLM-based Source(·) classifier (v0.8.0a0)

For natural-prose ingestion where the rule-based policies are too brittle (open-domain RAG, attacker-controlled retrieval, etc.), `policy="llm"` routes classification through a configurable LLM. The classifier is *pluggable*: wrap any OpenAI-compatible chat client. The library does not take a hard dependency on `openai`.

On real PoisonedRAG NQ (100 questions, gpt-4.1-mini) the bundled prompt scores 98% precision on clean Wikipedia/NQ passages and 71% recall on attacker-crafted adversarial passages — driving downstream attack-success rate from 0.46 (vanilla cosine RAG) to 0.21 when the labels are used by the LLM in the answer step.

```python
from openai import OpenAI
from trace_memory import (
    LLMSourceClassifier, MemoryAgent, SourceLabel, set_llm_classifier,
)

# Construct one classifier and reuse it (in-instance cache amortises cost).
classifier = LLMSourceClassifier(OpenAI(), model="gpt-4.1-mini")
set_llm_classifier(classifier)

# Now policy="llm" is available everywhere.
agent = MemoryAgent(dim=64, retrieval_k=3)
record = agent.add_with_inferred_source(
    "Rumour: the deploy was sabotaged by the ops team.",
    policy="llm",
)
# record.source_label == 'fabricated_or_uncertain'

# Standalone classification:
label = LLMSourceClassifier(OpenAI()).classify("Canberra is the capital of Australia.")
# label == SourceLabel.EXTERNAL
```

The classifier returns a `SourceLabel`. Any callable matching `Callable[[str], SourceLabel | str]` works in `set_llm_classifier(...)` — useful for stubs in tests or for plugging in a domain-tuned classifier of your own. Calling `policy="llm"` without first setting a classifier raises a clear error pointing at `set_llm_classifier`.

The bundled prompt template is the same one validated on PoisonedRAG; override `system_prompt=` on `LLMSourceClassifier` for domain-specific rubrics.

## Phase 3 addition (v0.3.0a0): pluggable persistent storage

Agents now accept a `storage=` argument. The default remains in-memory (state lost on process exit). Pass a `SQLiteStorage(path)` to persist to disk; the agent reloads its full store on construction, so records survive process restarts.

```python
from trace_memory import MemoryAgent, SQLiteStorage, SourceLabel

# Session 1: write and close.
with MemoryAgent(storage=SQLiteStorage("/path/to/vault.db")) as agent:
    agent.add("server returned 500 at 14:02", source=SourceLabel.EXTERNAL)
    agent.add_derived(
        "the server was failing under load",
        inputs=[...],
    )

# Session 2 (later process): all records reload automatically.
with MemoryAgent(storage=SQLiteStorage("/path/to/vault.db")) as agent:
    print(len(agent))                # all records from session 1
    print(agent.correction_nodes())  # correction chains too
    print(agent.audit_laundering())  # audit state recovered
```

All record types persist: ordinary content, derived records (with the derived flag round-tripped so `audit_laundering` works correctly after reload), correction nodes, and self-index metadata.

### Storage protocol

The `Storage` typing.Protocol defines a five-method contract: `save`, `load_all`, `delete`, `contains`, `close`. Two reference implementations ship:

| Backend | Persistence | Use case |
|---|---|---|
| `InMemoryStorage` (default) | None — state lost on process exit | Tests, ephemeral agents, the validation harnesses |
| `SQLiteStorage(path)` | Single file, embedded SQLite | Production single-process agents, persistence across restarts, no external service required |

Custom backends are straightforward: implement the `Storage` protocol and pass an instance to `MemoryAgent(storage=...)`. Vector serialization uses `np.save` (preserves dtype and shape across versions). Provenance and metadata serialize as compact JSON.

### Limitations (v0.3)

- **Single-process only.** Two `MemoryAgent` instances pointing at the same SQLite file produce last-write-wins behaviour. No coordination layer.
- **Read-path is in-memory.** The agent loads everything on construction and uses the in-memory store for retrieval and fold-force. NFR-3's ~10k-record ceiling still applies.
- **No async API yet.** The storage layer is synchronous; pair with `asyncio.to_thread(...)` for async callers. A first-class async surface is planned for v0.4.

## Performance

NFR-3 targets (PRD §5.3): `query(...)` against 10k records under 50 ms with hash embeddings, under 500 ms with sentence-transformers. Both pass with margin.

| Embedding | N | p50 query (ms) | p95 query (ms) | NFR-3 target |
|---|---|---|---|---|
| hash (64-dim) | 10,000 | **34.70** | 38.43 | < 50 ms |
| sentence-transformers (384-dim) | 10,000 | **81.02** | 122.22 | < 500 ms |

Other characteristics from the latest sweep:

- `add()` is O(1) in record count (~0.01 ms in-memory, ~0.04 ms SQLite-memory, ~1.93 ms SQLite-disk).
- `query()` is O(N) — linear scaling on cosine retrieval. Confirmed: 1.4 ms @ N=100 → 4.3 ms @ N=1k → 34.7 ms @ N=10k.
- SQLite-disk persistence adds ~1.9 ms per write but does not affect query latency (queries read from the in-memory cache).
- Memory footprint is ~1.1 KB/record with hash embeddings, ~3.6 KB/record with sentence-transformers.

Full sweep across `{in_memory, sqlite_memory, sqlite_disk} × {hash, sentence_transformers} × {N=100, 1000, 10000}` is in [`results/benchmarks/PERFORMANCE.md`](../results/benchmarks/PERFORMANCE.md). To reproduce:

```bash
python benchmarks/run_perf.py            # full sweep, ~3 min
python benchmarks/run_perf.py --quick    # N <= 1000 cells only, ~30 s
```

## Inference-laundering benchmark (v0.6)

163 deterministic scenarios across 7 domains (SRE, customer support, security, finance, healthcare, legal, research), 7 templated failure modes, and 16 hand-crafted hard cases. The dataset now validates its own truth fields by recomputing final and intermediate trust ceilings from the scenario graph before the benchmark runs. Three baselines use the same agent infrastructure, isolating only the inscription policy:

| Baseline | Laundering (self-audit) | Final ceiling violation | Chain-step ceiling violation | Provenance recall | Cascade gap |
|---|---|---|---|---|---|
| `no_source` (naive: everything as external) | 0.724 | **1.000** | 1.000 | 0.000 | 0.276 |
| `provenance_only` (lineage but no trust cap) | 0.000 | **0.448** | 0.433 | 1.000 | **0.448** |
| `trace_memory` (source-downgrading) | **0.000** | **0.000** | **0.000** | **1.000** | **0.000** |

Three findings worth surfacing:

- **trace-memory is the only baseline that achieves zero on both final and chain-step truth-grounded ceiling violations.** Across all 163 scenarios, including depth-5 late contamination, full-source-lattice fan-in, fan-out/fan-in provenance, lexical decoys, and clean multi-source convergence, source-downgrading never lets a derived record exceed its rightful trust ceiling.
- **`provenance_only` still fails even though provenance recall is perfect.** It preserves lineage but labels every derivation as inference, so 44.8% of final derived records and 43.3% of intermediate chain steps violate the computed source ceiling.
- **45 scenarios trigger cascade invisibility for `no_source`.** Pure-inference scenarios are silently mis-labeled as external observations; the system's own audit reports no laundering because there are no non-external inputs to flag, while the truth-grounded ceiling check identifies the violation.

Full report with per-domain and per-failure-mode breakdowns: [`results/benchmarks/LAUNDERING_BENCHMARK.md`](../results/benchmarks/LAUNDERING_BENCHMARK.md). To reproduce:

```bash
python benchmarks/laundering_dataset.py --output benchmarks/data/laundering_dataset.jsonl
python benchmarks/run_laundering_benchmark.py
```

Deterministic; no LLM in the loop; runs in under a second.

## Source-boundary benchmark

The laundering benchmark assumes source labels are already grounded. The source-boundary benchmark tests the prior boundary: can the current rule-based `Source(.)` policies recover labels from realistic ingestion text plus retrieval features?

126 labelled cases cover 7 domains, all six content source labels, and three boundary types: canonical markers, natural prose, and source decoys. The product `combined` policy is conservative about source upgrades and materially improves over the earlier feasibility-floor rules:

| Policy | Overall accuracy | False externalization | Trust upgrade | Canonical | Natural | Decoy |
|---|---|---|---|---|---|---|
| `uniform_external` | 0.167 | 1.000 | 0.833 | 0.167 | 0.167 | 0.167 |
| `lexical_rules` | 0.698 | 0.295 | 0.278 | 0.952 | 0.976 | 0.167 |
| `feature_threshold` | 0.389 | 0.467 | 0.611 | 0.500 | 0.500 | 0.167 |
| `combined` | **0.976** | **0.029** | **0.024** | 1.000 | 1.000 | 0.929 |

This validates the current `Source(.)` classifier as a useful product floor on the labelled boundary fixture. The remaining misses are retrieved-memory decoys that look like fresh external facts without boundary metadata, so production systems should still prefer explicit app-owned source labels for high-stakes ingestion.

Full report: [`results/benchmarks/SOURCE_BOUNDARY_BENCHMARK.md`](../results/benchmarks/SOURCE_BOUNDARY_BENCHMARK.md). To reproduce:

```bash
python benchmarks/source_boundary_dataset.py --output benchmarks/data/source_boundary_dataset.jsonl
python benchmarks/run_source_boundary_benchmark.py
```

## Mnestic-attentional coupling benchmark

The coupling benchmark tests the architecture boundary we care about before bringing in a live LLM: can source-labelled memory records condition an attention-like selector without unsafe contamination, trust laundering, or provenance loss?

70 deterministic cases cover 7 domains and 5 coupling modes: fabricated decoys, simulation decoys, retrieved-memory bridges, fabricated-only support, and simulation-only support. Four variants isolate the architecture choices:

| Variant | Decision accuracy | Unsafe contamination | Trust violation | Provenance recall | Source match |
|---|---|---|---|---|---|
| `no_memory` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `raw_memory` | 0.200 | 0.800 | 1.000 | 0.000 | 0.000 |
| `labels_only` | 1.000 | 0.000 | 0.400 | 0.500 | 0.600 |
| `trace_memory` | **1.000** | **0.000** | **0.000** | **1.000** | **1.000** |

This does **not** prove live LLM attention will behave correctly. It proves the narrower primitive that matters for the framework: once source-labelled records are made available to an attentional selector, trace-memory can support safe selection, quarantine contaminated support, and write the resulting decision through `add_derived(...)` without exceeding the correct trust ceiling.

Full report: [`results/benchmarks/COUPLING_BENCHMARK.md`](../results/benchmarks/COUPLING_BENCHMARK.md). To reproduce:

```bash
python benchmarks/coupling_dataset.py --output benchmarks/data/coupling_dataset.jsonl
python benchmarks/run_coupling_benchmark.py
```

## LLM-in-the-loop coupling benchmark (v0.7)

Live-LLM extension of the coupling benchmark. Same 70-case dataset; the deterministic selector is replaced by `gpt-4.1-mini`. Four prompting variants test what the LLM does with source-labelled context:

| Variant | Decision accuracy | Unsafe contamination |
|---|---|---|
| `no_memory` (just question + answers) | 0.600 | 0.000 |
| `raw_memory` (memories shown without source labels) | 0.743 | 0.000 |
| `labels_only` (memories with source labels, no instructions) | 0.743 | 0.000 |
| `trace_memory_envelope` (labels + trust ordering + abstention rules) | 0.743 | 0.000 |

**The honest finding: the LLM half is not the primary failure mode this benchmark catches.** A modern API model (`gpt-4.1-mini`) does not propagate source-labelled contamination into unsafe answer selection across any of the 70 cases. Unsafe-contamination is uniformly zero because RLHF training already refuses obviously antisocial answers ("accuse the ops team", "deny the refund because the account is supposedly abusive") on stylistic cues alone, with or without source labels.

The accuracy variation in `fabricated_only` and `simulation_only` modes (where the dataset expects `quarantine` as the right answer) reveals a real integration nuance: the LLM often picks `safe` (the right action regardless of memory) rather than `quarantine` (explicitly refuse to act on contaminated memory). This is a defensible choice but not what the dataset labels as correct.

The architectural takeaway: trace-memory's load-bearing guarantees are at *writeback* (`add_derived(...)` enforces trust ceilings on stored records); the LLM at the selection layer makes its own choices that are largely safe but not perfectly aligned with library-expected abstention. For applications where abstention is required when memory is contaminated, enforce it programmatically by filtering retrieved records by source before passing them to the LLM — don't rely on the LLM to infer abstention from a trust-ordering block in the prompt.

Full report: [`results/benchmarks/COUPLING_LLM_BENCHMARK.md`](../results/benchmarks/COUPLING_LLM_BENCHMARK.md). To reproduce:

```bash
export OPENAI_API_KEY=...
python benchmarks/run_coupling_llm_benchmark.py     # ~$0.03 in API costs
python benchmarks/run_coupling_llm_benchmark.py --smoke   # 8 calls, sanity check
```

## Async API (v0.4.0a0)

Every public mutating and read method has an `aX` async wrapper:

```python
import asyncio
from trace_memory import MemoryAgent, SourceLabel, ainfer_source

async def main():
    agent = MemoryAgent()
    r1 = await agent.aadd("observation 1", source=SourceLabel.EXTERNAL)
    r2 = await agent.aadd("observation 2", source=SourceLabel.EXTERNAL)
    derived = await agent.aadd_derived("inferred", inputs=[r1, r2])
    result = await agent.aquery("what did we observe?")
    audit = await agent.aaudit_laundering()
    label = await ainfer_source("hypothetical: ...")
    # All the sync methods have an aX counterpart:
    # aadd, aadd_derived, aadd_candidate, aflush_inscriptions,
    # aadd_with_inferred_source, aquery, arevise_belief, aaudit_laundering.

asyncio.run(main())
```

### Concurrency model

trace-memory is **async-compatible**, not natively async. Each `aX` method awaits a threadpool execution of its sync counterpart via `asyncio.to_thread`. This means:

- **Safe to use from async-only codebases.** No event-loop blocking on the agent's hot paths.
- **Mutations serialize through an internal `RLock`.** Two concurrent `aadd(...)` calls do not race on the underlying store or SQLite connection.
- **Reads are not lock-protected.** A read may interleave with a concurrent mutation; in CPython, individual dict and list operations are atomic under the GIL, so reads always see a consistent snapshot of an individual record.
- **CPU-bound work does not gain true parallelism.** Cosine retrieval and fold-force computation hold the GIL; multiple coroutines hitting `aquery` run one at a time even when offloaded to the threadpool. The async surface is valuable for boundary symmetry, not for CPU-bound throughput.
- **SQLite cross-thread access is enabled.** `SQLiteStorage` uses `check_same_thread=False`; the agent's lock guarantees no concurrent SQL operations.

If you need true parallelism on CPU-bound retrieval, run multiple `MemoryAgent` instances (one per process or worker), not multiple coroutines on the same agent.

## Bulk ingestion API (v0.5.0a0)

Production agents typically parse an LLM's output into typed chunks and route each to the right `aadd_*` call. The library now ships that plumbing. Two patterns are supported:

### Pattern 1: app-owned structure

Your application already has its own prompt and parser (the LLM emits a shape your app dictates). The library accepts typed requests directly — no envelope assumed:

```python
from trace_memory import (
    MemoryAgent, ObservationRequest, DerivationRequest, RevisionRequest, SourceLabel,
)

# Your prompt, your parser, your shape:
raw_output = await llm.complete(my_app_prompt)
my_parsed = my_app_parser(raw_output)

requests = [
    ObservationRequest(content=o["text"], source=SourceLabel.EXTERNAL,
                       provenance=tuple(o["sources"]))
    for o in my_parsed["facts_user_told_me"]
] + [
    DerivationRequest(content=i["text"], inputs=tuple(i["from_facts"]))
    for i in my_parsed["my_inferences"]
] + [
    RevisionRequest(
        prior_belief=r["prior"], evidence=r["evidence"],
        update_operation=r["op"], revised_belief=r["revised"],
        delta=r["delta"], confidence=r["conf"],
    )
    for r in my_parsed["belief_updates"]
]

results = await agent.aingest_batch(requests)
# results[i] is the MemoryRecord (or CorrectionNode) for requests[i]
```

The library does **not** impose a JSON schema on the LLM prompt. Your prompt and parser remain app-owned.

### Pattern 2: library-owned envelope

If you don't already have a structured output format, adopt the library's canonical envelope. Drop `StructuredEnvelope.system_prompt_block()` into your LLM system prompt; parse the output via `StructuredEnvelope.parse(...)`:

```python
from trace_memory import MemoryAgent, StructuredEnvelope

system_prompt = (
    "You are an SRE agent. Emit your output as JSON.\n\n"
    + StructuredEnvelope.system_prompt_block()
)

raw_output = await llm.complete(system=system_prompt, user=user_query)
envelope = StructuredEnvelope.parse(raw_output)
results = await agent.aingest_envelope(envelope)
```

`StructuredEnvelope.parse(...)` accepts both:
- **JSON form** — `{"observations": [...], "derivations": [...], "revisions": [...]}`
- **Inline marker form** — `OBSERVED: ... / TOOL: ... / INFERRED: ... / SIMULATED: ... / FABRICATED: ... / RETRIEVED: ...` per line, falling back when the LLM ignores the JSON instruction.

The system prompt block embeds the trust ordering and instructs the model to prefer the lowest-trust source that honestly applies. Inferences belong in `derivations` (with contributing input record IDs), not in `observations` with `source=external`.

### Which pattern to pick

- App with an existing prompt schema: **Pattern 1.** The library doesn't interfere.
- Greenfield agent, no prompt yet: **Pattern 2.** Cheaper to adopt the canonical envelope than to design your own.
- Mixed: use **Pattern 2** for the source-labelling subset, your own schema for the rest of the agent's output.

In both patterns the trust composition guarantees apply: a `DerivationRequest` routes through `add_derived(...)`, so source-downgrading and provenance propagation still hold.

## Beyond v0.5

- Type stubs (`.pyi`) and docstring polish.
- Chroma or other vector-DB backend (when retrieval at scale becomes the binding constraint).
- Compression / forgetting — pending validation beyond the source-downgrading artifact.
- Source(·) classifier learned from natural data — pending natural-prose validation.
- First external user integration.

## License

MIT — see [`LICENSE`](LICENSE).

## Author

Diego Falkowski Carboni / Tyxter — `diego@tyxter.dev`
