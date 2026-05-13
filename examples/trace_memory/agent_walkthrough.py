"""End-to-end walkthrough of the trace-memory v0.4 public API.

Simulates an SRE / devops LLM agent across two sessions:

    Session 1 -- the agent observes incident events, derives inferences,
                 receives contradictory evidence, revises its belief, and
                 persists everything to SQLite.

    Session 2 -- a fresh process opens the same SQLite vault, verifies
                 the agent's memory survived, and runs a final audit.

Demonstrates every Phase 1/2 validated primitive plus persistence (v0.3)
and the async surface (v0.4):

    FR-1 source-labelled ingestion             agent.aadd(...)
    FR-2 fold-force retrieval                  result.fold_force
    FR-3 source-sensitive routing              result.selected_route
    FR-4 source-downgrading derived writes     agent.aadd_derived(...)
    FR-5 utility inscription                   UtilityWritePolicy + aflush
    FR-6 Source(.) inference                   agent.aadd_with_inferred_source
    FR-7 correction-chain belief revision      agent.arevise_belief(...)
    FR-8 self-index scoping                    SelfIndex(...)
    FR-9 paired audit                          agent.aaudit_laundering(...)

Run:

    python examples/agent_walkthrough.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import warnings

from trace_memory import (
    MemoryAgent,
    SelfIndex,
    SourceLabel,
    SQLiteStorage,
    StructuredEnvelope,
    UtilityWritePolicy,
    ainfer_source,
)


def heading(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def step(label: str, value: object = "") -> None:
    if value == "":
        print(f"  {label}")
    else:
        print(f"  {label:.<48s} {value}")


async def session_one(db_path: str) -> None:
    heading("Session 1: SRE agent observes an incident")

    # Scope every record to alice's deploy project.
    agent = MemoryAgent(
        storage=SQLiteStorage(db_path),
        self_index=SelfIndex(
            user_id="alice",
            project_id="deploy",
            role="maintainer",
            permission_scope="prod",
        ),
        retrieval_k=3,
    )
    step("opened agent with self_index", agent.active_self_index)
    step("storage backend", type(agent.storage).__name__)

    # --- Turn 1: observe an external event -------------------------------
    obs1 = await agent.aadd(
        "deploy migration completed at 14:01 UTC",
        source=SourceLabel.EXTERNAL,
        provenance=("deploy_log_a8",),
    )
    step("FR-1 added external observation", obs1.record_id)

    # --- Turn 2: ingest a tool output (high trust but not external) ------
    obs2 = await agent.aadd(
        "monitoring tool returned: cache hit rate dropped from 92% to 47%",
        source=SourceLabel.TOOL_OUTPUT,
        provenance=("monitoring_alert_15",),
    )
    step("FR-1 added tool output", obs2.record_id)

    # --- Turn 3: derive an inference. Source is computed, NOT asserted. ---
    derived = await agent.aadd_derived(
        "cache regression appears tied to the recent deploy",
        inputs=[obs1, obs2],
        record_id="hyp_cache_regression",
    )
    step("FR-4 derived inference written", derived.record_id)
    step("FR-4 derived source (computed)", derived.source_label)
    step("FR-4 derived provenance", derived.provenance)

    # --- Turn 4: query memory for fold-force-ranked retrieval ------------
    result = await agent.aquery("cache regression after deploy")
    step("FR-2 retrieved records", [h.record.record_id for h in result.retrieved])
    step("FR-2 fold_force", f"{result.fold_force:.3f}")
    step("FR-2 gated", result.gated)
    step("FR-3 selected_route", result.selected_route)

    # --- Turn 5: a colleague ingests an unlabelled note. Use the         --
    # --- combined Source(.) inference policy to recover a label.        --
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        note = await agent.aadd_with_inferred_source(
            "unverified note from chat: maybe legal banned rollbacks today",
        )
    step("FR-6 inferred source for note", note.source_label)

    # --- Turn 6: contradictory evidence arrives. Revise the belief.      --
    revision = await agent.arevise_belief(
        prior_belief="cache regression appears tied to the recent deploy",
        evidence="post-mortem: cache regression caused by key rotation, not deploy",
        update_operation="replace_cause_hypothesis",
        revised_belief="cache regression caused by key rotation; deploy was incidental",
        delta="cause:deploy->key_rotation",
        provenance=("post_mortem_42",),
        confidence=0.95,
    )
    step("FR-7 correction node written", revision.node_id)
    step("FR-7 prior_belief preserved", revision.prior_belief[:50] + "...")
    step("FR-7 update_operation", revision.update_operation)
    step("FR-7 confidence", revision.confidence)

    # --- Turn 7: budgeted inscription -- queue several candidate writes  --
    # --- and let utility scoring pick which to commit.                   --
    budgeted_agent = MemoryAgent(
        inscription_policy=UtilityWritePolicy(budget=2),
    )
    await budgeted_agent.aadd_candidate(
        "trivial log: heartbeat tick",
        source=SourceLabel.EXTERNAL,
        predicted_utility=0.1,
    )
    await budgeted_agent.aadd_candidate(
        "important: customer-impacting failure at 14:03",
        source=SourceLabel.EXTERNAL,
        predicted_utility=0.95,
    )
    await budgeted_agent.aadd_candidate(
        "medium: latency above SLO threshold",
        source=SourceLabel.EXTERNAL,
        predicted_utility=0.6,
    )
    committed = await budgeted_agent.aflush_inscriptions()
    step("FR-5 budget=2 committed", [r.content[:40] for r in committed])
    budgeted_agent.close()

    # --- Turn 8: standalone Source(.) inference helper ------------------
    label = await ainfer_source("therefore the handler is the dominant factor")
    step("FR-6 ainfer_source returns", label)

    # --- Turn 9: bulk-ingest a structured LLM output --------------------
    # Simulates what an LLM would emit after being prompted with
    # StructuredEnvelope.system_prompt_block(). Inline marker form here
    # is the easier fallback when JSON isn't strictly enforced.
    simulated_llm_output = """
    OBSERVED: rollback executed at 14:08 UTC
    TOOL: post-rollback monitoring returned cache_hit_rate=88%
    INFERRED: cache_hit_rate recovery confirms key-rotation hypothesis
    """
    envelope = StructuredEnvelope.parse(simulated_llm_output)
    step("v0.5 envelope observations parsed", len(envelope.observations))
    ingest_results = await agent.aingest_envelope(envelope)
    step("v0.5 bulk-ingested records", [r.record_id for r in ingest_results])

    # --- Audit before closing -------------------------------------------
    audit = await agent.aaudit_laundering(
        truth_ceilings={"hyp_cache_regression": SourceLabel.INFERENCE},
    )
    step("FR-9 audit n_records", audit.n_records)
    step("FR-9 audit local_rate", audit.local_laundering_rate)
    step("FR-9 audit truth_grounded_rate", audit.truth_grounded_rate)
    step("FR-9 audit is_clean", audit.is_clean)

    step("agent total records (incl operation memory)", len(agent))
    step("correction nodes", len(agent.correction_nodes()))

    agent.close()
    step("session 1 closed; SQLite vault flushed to disk")


async def session_two(db_path: str) -> None:
    heading("Session 2: fresh process reopens the same vault")

    agent = MemoryAgent(storage=SQLiteStorage(db_path))
    step("opened agent with no active self_index")
    step("records loaded from SQLite", len(agent))

    # Scoped records require the right active index to surface.
    result = await agent.aquery("cache regression after deploy")
    step("FR-8 retrieved with no index", [h.record.record_id for h in result.retrieved])

    agent.active_self_index = SelfIndex(
        user_id="alice",
        project_id="deploy",
        role="maintainer",
        permission_scope="prod",
    )
    result = await agent.aquery("cache regression after deploy")
    step("FR-8 retrieved with alice's index", [h.record.record_id for h in result.retrieved])

    # Correction chain survived.
    nodes = agent.correction_nodes()
    step("FR-7 correction nodes recovered", len(nodes))
    if nodes:
        step("FR-7 first node update_operation", nodes[0].update_operation)
        step("FR-7 first node revised_belief", nodes[0].revised_belief[:50] + "...")

    # Audit after reload.
    audit = await agent.aaudit_laundering(
        truth_ceilings={"hyp_cache_regression": SourceLabel.INFERENCE},
    )
    step("FR-9 audit after reload is_clean", audit.is_clean)

    agent.close()
    step("session 2 closed")


async def main() -> int:
    fd, db_path = tempfile.mkstemp(suffix=".trace-memory-walkthrough.db")
    os.close(fd)
    try:
        await session_one(db_path)
        await session_two(db_path)
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
