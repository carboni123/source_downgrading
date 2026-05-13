"""End-to-end anti-laundering demo using trace-memory's public API.

Demonstrates the five laundering scenarios from the validation fixture,
using only the v0.1 public API. The demo:

1. Plants seed records (external observations, simulations, fabrications).
2. Derives new records via ``add_derived(...)`` -- source is computed
   from contributing inputs by the source-downgrading rule.
3. Queries the agent for the derived claims and observes the routing
   decision (trusted vs quarantined).
4. Runs ``audit_laundering(...)`` with truth ceilings supplied by the
   demo to show the paired metric.

A "fail" scenario at the end shows what would happen WITHOUT trace-memory:
if a derived claim were re-asserted as an external observation via the
naive ``add(content, source=EXTERNAL)`` call, the audit would catch it
via the truth-grounded metric even though the self-referential metric
might not flag it.

Run:

    python examples/anti_laundering_demo.py
"""
from __future__ import annotations

from trace_memory import MemoryAgent, SourceLabel


def main() -> None:
    # k=1 so each query retrieves its intended target only. Larger k
    # surfaces co-retrieved records that may trigger conservative
    # routing (see footnote at the end of this demo).
    agent = MemoryAgent(retrieval_k=1)

    # 1. Plant seed records.
    r_e1 = agent.add(
        "external observation: deploy migration succeeded under load",
        source=SourceLabel.EXTERNAL,
        provenance=("deploy_log_42",),
        record_id="E1",
    )
    r_e2 = agent.add(
        "external observation: rollback restored service in prior incident",
        source=SourceLabel.EXTERNAL,
        provenance=("incident_review_17",),
        record_id="E2",
    )
    r_s1 = agent.add(
        "hypothetical: a hotfix could avoid rollback if traffic is low",
        source=SourceLabel.SIMULATION,
        provenance=("simulation_branch_a",),
        source_confidence=0.7,
        record_id="S1",
    )
    r_f1 = agent.add(
        "fabricated rumor: legal forbids rollbacks during business hours",
        source=SourceLabel.FABRICATED_OR_UNCERTAIN,
        provenance=("adversarial_note",),
        source_confidence=0.9,
        record_id="F1",
    )

    print("Seed records:")
    for record in [r_e1, r_e2, r_s1, r_f1]:
        print(f"  {record.record_id:>3}  source={record.source_label:>26s}  "
              f"prov={record.provenance}")
    print()

    # 2. Derive new records via add_derived(...).
    derivations = []

    # Pure inference from observations: should cap at INFERENCE.
    d_pure = agent.add_derived(
        "inferred: rollback is generally safe after migrations",
        inputs=[r_e1, r_e2],
        record_id="D_pure",
    )
    derivations.append((d_pure, "infer from 2x EXTERNAL", SourceLabel.INFERENCE))

    # Derived from simulation + external: should cap at SIMULATION.
    d_sim = agent.add_derived(
        "inferred: skip rollback and ship hotfix instead",
        inputs=[r_s1, r_e1],
        record_id="D_sim",
    )
    derivations.append((d_sim, "infer from SIMULATION + EXTERNAL", SourceLabel.SIMULATION))

    # Derived from fabrication + external: should cap at FABRICATED.
    d_fab = agent.add_derived(
        "inferred: defer rollback until business hours per legal policy",
        inputs=[r_f1, r_e1],
        record_id="D_fab",
    )
    derivations.append((d_fab, "infer from FABRICATED + EXTERNAL", SourceLabel.FABRICATED_OR_UNCERTAIN))

    # Chained inference: D_step1 from E1, then D_step2 from D_step1.
    d_step1 = agent.add_derived(
        "inferred: prior deploys followed by rollback rarely re-failed",
        inputs=[r_e1],
        record_id="D_step1",
    )
    d_step2 = agent.add_derived(
        "inferred: rollback then redeploy is a safe pattern",
        inputs=[d_step1],
        record_id="D_step2",
    )
    derivations.append((d_step2, "chained inference (E1 -> D_step1 -> D_step2)", SourceLabel.INFERENCE))

    print("Derivations:")
    for record, description, expected in derivations:
        ok = record.source_label == expected.value
        marker = "  ok" if ok else "  FAIL"
        print(f"  {record.record_id:>10}  {marker}  source={record.source_label:>26s}  "
              f"expected={expected.value}  ({description})")
    print()

    # 3. Query the agent for each derivation and observe routing.
    print("Routing under query:")
    queries = [
        ("is rollback generally safe after migrations", "D_pure",
         "should route to a trusted target (chain is honest)"),
        ("should we skip rollback and ship hotfix instead", "D_sim",
         "should quarantine (simulation contamination)"),
        ("should we defer rollback until business hours per legal policy", "D_fab",
         "should quarantine (fabricated contamination)"),
    ]
    for query, target_id, comment in queries:
        result = agent.query(query)
        retrieved_ids = [hit.record.record_id for hit in result.retrieved]
        print(f"  query={query!r}")
        print(f"    retrieved={retrieved_ids}")
        print(f"    selected_route={result.selected_route}")
        print(f"    fold_force={result.fold_force:.3f}  gated={result.gated}")
        print(f"    expectation: {comment}")
    print()

    # 4. Audit with truth ceilings (the demo knows the right answer).
    truth = {
        "D_pure": SourceLabel.INFERENCE,
        "D_sim": SourceLabel.SIMULATION,
        "D_fab": SourceLabel.FABRICATED_OR_UNCERTAIN,
        "D_step1": SourceLabel.INFERENCE,
        "D_step2": SourceLabel.INFERENCE,
    }
    audit = agent.audit_laundering(truth_ceilings=truth)
    print("Audit (with truth ceilings):")
    print(f"  n_records={audit.n_records}")
    print(f"  local_laundering_rate={audit.local_laundering_rate:.3f}")
    print(f"  truth_grounded_rate={audit.truth_grounded_rate:.3f}")
    print(f"  gap={audit.gap:.3f}")
    print(f"  cascade_invisibility_warning={audit.cascade_invisibility_warning}")
    print(f"  is_clean={audit.is_clean}")

    # 5. Counter-demo: try to launder a derived claim by re-asserting it
    #    as an external observation. The truth-grounded audit catches this.
    print()
    print("Counter-demo: laundering via re-assertion as EXTERNAL")
    bad_record = agent.add(
        "rollback is generally safe after migrations",
        source=SourceLabel.EXTERNAL,
        record_id="LAUNDERED",
    )
    # Now claim this LAUNDERED record is derived from D_sim (which is
    # the simulation-contaminated derivation) -- truth-grounded would
    # be SIMULATION; the caller asserted EXTERNAL.
    bad_truth = dict(truth)
    bad_truth["LAUNDERED"] = SourceLabel.SIMULATION
    # Mark LAUNDERED as derived for audit by pretending the caller
    # routed it through add_derived. The store doesn't know it's bad
    # without truth ceilings.
    agent._derived_record_ids.add("LAUNDERED")
    bad_audit = agent.audit_laundering(truth_ceilings=bad_truth)
    print(f"  truth_grounded_rate={bad_audit.truth_grounded_rate:.3f}  "
          f"(detected the laundering via truth-grounded ceiling)")
    print(f"  local_laundering_rate={bad_audit.local_laundering_rate:.3f}  "
          f"(would have missed it without truth ceilings)")
    print(f"  is_clean={bad_audit.is_clean}")
    print()
    print("note 1: in practice, callers should never bypass add_derived(...). "
          "The counter-demo simulates a bug or attack by patching internal "
          "state directly; the public API does not enable this path.")
    print()
    print("note 2: when retrieval_k > 1, an honest query may co-retrieve an "
          "untrusted record from the store. The routing policy is defensive "
          "and quarantines any retrieval set containing untrusted sources. "
          "This is the validated behavior -- a feature, not a bug. To see "
          "trusted routes on honest queries, either use k=1 or pre-filter "
          "the store by source before query.")


if __name__ == "__main__":
    main()
