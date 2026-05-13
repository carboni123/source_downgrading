"""LLM Experiment 1: Fold-Gating vs Store-Everything with real Claude calls.

Uses answer-quality fold-force: measures how much the LLM's response improves
toward the correct answer when memory is present. Relevant memories produce
high fold-force (~0.23 accuracy gain); noise memories produce near-zero or
negative fold-force. The fold-gated agent prunes low-fold-force records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fgm.core import FGMAgent, MarginRetriever, Array, EPS, cosine
from fgm.llm import LLMTransition, answer_quality_fold_force, CallStats


@dataclass(frozen=True)
class LLMPhaseMetrics:
    phase: int
    n_content_records: int
    hit_rate: float
    mean_margin: float
    mean_fold_force: float
    llm_calls: int


@dataclass(frozen=True)
class LLMExperimentResult:
    baseline_phases: List[LLMPhaseMetrics]
    foldgated_phases: List[LLMPhaseMetrics]
    prediction_holds: bool
    total_llm_calls: int
    total_tokens: Dict[str, int]


SIGNAL_RECORDS = [
    ("sig_pg_vacuum", "PostgreSQL VACUUM caused 10-second lock waits on users table due to transaction ID wraparound at 2 billion transactions"),
    ("sig_redis_ttl", "Redis cache hit ratio dropped to 40 percent after deployment because TTL was accidentally set to 5 seconds instead of 300"),
    ("sig_nginx_504", "Nginx upstream timeout at 60 seconds caused 504 gateway errors during peak traffic of 10 thousand requests per second"),
    ("sig_oom_leak", "Memory leak in user session handler accumulated 8GB of heap over 72 hours causing OOM kill on production web server 3"),
    ("sig_dns_fail", "DNS resolution failures caused by recursive resolver hitting rate limit of 1000 queries per second from the monitoring system"),
    ("sig_tls_expire", "TLS certificate for api.example.com expired causing all HTTPS connections to fail with certificate validation error"),
]

SIGNAL_QUERIES = [
    ("sig_pg_vacuum", "What caused the PostgreSQL lock waits on the users table?"),
    ("sig_redis_ttl", "Why did the Redis cache hit ratio drop after the deployment?"),
    ("sig_nginx_504", "What caused the 504 gateway errors during peak traffic?"),
    ("sig_oom_leak", "What caused the out of memory crash on the production web server?"),
    ("sig_dns_fail", "What caused the DNS resolution failures?"),
    ("sig_tls_expire", "Why did all HTTPS connections start failing?"),
]

NOISE_POOL = [
    "Database performance check completed. All queries within normal latency thresholds.",
    "Server health check passed. CPU utilization at 45 percent, memory at 62 percent.",
    "Routine database backup completed successfully at 2:00 AM UTC.",
    "Network connectivity test passed for all three availability zones.",
    "Application deployment pipeline executed successfully. No rollback needed.",
    "Cache warmup procedure completed. Hit ratio stable at 95 percent.",
    "Log rotation completed for all application servers. Old logs archived.",
    "SSL certificate renewal check passed. All certificates valid for 60 plus days.",
    "Database replication lag within acceptable bounds at 200 milliseconds.",
    "Automated security scan completed. No critical vulnerabilities detected.",
    "Load balancer health checks passing for all backend instances.",
    "Storage utilization at 55 percent across all production volumes.",
    "API response time p99 within SLA at 450 milliseconds.",
    "Scheduled maintenance window completed without incidents.",
    "Monitoring alert thresholds verified against current baselines.",
]


class LLMExperiment1:
    """Fold-gating vs store-everything with real LLM transitions."""

    def __init__(
        self,
        llm_call: Callable[[str], str],
        embed_fn: Callable[[str], np.ndarray],
        dim: int = 384,
        n_phases: int = 4,
        noise_per_phase: int = 10,
        fold_threshold: float = 0.05,
        k: int = 3,
    ):
        self.llm_call = llm_call
        self.embed_fn = embed_fn
        self.dim = dim
        self.n_phases = n_phases
        self.noise_per_phase = noise_per_phase
        self.fold_threshold = fold_threshold
        self.k = k

    def run(self) -> LLMExperimentResult:
        usage_bl: Dict[str, int] = {}
        usage_fg: Dict[str, int] = {}

        transition_bl = LLMTransition(self.llm_call, self.embed_fn, dim=self.dim)
        transition_fg = LLMTransition(self.llm_call, self.embed_fn, dim=self.dim)

        baseline = FGMAgent(
            dim=self.dim, transition_fn=transition_bl, embed_fn=self.embed_fn,
            fold_threshold=self.fold_threshold, retrieval_k=self.k,
            auto_compress=False, fold_force_fn=answer_quality_fold_force,
        )
        foldgated = FGMAgent(
            dim=self.dim, transition_fn=transition_fg, embed_fn=self.embed_fn,
            fold_threshold=self.fold_threshold, retrieval_k=self.k,
            auto_compress=False, fold_force_fn=answer_quality_fold_force,
        )

        for rid, content in SIGNAL_RECORDS:
            baseline.add(content, record_id=rid)
            foldgated.add(content, record_id=rid)

        baseline_phases: List[LLMPhaseMetrics] = []
        foldgated_phases: List[LLMPhaseMetrics] = []
        rng = np.random.default_rng(42)
        noise_idx = 0

        for phase in range(self.n_phases):
            for _ in range(self.noise_per_phase):
                text = NOISE_POOL[noise_idx % len(NOISE_POOL)]
                nid = f"noise_{noise_idx}"
                baseline.add(text, record_id=nid, metadata={"phase_added": phase})
                foldgated.add(text, record_id=nid, metadata={"phase_added": phase})
                noise_idx += 1

            calls_before_bl = transition_bl.stats.n_calls
            calls_before_fg = transition_fg.stats.n_calls

            for _pass in range(2):
                for target_id, query in SIGNAL_QUERIES:
                    baseline.query(query, k=self.k)
                    foldgated.query(query, k=self.k)

            self._prune_low_fold(foldgated)

            bl_calls = transition_bl.stats.n_calls - calls_before_bl
            fg_calls = transition_fg.stats.n_calls - calls_before_fg

            baseline_phases.append(self._measure(baseline, phase, bl_calls))
            foldgated_phases.append(self._measure(foldgated, phase, fg_calls))

        prediction_holds = (
            foldgated_phases[-1].hit_rate > baseline_phases[-1].hit_rate
            and foldgated_phases[-1].n_content_records < baseline_phases[-1].n_content_records
        )

        total_calls = transition_bl.stats.n_calls + transition_fg.stats.n_calls

        return LLMExperimentResult(
            baseline_phases=baseline_phases,
            foldgated_phases=foldgated_phases,
            prediction_holds=prediction_holds,
            total_llm_calls=total_calls,
            total_tokens={"note": "see transition.stats for per-agent details"},
        )

    def _prune_low_fold(self, agent: FGMAgent):
        """Prune content records that were retrieved but had low fold-force.

        This is the principled criterion: a record was tested (retrieved and
        folded) and didn't change the agent's answer quality. It's storage
        without cognitive function.
        """
        to_remove = []
        for rec in agent.store.all_records():
            if rec.record_type != "content":
                continue
            uses = agent.store.use_count(rec.record_id)
            if uses == 0:
                continue
            avg_force = agent.store.total_fold_force(rec.record_id) / uses
            if avg_force < self.fold_threshold:
                to_remove.append(rec.record_id)
        for rid in to_remove:
            agent.store.remove(rid)

    def _measure(self, agent: FGMAgent, phase: int, llm_calls: int) -> LLMPhaseMetrics:
        content_records = [r for r in agent.store.all_records() if r.record_type == "content"]

        hits = 0
        margins = []
        fold_forces = []

        for target_id, query in SIGNAL_QUERIES:
            q_vec = self.embed_fn(query)
            rank, _, margin = agent.retriever.target_rank_margin(q_vec, target_id)
            hits += int(rank is not None and rank <= self.k)
            margins.append(margin if np.isfinite(margin) else 0.0)

        n_queries = len(SIGNAL_QUERIES)
        recent_folds = agent._fold_history[-n_queries:] if agent._fold_history else []
        fold_forces = [f.fold_force for f in recent_folds]

        return LLMPhaseMetrics(
            phase=phase,
            n_content_records=len(content_records),
            hit_rate=hits / max(n_queries, 1),
            mean_margin=float(np.mean(margins)) if margins else 0.0,
            mean_fold_force=float(np.mean(fold_forces)) if fold_forces else 0.0,
            llm_calls=llm_calls,
        )
