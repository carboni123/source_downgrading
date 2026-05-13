"""Toy coupled-field intervention probes.

These probes test the minimal empirical claims of MAFC at controlled scale:
memory changes attention, attention changes write pressure, novelty can break
lock-in, and source discounting reduces echo amplification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoupledFieldReport:
    attention_shift_after_memory_ablation: float
    write_shift_after_attention_ablation: float
    novelty_breakthrough_threshold: float
    lock_in_rate: float
    echo_amplification_rate: float


def coupled_field_probe(
    *,
    source_aware: bool = True,
    attention_beta: float = 4.0,
    write_gain: float = 6.0,
    breakthrough_p: float = 0.45,
) -> CoupledFieldReport:
    """Run a deterministic two-attractor coupled-field intervention."""
    external_signal = np.array([0.15, 0.20, 0.05])
    memory_bias = np.array([0.80, 0.05, 0.00])
    internal_source_discount = 0.15 if source_aware else 1.0

    attention_with_memory = _softmax(attention_beta * (external_signal + memory_bias * internal_source_discount))
    attention_without_memory = _softmax(attention_beta * external_signal)
    attention_shift = float(np.linalg.norm(attention_with_memory - attention_without_memory))

    write_with_attention = _sigmoid(write_gain * (attention_with_memory[0] - 0.35))
    write_without_attention = _sigmoid(write_gain * (0.0 - 0.35))
    write_shift = abs(write_with_attention - write_without_attention)

    threshold = _novelty_breakthrough_threshold(
        base_signal=external_signal + memory_bias * internal_source_discount,
        beta=attention_beta,
        target_index=1,
        p=breakthrough_p,
    )
    lock_in = float(attention_with_memory[0] > 0.85 and threshold > 0.5)
    echo = float(write_with_attention > 0.5 and not source_aware)

    return CoupledFieldReport(
        attention_shift_after_memory_ablation=attention_shift,
        write_shift_after_attention_ablation=float(write_shift),
        novelty_breakthrough_threshold=float(threshold),
        lock_in_rate=lock_in,
        echo_amplification_rate=echo,
    )


def _novelty_breakthrough_threshold(
    *,
    base_signal: np.ndarray,
    beta: float,
    target_index: int,
    p: float,
) -> float:
    for novelty in np.linspace(0.0, 2.0, 401):
        candidate = base_signal.copy()
        candidate[target_index] += novelty
        if _softmax(beta * candidate)[target_index] >= p:
            return float(novelty)
    return float("inf")


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))
