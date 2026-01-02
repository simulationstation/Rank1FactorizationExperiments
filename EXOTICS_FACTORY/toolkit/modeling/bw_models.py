"""Breit-Wigner model helpers (evaluation only)."""
from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Iterable


@dataclass
class BWParams:
    mass: float
    width: float
    phase: float = 0.0
    amplitude: float = 1.0


def breit_wigner(m: float, params: BWParams) -> complex:
    """Complex Breit-Wigner amplitude."""
    denom = (params.mass**2 - m**2) - 1j * params.mass * params.width
    return params.amplitude * cmath.exp(1j * params.phase) / denom


def coherent_two_bw(m: float, bw1: BWParams, bw2: BWParams) -> float:
    """Coherent sum of two BW amplitudes (intensity)."""
    amp = breit_wigner(m, bw1) + breit_wigner(m, bw2)
    return abs(amp) ** 2


def shared_subspace_model(
    m: float,
    channel_a: Iterable[BWParams],
    channel_b: Iterable[BWParams],
) -> dict[str, float]:
    """Shared-subspace model where channel A may have more poles than B."""
    amp_a = sum(breit_wigner(m, bw) for bw in channel_a)
    amp_b = sum(breit_wigner(m, bw) for bw in channel_b)
    return {"channel_a": abs(amp_a) ** 2, "channel_b": abs(amp_b) ** 2}


def one_pole_shared_shape(m: float, bw: BWParams, scale_a: float, scale_b: float) -> dict[str, float]:
    """Single shared pole with channel-specific scales."""
    intensity = abs(breit_wigner(m, bw)) ** 2
    return {"channel_a": intensity * scale_a, "channel_b": intensity * scale_b}
