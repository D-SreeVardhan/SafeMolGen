"""Shared condition vector construction for generator conditioning.

Training and inference must use the same endpoint key order (sorted(admet.keys()))
and COND_DIM so condition vectors align. Use these helpers everywhere.
"""

from typing import Dict, Optional

import torch

from models.generator.transformer import COND_DIM

def get_target_condition_for_rl(device: str = "cpu") -> torch.Tensor:
    """Return a (1, COND_DIM) condition vector for RL: high phase probs, neutral ADMET.

    Used when sampling during RL so the policy learns to generate toward this profile.
    """
    return get_target_condition(device=device, phase=0.5)


def get_target_condition(
    device: str = "cpu",
    phase: float = 0.5,
    phase1: Optional[float] = None,
    phase2: Optional[float] = None,
    phase3: Optional[float] = None,
) -> torch.Tensor:
    """Return a (1, COND_DIM) target condition: neutral ADMET, desired phase probs.

    Used for target-condition pretraining (Option B): same phase for all three
    unless phase1/2/3 are set. E.g. phase=0.6 -> aim for higher success.
    """
    p1 = phase1 if phase1 is not None else phase
    p2 = phase2 if phase2 is not None else phase
    p3 = phase3 if phase3 is not None else phase
    n_admet = COND_DIM - 3
    vals = [0.5] * n_admet if n_admet > 0 else []
    while len(vals) < n_admet:
        vals.append(0.0)
    vals.extend([p1, p2, p3])
    vec = torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
    return vec


__all__ = [
    "COND_DIM",
    "build_condition_vector",
    "build_condition_vector_toward_target",
    "build_condition_vector_toward_target_phase_aware",
    "get_target_condition",
    "get_target_condition_for_rl",
]


def build_condition_vector(
    admet: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    device: str = "cpu",
) -> torch.Tensor:
    """Build a (1, COND_DIM) condition vector from ADMET dict and phase probs.

    Uses sorted(admet.keys())[:COND_DIM-3] for ADMET, then 3 phase dimensions.
    Callers must use the same endpoint list so key order is consistent.
    """
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    vals.extend([phase1, phase2, phase3])
    vec = torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
    return vec


def build_condition_vector_toward_target(
    admet: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    target_success: float = 0.5,
    blend: float = 0.92,
    device: str = "cpu",
    phase_boost: float = 0.0,
) -> torch.Tensor:
    """Build condition vector with phase probs nudged toward target (for steering).

    ADMET part unchanged; phase dims are blended toward target_phase so the
    generator is encouraged toward higher-success molecules. phase_boost raises
    the target over iterations so we aim higher as we go.
    """
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    # Steer toward phase probs that yield >50% success; target_success is calibrated (0.5 = 50%)
    target_phase = min(0.75, max(0.55, target_success * 1.2) + phase_boost)
    p1 = phase1 + blend * max(0.0, target_phase - phase1)
    p2 = phase2 + blend * max(0.0, target_phase - phase2)
    p3 = phase3 + blend * max(0.0, target_phase - phase3)
    vals.extend([p1, p2, p3])
    vec = torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
    return vec


def build_condition_vector_toward_target_phase_aware(
    admet: Dict[str, float],
    phase1: float,
    phase2: float,
    phase3: float,
    target_success: float = 0.5,
    blend: float = 0.92,
    worst_phase_blend: float = 0.98,
    device: str = "cpu",
    phase_boost: float = 0.0,
) -> torch.Tensor:
    """Build condition vector with extra push on the worst phase (Phase I/II/III).

    Identifies the worst phase (min of p1, p2, p3) and applies a stronger blend
    for that dimension so the generator is explicitly steered to improve it.
    """
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    target_phase = min(0.75, max(0.55, target_success * 1.2) + phase_boost)
    phases = [(phase1, 0), (phase2, 1), (phase3, 2)]
    worst_idx = min(phases, key=lambda x: x[0])[1]
    p1 = phase1 + (worst_phase_blend if worst_idx == 0 else blend) * max(0.0, target_phase - phase1)
    p2 = phase2 + (worst_phase_blend if worst_idx == 1 else blend) * max(0.0, target_phase - phase2)
    p3 = phase3 + (worst_phase_blend if worst_idx == 2 else blend) * max(0.0, target_phase - phase3)
    vals.extend([p1, p2, p3])
    vec = torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
    return vec
