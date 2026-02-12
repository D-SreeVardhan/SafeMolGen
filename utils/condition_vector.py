"""Shared condition vector construction for generator conditioning.

Training and inference must use the same endpoint key order (sorted(admet.keys()))
and COND_DIM so condition vectors align. Use these helpers everywhere.
"""

from typing import Dict

import torch

from models.generator.transformer import COND_DIM

def get_target_condition_for_rl(device: str = "cpu") -> torch.Tensor:
    """Return a (1, COND_DIM) condition vector for RL: high phase probs, neutral ADMET.

    Used when sampling during RL so the policy learns to generate toward this profile.
    """
    # ADMET: neutral (0.5 for classification-like, or zeros)
    n_admet = COND_DIM - 3
    vals = [0.5] * n_admet if n_admet > 0 else []
    while len(vals) < n_admet:
        vals.append(0.0)
    # Phase: target toward ~25% overall (e.g. 0.5 * 0.5 * 0.5) so 0.5, 0.5, 0.5
    vals.extend([0.5, 0.5, 0.5])
    vec = torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
    return vec


__all__ = [
    "COND_DIM",
    "build_condition_vector",
    "build_condition_vector_toward_target",
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
    target_success: float = 0.25,
    blend: float = 0.25,
    device: str = "cpu",
) -> torch.Tensor:
    """Build condition vector with phase probs nudged toward target (for steering).

    ADMET part unchanged; phase dims are blended toward target_phase so the
    generator is encouraged toward higher-success molecules.
    """
    keys = sorted(admet.keys())[: COND_DIM - 3]
    vals = [float(admet.get(k, 0.0)) for k in keys]
    while len(vals) < COND_DIM - 3:
        vals.append(0.0)
    target_phase = min(0.5, target_success * 1.2)
    p1 = phase1 + blend * max(0.0, target_phase - phase1)
    p2 = phase2 + blend * max(0.0, target_phase - phase2)
    p3 = phase3 + blend * max(0.0, target_phase - phase3)
    vals.extend([p1, p2, p3])
    vec = torch.tensor([vals[:COND_DIM]], dtype=torch.float32, device=device)
    return vec
