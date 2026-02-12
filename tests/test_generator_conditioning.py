"""Tests that generator conditioning affects output (condition=zeros vs non-zero).

When the generator is trained with real condition vectors, different conditions
should produce different molecules. With same seed, condition=zeros vs
condition=non_zero should yield different SMILES.
"""

import os
import random
from pathlib import Path

import pytest
import torch

from models.generator.safemolgen import SafeMolGen
from models.generator.transformer import COND_DIM
from utils.condition_vector import build_condition_vector_toward_target


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _generator_available() -> bool:
    root = _project_root()
    return (root / "checkpoints" / "generator" / "model.pt").exists()


@pytest.mark.skipif(not _generator_available(), reason="checkpoints/generator not found")
def test_condition_changes_output():
    """With cond_dim > 0, generating with zeros vs non-zero condition should differ (same seed)."""
    root = _project_root()
    gen = SafeMolGen.from_pretrained(str(root / "checkpoints" / "generator"), device="cpu")
    if getattr(gen.model, "cond_dim", 0) == 0:
        pytest.skip("Generator has cond_dim=0; conditioning not supported")

    seed = 42
    n = 15
    device = "cpu"

    # Fixed non-zero condition (target-like: high phase probs)
    admet = {f"k{i}": 0.5 for i in range(COND_DIM - 3)}
    non_zero_cond = build_condition_vector_toward_target(
        admet, 0.4, 0.35, 0.3, target_success=0.25, device=device
    )

    def run(condition):
        torch.manual_seed(seed)
        random.seed(seed)
        return gen.generate(
            n=n,
            temperature=0.75,
            top_k=40,
            device=device,
            condition=condition,
        )

    with torch.no_grad():
        mols_zeros = run(condition=torch.zeros(1, COND_DIM, dtype=torch.float32))
        mols_nonzero = run(condition=non_zero_cond)

    assert len(mols_zeros) == n and len(mols_nonzero) == n
    # Different condition should yield different molecules (at least one difference)
    assert mols_zeros != mols_nonzero, (
        "Expected condition=zeros and condition=non_zero to produce different outputs; "
        "model may not be trained to use conditioning."
    )


def test_condition_vector_shape():
    """Shared helper produces (1, COND_DIM) tensor."""
    admet = {"a": 0.1, "b": 0.2}
    vec = build_condition_vector_toward_target(admet, 0.5, 0.4, 0.3, device="cpu")
    assert vec.shape == (1, COND_DIM)
    assert vec.dtype == torch.float32
