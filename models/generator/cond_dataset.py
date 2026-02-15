"""Conditioned SMILES dataset for pretraining with real Oracle/ADMET condition vectors."""

from typing import Dict, List, Optional

import torch

from models.generator.tokenizer import SMILESTokenizer
from utils.condition_vector import COND_DIM, build_condition_vector


class ImitationDataset(torch.utils.data.Dataset):
    """Dataset of (condition, smiles) pairs for imitation: maximize log P(smiles | condition)."""

    def __init__(
        self,
        pairs: List[tuple],
        tokenizer: SMILESTokenizer,
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        cond_list, smi = self.pairs[idx]
        ids = self.tokenizer.encode(smi)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        cond = torch.tensor(cond_list[:COND_DIM], dtype=torch.float32)
        if cond.numel() < COND_DIM:
            cond = torch.nn.functional.pad(cond, (0, COND_DIM - cond.numel()), value=0.0)
        return input_ids, target_ids, cond


class TargetCondSMILESDataset(torch.utils.data.Dataset):
    """Dataset that returns (input_ids, target_ids, condition) with a FIXED target condition.

    Used for Option B: train the generator to associate a desired (high) condition
    with high-oracle SMILES, so at inference P(smiles | target_condition) yields
    molecules in oracle-success space. Condition is the same for every sample.
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        target_condition: torch.Tensor,
    ):
        self.smiles = list(smiles_list)
        self.tokenizer = tokenizer
        # Store (COND_DIM,) on CPU; dataloader will stack to (B, COND_DIM)
        if target_condition.dim() == 2:
            self._cond = target_condition.squeeze(0).cpu().to(torch.float32)
        else:
            self._cond = target_condition.cpu().to(torch.float32)
        assert self._cond.shape == (COND_DIM,), f"Expected (COND_DIM,) got {self._cond.shape}"

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        ids = self.tokenizer.encode(smi)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, target_ids, self._cond.clone()


class CondSMILESDataset(torch.utils.data.Dataset):
    """Dataset that returns (input_ids, target_ids, condition) per SMILES.

    Condition is computed from Oracle prediction (ADMET + phase probs) so the
    generator learns to associate condition vectors with molecule sequences.
    Invalid or failed predictions use a zero condition. Results are cached by
    SMILES to avoid recomputing.
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        oracle,  # DrugOracle
        device: str = "cpu",
        cache: Optional[Dict[str, torch.Tensor]] = None,
        zero_condition_fraction: float = 0.0,
    ):
        self.smiles = list(smiles_list)
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.device = device
        # Store conditions on CPU in cache to avoid GPU memory; batch is moved in trainer
        self._cache = cache if cache is not None else {}
        self.zero_condition_fraction = zero_condition_fraction
        self._rng = None  # lazy init for optional randomness

    def _get_condition(self, smi: str) -> torch.Tensor:
        if smi in self._cache:
            return self._cache[smi]
        pred = self.oracle.predict(smi)
        if pred is None:
            cond = torch.zeros(COND_DIM, dtype=torch.float32)
        else:
            vec = build_condition_vector(
                pred.admet_predictions or {},
                pred.phase1_prob,
                pred.phase2_prob,
                pred.phase3_prob,
                device="cpu",
            )
            cond = vec.squeeze(0)  # (COND_DIM,)
        self._cache[smi] = cond
        return cond

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        ids = self.tokenizer.encode(smi)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        if self.zero_condition_fraction > 0:
            if self._rng is None:
                import random
                self._rng = random.Random(42)
            if self._rng.random() < self.zero_condition_fraction:
                cond = torch.zeros(COND_DIM, dtype=torch.float32)
            else:
                cond = self._get_condition(smi)
        else:
            cond = self._get_condition(smi)
        return input_ids, target_ids, cond
