"""Reward functions for SafeMolGen RL fine-tuning."""

from typing import Callable, Dict, List, Optional

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import QED
RDLogger.DisableLog("rdApp.error")


def validity_reward(smiles: str) -> float:
    return 1.0 if Chem.MolFromSmiles(smiles) is not None else 0.0


def qed_reward(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return float(QED.qed(mol))


def diversity_reward(smiles_list: List[str]) -> float:
    unique = len(set(smiles_list))
    return unique / max(len(smiles_list), 1)


def compute_reward_per_smiles(
    smiles: str,
    oracle_score_fn: Optional[Callable[[str], float]] = None,
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
) -> float:
    validity = validity_reward(smiles)
    qed = qed_reward(smiles)
    oracle = oracle_score_fn(smiles) if oracle_score_fn else 0.0
    return w_validity * validity + w_qed * qed + w_oracle * oracle


def compute_rewards_per_sample(
    smiles_list: List[str],
    oracle_score_fn: Optional[Callable[[str], float]] = None,
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
    w_diversity: float = 0.1,
) -> List[float]:
    base = [
        compute_reward_per_smiles(
            s,
            oracle_score_fn=oracle_score_fn,
            w_validity=w_validity,
            w_qed=w_qed,
            w_oracle=w_oracle,
        )
        for s in smiles_list
    ]
    diversity = diversity_reward(smiles_list)
    return [b + (w_diversity * diversity) for b in base]


def compute_rewards(
    smiles_list: List[str],
    oracle_score_fn: Optional[Callable[[str], float]] = None,
    w_validity: float = 0.3,
    w_qed: float = 0.3,
    w_oracle: float = 0.3,
    w_diversity: float = 0.1,
) -> Dict[str, float]:
    validity = sum(validity_reward(s) for s in smiles_list) / max(len(smiles_list), 1)
    qed = sum(qed_reward(s) for s in smiles_list) / max(len(smiles_list), 1)
    oracle = 0.0
    if oracle_score_fn:
        oracle = sum(oracle_score_fn(s) for s in smiles_list) / max(len(smiles_list), 1)
    diversity = diversity_reward(smiles_list)
    total = w_validity * validity + w_qed * qed + w_oracle * oracle + w_diversity * diversity
    return {
        "validity": validity,
        "qed": qed,
        "oracle": oracle,
        "diversity": diversity,
        "total": total,
    }
