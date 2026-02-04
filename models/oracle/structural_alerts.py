"""Structural alerts database for toxicity prediction."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from loguru import logger


@dataclass
class StructuralAlert:
    name: str
    smarts: str
    category: str
    severity: str
    recommendation: str

    def pattern(self):
        return Chem.MolFromSmarts(self.smarts)


STRUCTURAL_ALERTS_DB: Dict[str, StructuralAlert] = {
    "nitro_aromatic": StructuralAlert(
        name="Aromatic Nitro",
        smarts="[$(c1ccccc1[N+](=O)[O-]),$(c1ccncc1[N+](=O)[O-]),$(c1cnccc1[N+](=O)[O-])]",
        category="mutagenicity",
        severity="high",
        recommendation="Replace -NO2 with -CN or -CF3",
    ),
    "aromatic_amine": StructuralAlert(
        name="Aromatic Amine (Aniline)",
        smarts="[NH2,NH1,NH0;!$(N-C=O)]c1ccccc1",
        category="mutagenicity",
        severity="high",
        recommendation="Convert to amide or replace with -OH/-OCH3",
    ),
    "nitroso": StructuralAlert(
        name="Nitroso Group",
        smarts="[#6]N=O",
        category="mutagenicity",
        severity="critical",
        recommendation="Remove nitroso group",
    ),
    "azo": StructuralAlert(
        name="Azo Compound",
        smarts="[#6]N=N[#6]",
        category="mutagenicity",
        severity="medium",
        recommendation="Replace azo linkage with amide/ether",
    ),
    "epoxide": StructuralAlert(
        name="Epoxide",
        smarts="C1OC1",
        category="reactivity",
        severity="medium",
        recommendation="Avoid epoxide ring",
    ),
}


def detect_structural_alerts(smiles: str) -> Tuple[List[str], np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], None
    hits = []
    alert_atoms = np.zeros(mol.GetNumAtoms(), dtype=int)
    for key, alert in STRUCTURAL_ALERTS_DB.items():
        pattern = alert.pattern()
        if pattern is None:
            logger.warning(f"Invalid SMARTS for alert: {key}")
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            hits.append(alert.name)
            for match in matches:
                for idx in match:
                    alert_atoms[idx] = 1
    return hits, alert_atoms
