"""Structural alerts database for toxicity prediction.

Alerts are loaded from data/structural_alerts.csv when present; otherwise
the built-in set is used. See scripts/download_structural_alerts.py to
fetch the Hamburg SMARTS dataset (PAINS, Enoch) and regenerate the CSV.
"""

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from loguru import logger

_DEFAULT_ALERTS_PATH = Path(__file__).resolve().parents[2] / "data" / "structural_alerts.csv"


@dataclass
class StructuralAlert:
    name: str
    smarts: str
    category: str
    severity: str
    recommendation: str

    def pattern(self):
        return Chem.MolFromSmarts(self.smarts)


_BUILTIN_STRUCTURAL_ALERTS: Dict[str, StructuralAlert] = {
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


def load_structural_alerts_from_csv(path: Path) -> Dict[str, StructuralAlert]:
    """Load structural alerts from a CSV with columns: id, name, smarts, category, severity, recommendation.
    Rows with invalid SMARTS are skipped. Returns dict keyed by id."""
    result: Dict[str, StructuralAlert] = {}
    if not path.exists():
        return result
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = (row.get("id") or "").strip()
                name = (row.get("name") or "").strip()
                smarts = (row.get("smarts") or "").strip()
                if not uid or not name or not smarts:
                    continue
                category = (row.get("category") or "").strip() or "general"
                severity = (row.get("severity") or "").strip() or "medium"
                recommendation = (row.get("recommendation") or "").strip() or "Review substructure"
                if Chem.MolFromSmarts(smarts) is None:
                    logger.warning(f"Invalid SMARTS for alert {uid}, skipping")
                    continue
                result[uid] = StructuralAlert(
                    name=name,
                    smarts=smarts,
                    category=category,
                    severity=severity,
                    recommendation=recommendation,
                )
    except Exception as e:
        logger.warning(f"Could not load structural alerts from {path}: {e}")
    return result


def _get_structural_alerts_db() -> Dict[str, StructuralAlert]:
    loaded = load_structural_alerts_from_csv(_DEFAULT_ALERTS_PATH)
    if loaded:
        return loaded
    return _BUILTIN_STRUCTURAL_ALERTS.copy()


STRUCTURAL_ALERTS_DB: Dict[str, StructuralAlert] = _get_structural_alerts_db()


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
