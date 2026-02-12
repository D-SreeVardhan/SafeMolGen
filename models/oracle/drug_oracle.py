"""DrugOracle main model and prediction dataclasses."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from models.admet.inference import load_model, predict_smiles
from models.oracle.phase_predictors import CascadedPhasePredictors
from models.oracle.recommender import generate_recommendations
from models.oracle.structural_alerts import detect_structural_alerts
from utils.chemistry import calculate_properties, validate_smiles


@dataclass
class RiskFactor:
    name: str
    category: str
    description: str
    impact: float
    source: str


@dataclass
class OraclePrediction:
    phase1_prob: float
    phase2_prob: float
    phase3_prob: float
    overall_prob: float
    admet_predictions: Dict[str, float]
    risk_factors: List[RiskFactor]
    structural_alerts: List[str]
    recommendations: List[Dict]
    alert_atoms: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "phase1_prob": self.phase1_prob,
            "phase2_prob": self.phase2_prob,
            "phase3_prob": self.phase3_prob,
            "overall_prob": self.overall_prob,
            "admet_predictions": self.admet_predictions,
            "risk_factors": [rf.__dict__ for rf in self.risk_factors],
            "structural_alerts": self.structural_alerts,
            "recommendations": self.recommendations,
        }


class DrugOracle:
    """Clinical phase predictor with ADMET-guided explainability."""

    def __init__(
        self,
        oracle_model: CascadedPhasePredictors,
        admet_model,
        endpoint_task_types: Dict[str, str],
        device: str = "cpu",
    ):
        self.oracle_model = oracle_model.to(device)
        self.admet_model = admet_model
        self.endpoint_task_types = endpoint_task_types
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        oracle_path: str,
        admet_path: str,
        endpoint_names: List[str],
        endpoint_task_types: Dict[str, str],
        input_dim: int,
        device: str = "cpu",
    ) -> "DrugOracle":
        admet_model = load_model(
            checkpoint_path=admet_path,
            endpoint_names=endpoint_names,
            num_node_features=input_dim,
            hidden_dim=128,
            num_layers=3,
            dropout=0.1,
            device=device,
        )
        state = torch.load(oracle_path, map_location=device, weights_only=False)
        model_state = state.get("model", state)
        w = model_state["phase1.net.0.weight"]
        in_dim_ckpt = int(w.shape[1])
        hidden_dim_ckpt = int(w.shape[0])
        oracle = CascadedPhasePredictors(in_dim=in_dim_ckpt, hidden_dim=hidden_dim_ckpt).to(device)
        oracle.load_state_dict(model_state, strict=True)
        oracle.eval()
        return cls(oracle, admet_model, endpoint_task_types, device=device)

    def _predict_oracle(self, admet_preds: Dict[str, float]) -> Dict[str, float]:
        self.oracle_model.eval()
        x = torch.tensor([list(admet_preds.values())], dtype=torch.float).to(self.device)
        with torch.no_grad():
            p1, p2, p3 = self.oracle_model(x)
        p1 = float(torch.sigmoid(p1).cpu().numpy())
        p2 = float(torch.sigmoid(p2).cpu().numpy())
        p3 = float(torch.sigmoid(p3).cpu().numpy())
        return {"phase1": p1, "phase2": p2, "phase3": p3, "overall": p1 * p2 * p3}

    def predict(self, smiles: str) -> Optional[OraclePrediction]:
        if not validate_smiles(smiles):
            return None

        admet_preds = predict_smiles(
            self.admet_model, smiles, self.endpoint_task_types, device=self.device
        )
        if not admet_preds:
            return None

        probs = self._predict_oracle(admet_preds)
        alerts, alert_atoms = detect_structural_alerts(smiles)
        recs = generate_recommendations(admet_preds, alerts)

        risk_factors = []
        for name, val in admet_preds.items():
            if name in {"herg", "ames", "dili"} and val > 0.5:
                risk_factors.append(
                    RiskFactor(
                        name=name.upper(),
                        category="toxicity",
                        description="Predicted risk above threshold",
                        impact=float(val),
                        source="ADMET",
                    )
                )

        return OraclePrediction(
            phase1_prob=probs["phase1"],
            phase2_prob=probs["phase2"],
            phase3_prob=probs["phase3"],
            overall_prob=probs["overall"],
            admet_predictions=admet_preds,
            risk_factors=risk_factors,
            structural_alerts=alerts,
            recommendations=recs,
            alert_atoms=alert_atoms,
        )
