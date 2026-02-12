"""Integrated SafeMolGen-DrugOracle Pipeline."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import json
import time

import torch
from loguru import logger
from rdkit import Chem

from models.generator.safemolgen import SafeMolGen
from models.oracle.drug_oracle import DrugOracle, OraclePrediction
from utils.condition_vector import (
    build_condition_vector_toward_target,
)
from models.oracle.structural_alerts import STRUCTURAL_ALERTS_DB
from utils.chemistry import MoleculeProcessor, validate_smiles, calculate_properties

# #region agent log
_DEBUG_LOG_PATH = "/Users/sreevardhandesu/Documents/Projects/MiniProject/.cursor/debug.log"
def _debug_log(message: str, data: dict, hypothesis_id: str = ""):
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({"timestamp": int(time.time() * 1000), "location": "pipeline.py", "message": message, "data": data, "hypothesisId": hypothesis_id}) + "\n")
    except Exception:
        pass
# #endregion


def _passed_safety(
    prediction: OraclePrediction,
    safety_threshold: float,
    require_no_structural_alerts: bool,
) -> bool:
    """True if iteration best passes safety criteria."""
    if prediction.overall_prob < safety_threshold:
        return False
    if require_no_structural_alerts and prediction.structural_alerts:
        return False
    return True


def _encode_oracle_feedback(
    prediction: OraclePrediction, target_success: float = 0.25
) -> Dict:
    """Encode Oracle output into target profile, avoid set, and condition vector.
    Condition vector steers toward target phase probs so the generator improves, not toward the current poor profile.
    """
    target_profile = {}
    admet = prediction.admet_predictions or {}
    if admet.get("herg", 0) > 0.5:
        target_profile["herg_max"] = 0.5
    if admet.get("bioavailability_ma", 1) < 0.5:
        target_profile["bioavailability_ma_min"] = 0.5
    avoid_smarts = []
    for alert_name in prediction.structural_alerts or []:
        for alert in STRUCTURAL_ALERTS_DB.values():
            if alert.name == alert_name:
                avoid_smarts.append(alert.smarts)
                break
    admet = prediction.admet_predictions or {}
    condition_vector = build_condition_vector_toward_target(
        admet,
        prediction.phase1_prob,
        prediction.phase2_prob,
        prediction.phase3_prob,
        target_success=target_success,
    )
    return {
        "target_profile": target_profile,
        "avoid_smarts": avoid_smarts,
        "condition_vector": condition_vector,
    }


def _filter_by_avoid(smiles_list: List[str], avoid_smarts: List[str]) -> List[str]:
    """Drop candidates that contain any of the avoid SMARTS substructures."""
    if not avoid_smarts:
        return smiles_list
    patterns = []
    for sma in avoid_smarts:
        p = Chem.MolFromSmarts(sma)
        if p is not None:
            patterns.append(p)
    if not patterns:
        return smiles_list
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if any(mol.HasSubstructMatch(pat) for pat in patterns):
            continue
        out.append(smi)
    return out


def _score_for_oracle_target(
    pred: OraclePrediction, target_profile: Dict
) -> float:
    """Higher is better: bonus for meeting target profile."""
    score = 0.0
    admet = pred.admet_predictions or {}
    if "herg_max" in target_profile:
        v = admet.get("herg", 0)
        if v <= target_profile["herg_max"]:
            score += 0.1
    if "bioavailability_ma_min" in target_profile:
        v = admet.get("bioavailability_ma", 0)
        if v >= target_profile["bioavailability_ma_min"]:
            score += 0.1
    return score


def _filter_by_property_targets(
    smiles_list: List[str], property_targets: Dict
) -> List[str]:
    """Keep candidates that satisfy structural targets: logp, mw, hbd, hba, tpsa, qed."""
    if not property_targets:
        return smiles_list
    logp_range = property_targets.get("logp")
    mw_max = property_targets.get("mw")
    qed_min = property_targets.get("qed")
    hbd_max = property_targets.get("hbd")
    hba_max = property_targets.get("hba")
    tpsa_max = property_targets.get("tpsa")
    out = []
    for smi in smiles_list:
        props = calculate_properties(smi)
        if props is None:
            continue
        if logp_range is not None:
            lo, hi = logp_range if isinstance(logp_range, (list, tuple)) else (logp_range, logp_range)
            if not (lo <= props.get("logp", 0) <= hi):
                continue
        if mw_max is not None and props.get("mw", 0) > mw_max:
            continue
        if qed_min is not None and props.get("qed", 0) < qed_min:
            continue
        if hbd_max is not None and props.get("hbd", 0) > hbd_max:
            continue
        if hba_max is not None and props.get("hba", 0) > hba_max:
            continue
        if tpsa_max is not None and props.get("tpsa", 0) > tpsa_max:
            continue
        out.append(smi)
    return out if out else smiles_list


def _filter_evaluated_by_admet_targets(
    evaluated: List[tuple], property_targets: Dict
) -> List[tuple]:
    """Filter (smiles, OraclePrediction) pairs by user ADMET targets (solubility, ppbr, clearance)."""
    if not property_targets or not evaluated:
        return evaluated
    solubility_range = property_targets.get("solubility")
    ppbr_range = property_targets.get("ppbr")
    clearance_max = property_targets.get("clearance_hepatocyte_max")
    if solubility_range is None and ppbr_range is None and clearance_max is None:
        return evaluated
    out = []
    for smi, pred in evaluated:
        admet = pred.admet_predictions or {}
        if solubility_range is not None:
            val = admet.get("solubility_aqsoldb")
            if val is None:
                continue
            lo, hi = solubility_range if isinstance(solubility_range, (list, tuple)) else (solubility_range, solubility_range)
            if not (lo <= val <= hi):
                continue
        if ppbr_range is not None:
            val = admet.get("ppbr_az")
            if val is None:
                continue
            lo, hi = ppbr_range if isinstance(ppbr_range, (list, tuple)) else (ppbr_range, ppbr_range)
            if not (lo <= val <= hi):
                continue
        if clearance_max is not None:
            val = admet.get("clearance_hepatocyte_az")
            if val is not None and val > clearance_max:
                continue
        out.append((smi, pred))
    return out if out else evaluated


def _filter_by_seed_scaffold(smiles_list: List[str], seed_smiles: Optional[str]) -> List[str]:
    """Keep candidates that contain the seed as a substructure (scaffold)."""
    if not seed_smiles or not validate_smiles(seed_smiles):
        return smiles_list
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        return smiles_list
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.HasSubstructMatch(seed_mol):
            out.append(smi)
    return out if out else smiles_list


@dataclass
class IterationResult:
    """Result from a single iteration."""

    iteration: int
    smiles: str
    prediction: OraclePrediction
    improvements: List[str] = field(default_factory=list)
    passed_safety: bool = True
    used_oracle_feedback: bool = False

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "smiles": self.smiles,
            "phase1_prob": self.prediction.phase1_prob,
            "phase2_prob": self.prediction.phase2_prob,
            "phase3_prob": self.prediction.phase3_prob,
            "overall_prob": self.prediction.overall_prob,
            "improvements": self.improvements,
            "structural_alerts": self.prediction.structural_alerts,
            "recommendations": self.prediction.recommendations,
            "passed_safety": self.passed_safety,
            "used_oracle_feedback": self.used_oracle_feedback,
        }


@dataclass
class DesignResult:
    """Complete result from molecule design."""

    final_smiles: str
    final_prediction: OraclePrediction
    iteration_history: List[IterationResult]
    target_achieved: bool
    total_iterations: int

    def to_dict(self) -> Dict:
        return {
            "final_smiles": self.final_smiles,
            "final_phase1": self.final_prediction.phase1_prob,
            "final_phase2": self.final_prediction.phase2_prob,
            "final_phase3": self.final_prediction.phase3_prob,
            "final_overall": self.final_prediction.overall_prob,
            "target_achieved": self.target_achieved,
            "total_iterations": self.total_iterations,
            "history": [r.to_dict() for r in self.iteration_history],
        }


class SafeMolGenDrugOracle:
    """Integrated SafeMolGen-DrugOracle Pipeline."""

    def __init__(
        self, generator: SafeMolGen, oracle: DrugOracle, device: str = "cpu"
    ) -> None:
        self.generator = generator
        self.oracle = oracle
        self.device = device
        self.processor = MoleculeProcessor()

    @classmethod
    def from_pretrained(
        cls,
        generator_path: str,
        oracle_path: str,
        admet_path: Optional[str],
        endpoint_names: List[str],
        endpoint_task_types: Dict[str, str],
        admet_input_dim: int,
        device: str = "cpu",
    ) -> "SafeMolGenDrugOracle":
        logger.info("Loading SafeMolGen-DrugOracle pipeline...")
        generator = SafeMolGen.from_pretrained(generator_path, device=device)
        oracle = DrugOracle.from_pretrained(
            oracle_path=oracle_path,
            admet_path=admet_path,
            endpoint_names=endpoint_names,
            endpoint_task_types=endpoint_task_types,
            input_dim=admet_input_dim,
            device=device,
        )
        return cls(generator, oracle, device=device)

    def evaluate_molecule(self, smiles: str) -> Optional[OraclePrediction]:
        if not validate_smiles(smiles):
            return None
        return self.oracle.predict(smiles)

    def generate_candidates(
        self,
        n: int = 100,
        temperature: float = 0.75,
        top_k: int = 40,
        condition: Optional[torch.Tensor] = None,
    ) -> List[str]:
        return self.generator.generate(
            n=n,
            temperature=temperature,
            top_k=top_k,
            device=self.device,
            condition=condition,
        )

    def design_molecule(
        self,
        target_success: float = 0.25,
        max_iterations: int = 10,
        candidates_per_iteration: int = 100,
        show_progress: bool = True,
        top_k: int = 40,
        safety_threshold: float = 0.2,
        require_no_structural_alerts: bool = False,
        use_oracle_feedback: bool = True,
        property_targets: Optional[Dict] = None,
        seed_smiles: Optional[str] = None,
        on_iteration_done: Optional[Callable[["DesignResult"], None]] = None,
    ) -> DesignResult:
        iteration_history = []
        best_smiles = None
        best_prediction = None
        best_score = 0.0
        oracle_feedback_for_next: Optional[Dict] = None  # set when previous iter failed safety

        temp_schedule = [
            1.1,
            1.0,
            0.95,
            0.9,
            0.85,
            0.8,
            0.75,
            0.75,
            0.7,
            0.65,
        ]

        for iteration in range(max_iterations):
            temperature = temp_schedule[min(iteration, len(temp_schedule) - 1)]
            if show_progress:
                logger.info(
                    f"\n--- Iteration {iteration + 1}/{max_iterations} (T={temperature:.2f}) ---"
                )
            condition = None
            if oracle_feedback_for_next:
                condition = oracle_feedback_for_next.get("condition_vector")
            # #region agent log
            cond_phase = None
            if condition is not None and hasattr(condition, "shape") and condition.numel() >= 3:
                cond_phase = condition.flatten()[-3:].tolist()
            _debug_log("iter_start", {"iteration": iteration + 1, "has_condition": condition is not None, "condition_phase_dims": cond_phase}, "H1,H5")
            # #endregion
            evaluated = []
            for attempt in range(2):
                n_cand = candidates_per_iteration * (2 ** attempt)
                candidates = self.generate_candidates(
                    n=n_cand,
                    temperature=temperature,
                    top_k=top_k,
                    condition=condition,
                )
                if property_targets:
                    candidates = _filter_by_property_targets(candidates, property_targets)
                if seed_smiles:
                    candidates = _filter_by_seed_scaffold(candidates, seed_smiles)
                if oracle_feedback_for_next:
                    avoid_smarts = oracle_feedback_for_next.get("avoid_smarts", [])
                    candidates_after_avoid = _filter_by_avoid(candidates, avoid_smarts)
                    min_after_avoid = max(20, int(0.2 * n_cand))
                    used_fallback = len(candidates_after_avoid) < min_after_avoid
                    if len(candidates_after_avoid) >= min_after_avoid:
                        candidates = candidates_after_avoid
                        if show_progress and len(candidates) < n_cand:
                            logger.info(
                                f"Oracle filter: {len(candidates)}/{n_cand} candidates after avoid-substructure filter"
                            )
                    else:
                        if show_progress and avoid_smarts:
                            logger.info(
                                f"Oracle filter would leave {len(candidates_after_avoid)} candidates (min {min_after_avoid}); keeping full set to preserve diversity"
                            )
                    # #region agent log
                    _debug_log("after_avoid", {"iteration": iteration + 1, "n_candidates": len(candidates), "n_after_avoid": len(candidates_after_avoid), "min_after_avoid": min_after_avoid, "used_fallback": used_fallback}, "H2")
                    # #endregion

                evaluated = []
                for smi in candidates:
                    pred = self.evaluate_molecule(smi)
                    if pred is not None:
                        evaluated.append((smi, pred))
                if evaluated:
                    break
                if attempt == 0 and show_progress:
                    logger.info(
                        f"Iteration {iteration + 1}: no evaluated candidates, retrying with 2x candidates ({2 * candidates_per_iteration})"
                    )
            if not evaluated:
                continue
            if property_targets:
                evaluated = _filter_evaluated_by_admet_targets(evaluated, property_targets)
            if not evaluated:
                continue

            if oracle_feedback_for_next:
                target_profile = oracle_feedback_for_next.get("target_profile", {})
                evaluated.sort(
                    key=lambda x: x[1].overall_prob
                    + _score_for_oracle_target(x[1], target_profile),
                    reverse=True,
                )
            else:
                evaluated.sort(key=lambda x: x[1].overall_prob, reverse=True)
            iter_best_smi, iter_best_pred = evaluated[0]
            # #region agent log
            best_by_overall_only = max(evaluated, key=lambda x: x[1].overall_prob) if evaluated else None
            _debug_log("iter_best", {
                "iteration": iteration + 1,
                "n_evaluated": len(evaluated),
                "sort_used_target_profile": oracle_feedback_for_next is not None,
                "iter_best_overall": iter_best_pred.overall_prob,
                "iter_best_phase1": iter_best_pred.phase1_prob,
                "iter_best_phase2": iter_best_pred.phase2_prob,
                "iter_best_phase3": iter_best_pred.phase3_prob,
                "best_by_overall_only_overall": best_by_overall_only[1].overall_prob if best_by_overall_only else None,
                "same_as_best_by_overall": best_by_overall_only is not None and best_by_overall_only[1].overall_prob == iter_best_pred.overall_prob,
            }, "H3,H4")
            # #endregion

            passed_safety = _passed_safety(
                iter_best_pred, safety_threshold, require_no_structural_alerts
            )
            used_oracle_feedback = oracle_feedback_for_next is not None
            # Use Oracle feedback for next iteration when below target (e.g. 25%), not only below safety_threshold,
            # so the loop keeps steering until we hit the goal.
            if iter_best_pred.overall_prob < target_success and use_oracle_feedback:
                oracle_feedback_for_next = _encode_oracle_feedback(
                    iter_best_pred, target_success
                )
                # #region agent log
                enc = oracle_feedback_for_next.get("condition_vector")
                enc_phase = enc.flatten()[-3:].tolist() if enc is not None and hasattr(enc, "flatten") else None
                _debug_log("encode_feedback", {"iteration": iteration + 1, "source_overall": iter_best_pred.overall_prob, "source_phase1": iter_best_pred.phase1_prob, "source_phase2": iter_best_pred.phase2_prob, "source_phase3": iter_best_pred.phase3_prob, "encoded_phase_dims": enc_phase}, "H1")
                # #endregion
            else:
                oracle_feedback_for_next = None

            improvements = []
            if best_prediction is not None:
                if iter_best_pred.phase1_prob > best_prediction.phase1_prob:
                    diff = iter_best_pred.phase1_prob - best_prediction.phase1_prob
                    improvements.append(f"Phase I: +{diff:.1%}")
                if iter_best_pred.phase2_prob > best_prediction.phase2_prob:
                    diff = iter_best_pred.phase2_prob - best_prediction.phase2_prob
                    improvements.append(f"Phase II: +{diff:.1%}")
                if iter_best_pred.overall_prob > best_prediction.overall_prob:
                    diff = iter_best_pred.overall_prob - best_prediction.overall_prob
                    improvements.append(f"Overall: +{diff:.1%}")

            iteration_history.append(
                IterationResult(
                    iteration=iteration + 1,
                    smiles=iter_best_smi,
                    prediction=iter_best_pred,
                    improvements=improvements,
                    passed_safety=passed_safety,
                    used_oracle_feedback=used_oracle_feedback,
                )
            )

            if iter_best_pred.overall_prob > best_score:
                best_smiles = iter_best_smi
                best_prediction = iter_best_pred
                best_score = iter_best_pred.overall_prob
                if show_progress:
                    logger.info(f"✓ New best: {best_score:.2%}")

            if on_iteration_done is not None:
                _cur_best_smi = best_smiles if best_smiles is not None else iter_best_smi
                _cur_best_pred = best_prediction if best_prediction is not None else iter_best_pred
                _partial = DesignResult(
                    final_smiles=_cur_best_smi or "",
                    final_prediction=_cur_best_pred,
                    iteration_history=iteration_history.copy(),
                    target_achieved=False,
                    total_iterations=iteration + 1,
                )
                on_iteration_done(_partial)

            if best_score >= target_success:
                if show_progress:
                    logger.info(
                        f"\n🎉 Target achieved at iteration {iteration + 1}!"
                    )
                return DesignResult(
                    final_smiles=best_smiles,
                    final_prediction=best_prediction,
                    iteration_history=iteration_history,
                    target_achieved=True,
                    total_iterations=iteration + 1,
                )

        if best_smiles is None and iteration_history:
            last = iteration_history[-1]
            best_smiles = last.smiles
            best_prediction = last.prediction
        if best_prediction is None:
            best_prediction = OraclePrediction(
                phase1_prob=0.0,
                phase2_prob=0.0,
                phase3_prob=0.0,
                overall_prob=0.0,
                admet_predictions={},
                risk_factors=[],
                structural_alerts=[],
                recommendations=[],
            )
        return DesignResult(
            final_smiles=best_smiles or "",
            final_prediction=best_prediction,
            iteration_history=iteration_history,
            target_achieved=False,
            total_iterations=max_iterations,
        )

    def compare_molecules(self, smiles_list: List[str]) -> List[Dict]:
        results = []
        for smi in smiles_list:
            pred = self.evaluate_molecule(smi)
            if pred is None:
                continue
            props = calculate_properties(smi)
            results.append(
                {"smiles": smi, "prediction": pred.to_dict(), "properties": props}
            )
        results.sort(
            key=lambda x: x["prediction"].get("overall_prob", 0.0), reverse=True
        )
        return results

    def save_result(self, result: DesignResult, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Result saved to {path}")
