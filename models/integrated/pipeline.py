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
    build_condition_vector_toward_target_phase_aware,
)
from models.oracle.structural_alerts import STRUCTURAL_ALERTS_DB
from utils.chemistry import MoleculeProcessor, validate_smiles, calculate_properties, generate_mutations, tanimoto_similarity

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
    prediction: OraclePrediction,
    target_success: float = 0.5,
    device: str = "cpu",
    phase_boost: float = 0.0,
    use_phase_aware_steering: bool = True,
) -> Dict:
    """Encode Oracle output into target profile, avoid set, and condition vector.
    Condition vector steers toward target phase probs so the generator improves.
    When use_phase_aware_steering is True, the worst phase gets a stronger blend.
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
    if use_phase_aware_steering:
        condition_vector = build_condition_vector_toward_target_phase_aware(
            admet,
            prediction.phase1_prob,
            prediction.phase2_prob,
            prediction.phase3_prob,
            target_success=target_success,
            device=device,
            phase_boost=phase_boost,
        )
    else:
        condition_vector = build_condition_vector_toward_target(
            admet,
            prediction.phase1_prob,
            prediction.phase2_prob,
            prediction.phase3_prob,
            target_success=target_success,
            device=device,
            phase_boost=phase_boost,
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


def _pareto_front(evaluated: List[tuple]) -> List[tuple]:
    """Return (smi, pred) pairs that are non-dominated on (phase1, phase2, phase3) (plan 5.2)."""
    if not evaluated:
        return []
    dominated = [False] * len(evaluated)
    for i, (_, pi) in enumerate(evaluated):
        for j, (_, pj) in enumerate(evaluated):
            if i == j:
                continue
            if (
                pj.phase1_prob >= pi.phase1_prob
                and pj.phase2_prob >= pi.phase2_prob
                and pj.phase3_prob >= pi.phase3_prob
                and (
                    pj.phase1_prob > pi.phase1_prob
                    or pj.phase2_prob > pi.phase2_prob
                    or pj.phase3_prob > pi.phase3_prob
                )
            ):
                dominated[i] = True
                break
    return [evaluated[i] for i in range(len(evaluated)) if not dominated[i]]


def _select_diverse(
    evaluated: List[tuple],
    ref_smiles: Optional[str],
    tanimoto_max: float = 0.7,
) -> tuple:
    """Pick best overall among molecules structurally different from ref (plan 5.2)."""
    if not evaluated:
        return evaluated[0] if evaluated else (None, None)
    sorted_eval = sorted(evaluated, key=lambda x: x[1].overall_prob, reverse=True)
    if not ref_smiles:
        return sorted_eval[0]
    for smi, pred in sorted_eval:
        if tanimoto_similarity(smi, ref_smiles) <= tanimoto_max:
            return (smi, pred)
    return sorted_eval[0]


def _score_phase_weighted(pred, w1: float = 0.2, w2: float = 0.5, w3: float = 0.3) -> float:
    """Score for phase-weighted selection: higher weight on Phase II so we favor improving it."""
    return w1 * pred.phase1_prob + w2 * pred.phase2_prob + w3 * pred.phase3_prob


def _select_phase_weighted(evaluated: List[tuple], w1: float = 0.2, w2: float = 0.5, w3: float = 0.3) -> tuple:
    """Pick candidate that maximizes w1*p1 + w2*p2 + w3*p3 (favors Phase II improvement)."""
    if not evaluated:
        return (None, None)
    return max(evaluated, key=lambda x: _score_phase_weighted(x[1], w1, w2, w3))


def _select_bottleneck(evaluated: List[tuple]) -> tuple:
    """Pick candidate that maximizes min(p1, p2, p3) to improve the worst phase."""
    if not evaluated:
        return (None, None)
    return max(evaluated, key=lambda x: min(x[1].phase1_prob, x[1].phase2_prob, x[1].phase3_prob))


def _filter_by_property_targets(
    smiles_list: List[str], property_targets: Dict
) -> List[str]:
    """Keep candidates that satisfy structural targets: logp, mw, mw_min, hbd, hba, tpsa, qed.
    When no candidate passes, returns [] (no fallback to unfiltered list) so trivial molecules
    are not considered.
    """
    if not property_targets:
        return smiles_list
    logp_range = property_targets.get("logp")
    mw_max = property_targets.get("mw")
    mw_min = property_targets.get("mw_min")
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
        if mw_min is not None and props.get("mw", 0) < mw_min:
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
    # Do not fall back to unfiltered list when no one passes; avoids trivial molecules winning
    return out


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


# Minimum heavy atoms for a molecule to count as "drug-like" (exclude trivial e.g. C=C, benzene-only).
MIN_HEAVY_ATOMS_DRUGLIKE = 10


def _is_druglike_complexity(smiles: str, min_heavy_atoms: int = MIN_HEAVY_ATOMS_DRUGLIKE) -> bool:
    """True if the molecule has at least min_heavy_atoms (excludes trivial molecules like C=C)."""
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() >= min_heavy_atoms


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
            "recommendations": getattr(self.final_prediction, "recommendations", []) or [],
        }


class SafeMolGenDrugOracle:
    """Integrated SafeMolGen-DrugOracle Pipeline."""

    def __init__(
        self,
        generator: SafeMolGen,
        oracle: DrugOracle,
        device: str = "cpu",
        reranker=None,
        generator_early: Optional[SafeMolGen] = None,
    ) -> None:
        self.generator = generator
        self.oracle = oracle
        self.device = device
        self.processor = MoleculeProcessor()
        self.reranker = reranker
        self.generator_early = generator_early

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
        reranker_path: Optional[str] = None,
        generator_early: Optional[SafeMolGen] = None,
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
        reranker = None
        if reranker_path:
            from models.reranker.model import load_reranker
            reranker = load_reranker(reranker_path, generator.tokenizer, device=device)
        return cls(generator, oracle, device=device, reranker=reranker, generator_early=generator_early)

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
        generator_override: Optional[SafeMolGen] = None,
    ) -> List[str]:
        gen = generator_override if generator_override is not None else self.generator
        return gen.generate(
            n=n,
            temperature=temperature,
            top_k=top_k,
            device=self.device,
            condition=condition,
        )

    def _rerank_candidates(
        self,
        condition: torch.Tensor,
        candidates: List[str],
        top_k: int,
    ) -> List[str]:
        """Score candidates with reranker and return top_k by predicted oracle score (plan 4.3)."""
        if not self.reranker or not candidates:
            return candidates
        tokenizer = self.generator.tokenizer
        cond_batch = condition.expand(len(candidates), -1)
        ids = [tokenizer.encode(s) for s in candidates]
        max_len = tokenizer.max_length
        pad_id = tokenizer.vocab.get(tokenizer.PAD_TOKEN, 0)
        padded = []
        for seq in ids:
            if len(seq) < max_len:
                seq = seq + [pad_id] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            padded.append(seq)
        ids_t = torch.tensor(padded, dtype=torch.long, device=self.device)
        cond_batch = cond_batch.to(self.device)
        with torch.no_grad():
            scores = self.reranker(cond_batch, ids_t)
        sorted_idx = scores.cpu().argsort(descending=True)
        return [candidates[i] for i in sorted_idx[:top_k]]

    def design_molecule(
        self,
        target_success: float = 0.5,
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
        imitation_callback: Optional[Callable[["DesignResult", Optional[torch.Tensor], str], None]] = None,
        use_reranker: bool = False,
        reranker_top_k: Optional[int] = None,
        selection_mode: str = "overall",
        diversity_tanimoto_max: float = 0.7,
        use_phase_aware_steering: bool = True,
        exploration_fraction: float = 0.25,
        first_iteration_temperature: Optional[float] = None,
    ) -> DesignResult:
        iteration_history = []
        best_smiles = None
        best_prediction = None
        best_score = 0.0
        oracle_feedback_for_next: Optional[Dict] = None  # set when previous iter failed safety

        # Lower temperatures so the condition signal has more effect (less random sampling).
        temp_schedule = [
            0.9,
            0.85,
            0.82,
            0.8,
            0.78,
            0.76,
            0.74,
            0.72,
            0.7,
            0.65,
        ]

        for iteration in range(max_iterations):
            # Iteration 0: optional weak start (early generator + higher temp)
            if iteration == 0:
                temperature = (
                    first_iteration_temperature
                    if first_iteration_temperature is not None
                    else 1.4
                )
                generator_override = self.generator_early
            else:
                temperature = temp_schedule[min(iteration, len(temp_schedule) - 1)]
                generator_override = None
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

            # Mutation-based local search: evaluate neighbors of current best so we can improve even when generator is unconditioned (cond_dim=0)
            evaluated = []
            if best_smiles and iteration > 0:
                mutations = generate_mutations(best_smiles, n=100, random_seed=iteration + 42)
                for smi in mutations:
                    pred = self.evaluate_molecule(smi)
                    if pred is not None:
                        evaluated.append((smi, pred))
                if show_progress and mutations:
                    logger.info(
                        f"Iteration {iteration + 1}: {len(evaluated)} mutation candidates evaluated"
                    )

            for attempt in range(2):
                n_cand = candidates_per_iteration * (2 ** attempt)
                # Multi-temperature sampling: include higher temp to escape local plateau
                temps = [temperature]
                if attempt == 0:
                    temps.append(min(0.95, temperature * 1.15))
                    temps.append(1.0)
                candidates = []
                for t in temps:
                    n_here = n_cand if t == temperature else (max(50, n_cand // 2) if t < 1.0 else max(40, n_cand // 3))
                    candidates.extend(
                        self.generate_candidates(
                            n=n_here,
                            temperature=t,
                            top_k=top_k,
                            condition=condition,
                            generator_override=generator_override,
                        )
                    )
                # Exploration: add unconditioned candidates to escape local optima (iteration >= 1)
                if attempt == 0 and iteration >= 1 and exploration_fraction > 0:
                    n_explore = max(40, int(exploration_fraction * n_cand))
                    candidates.extend(
                        self.generate_candidates(
                            n=n_explore,
                            temperature=1.2,
                            top_k=min(top_k + 15, 80),
                            condition=None,
                        )
                    )
                candidates = list(dict.fromkeys(candidates))  # dedupe

                if use_reranker and self.reranker and condition is not None:
                    k = reranker_top_k if reranker_top_k is not None else 200
                    candidates = self._rerank_candidates(condition, candidates, min(k, len(candidates)))
                    if show_progress and k < len(candidates):
                        logger.info(f"Reranker: kept top {k} of candidates for Oracle evaluation")

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

                for smi in candidates:
                    pred = self.evaluate_molecule(smi)
                    if pred is not None:
                        evaluated.append((smi, pred))
                if len(evaluated) >= 1:
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

            # Force drug-like only: never select or display trivial molecules (e.g. CC, C=C)
            evaluated_druglike = [(s, p) for s, p in evaluated if _is_druglike_complexity(s)]
            if not evaluated_druglike:
                # No drug-like candidate this iteration: carry over previous best or skip
                if best_smiles is not None:
                    iter_best_smi, iter_best_pred = best_smiles, best_prediction
                else:
                    # First iteration with no drug-like candidates; skip and try next iteration
                    if show_progress:
                        logger.info(
                            f"Iteration {iteration + 1}: no drug-like candidates (min {MIN_HEAVY_ATOMS_DRUGLIKE} heavy atoms), skipping"
                        )
                    continue
            else:
                selection_pool = evaluated_druglike
                # Pick iteration best from drug-like only
                if selection_mode == "pareto":
                    front = _pareto_front(selection_pool)
                    iter_best_smi, iter_best_pred = (
                        max(front, key=lambda x: x[1].overall_prob) if front else selection_pool[0]
                    )
                elif selection_mode == "diversity":
                    iter_best_smi, iter_best_pred = _select_diverse(
                        selection_pool, best_smiles, diversity_tanimoto_max
                    )
                elif selection_mode == "phase_weighted":
                    iter_best_smi, iter_best_pred = _select_phase_weighted(selection_pool)
                elif selection_mode == "bottleneck":
                    iter_best_smi, iter_best_pred = _select_bottleneck(selection_pool)
                else:
                    iter_best_smi, iter_best_pred = max(selection_pool, key=lambda x: x[1].overall_prob)
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

            # Improvements vs previous global best (before we update)
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

            # Update global best only when drug-like (min heavy atoms), so we never pick C=C or trivial molecules
            if (
                _is_druglike_complexity(iter_best_smi)
                and iter_best_pred.overall_prob > best_score
            ):
                best_smiles = iter_best_smi
                best_prediction = iter_best_pred
                best_score = iter_best_pred.overall_prob
                if show_progress:
                    smi_short = (iter_best_smi[:50] + "…") if len(iter_best_smi) > 50 else iter_best_smi
                    logger.info(f"✓ New best: {best_score:.2%} | {smi_short}")

            # Use Oracle feedback for next iteration when below target. Encode from global best
            # (best_prediction) so we steer toward improving from the best molecule seen so far.
            source_for_feedback = best_prediction if best_prediction is not None else iter_best_pred
            if source_for_feedback.overall_prob < target_success and use_oracle_feedback:
                # Raise target phase over iterations; use stronger push when user asks for high target (e.g. 0.7)
                if target_success >= 0.6:
                    phase_boost = min(0.22, (iteration + 1) * 0.028)
                else:
                    phase_boost = min(0.15, (iteration + 1) * 0.02)
                oracle_feedback_for_next = _encode_oracle_feedback(
                    source_for_feedback, target_success, device=self.device, phase_boost=phase_boost,
                    use_phase_aware_steering=use_phase_aware_steering,
                )
                # #region agent log
                enc = oracle_feedback_for_next.get("condition_vector")
                enc_phase = enc.flatten()[-3:].tolist() if enc is not None and hasattr(enc, "flatten") else None
                _debug_log("encode_feedback", {"iteration": iteration + 1, "source_overall": source_for_feedback.overall_prob, "source_phase1": source_for_feedback.phase1_prob, "source_phase2": source_for_feedback.phase2_prob, "source_phase3": source_for_feedback.phase3_prob, "encoded_phase_dims": enc_phase}, "H1")
                # #endregion
            else:
                oracle_feedback_for_next = None

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
            if imitation_callback is not None:
                _cur_best_smi = best_smiles if best_smiles is not None else iter_best_smi
                _partial_im = DesignResult(
                    final_smiles=_cur_best_smi or "",
                    final_prediction=best_prediction or iter_best_pred,
                    iteration_history=iteration_history.copy(),
                    target_achieved=False,
                    total_iterations=iteration + 1,
                )
                imitation_callback(_partial_im, condition, _cur_best_smi or iter_best_smi)

            if best_score >= target_success:
                if show_progress:
                    logger.info(
                        f"\n🎉 Target achieved at iteration {iteration + 1}! (continuing to max_iterations)"
                    )
            # Run all iterations (no early return) so the UI gets full history for the optimization graph.

        if best_smiles is None and iteration_history:
            # Prefer best drug-like molecule from history (never use trivial e.g. C=C)
            druglike_from_history = [
                (h.smiles, h.prediction) for h in iteration_history
                if _is_druglike_complexity(h.smiles)
            ]
            if druglike_from_history:
                best_smiles, best_prediction = max(
                    druglike_from_history, key=lambda x: x[1].overall_prob
                )
                best_score = best_prediction.overall_prob
            else:
                # No drug-like molecule in any iteration; leave best empty rather than returning C=C etc.
                best_smiles = ""
                best_prediction = iteration_history[-1].prediction
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
            target_achieved=best_score >= target_success if best_score is not None else False,
            total_iterations=max_iterations,
        )

    def design_molecule_with_restarts(
        self,
        n_restarts: int = 20,
        restart_seed_base: int = 42,
        show_progress: bool = True,
        **kwargs,
    ) -> DesignResult:
        """Run design_molecule multiple times with different seeds; return the best result (plan 5.1)."""
        import random
        import numpy as np
        best_result: Optional[DesignResult] = None
        best_score = -1.0
        for i in range(n_restarts):
            seed = restart_seed_base + i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            inner_show = show_progress and (i == 0)
            res = self.design_molecule(**kwargs, show_progress=inner_show)
            score = res.final_prediction.overall_prob
            if score > best_score:
                best_score = score
                best_result = res
            if show_progress and n_restarts > 1:
                logger.info(f"Restart {i + 1}/{n_restarts}  best_overall={best_score:.2%}")
        return best_result or self.design_molecule(**kwargs, show_progress=False)

    def design_molecule_evolutionary(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutations_per_mol: int = 30,
        generator_offspring: int = 100,
        target_success: float = 0.5,
        condition: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> DesignResult:
        """Evolutionary loop: population of best molecules, mutate + generate children, oracle score, select (plan 5.1)."""
        from utils.condition_vector import get_target_condition
        if condition is None and getattr(self.generator.model, "cond_dim", 0) > 0:
            condition = get_target_condition(device=self.device, phase=0.6)
        pool = self.generate_candidates(
            n=population_size * 5,
            temperature=0.9,
            top_k=40,
            condition=condition,
        )
        evaluated: List[tuple] = []
        for smi in pool:
            pred = self.evaluate_molecule(smi)
            if pred is not None:
                evaluated.append((smi, pred))
        if not evaluated:
            return DesignResult(
                final_smiles="",
                final_prediction=OraclePrediction(
                    phase1_prob=0.0, phase2_prob=0.0, phase3_prob=0.0, overall_prob=0.0,
                    admet_predictions={}, risk_factors=[], structural_alerts=[], recommendations=[],
                ),
                iteration_history=[],
                target_achieved=False,
                total_iterations=0,
            )
        evaluated.sort(key=lambda x: x[1].overall_prob, reverse=True)
        population = [s for s, _ in evaluated[:population_size]]
        best_smi, best_pred = evaluated[0]
        for gen in range(generations - 1):
            candidates = list(population)
            for smi in population:
                mutations = generate_mutations(smi, n=mutations_per_mol, random_seed=gen * 1000 + sum(ord(c) for c in smi[:20]) % 1000)
                candidates.extend(mutations)
            if generator_offspring > 0:
                new_smis = self.generate_candidates(
                    n=generator_offspring,
                    temperature=0.8,
                    top_k=40,
                    condition=condition,
                )
                candidates.extend(new_smis)
            candidates = list(dict.fromkeys(candidates))
            evaluated = []
            for smi in candidates:
                pred = self.evaluate_molecule(smi)
                if pred is not None:
                    evaluated.append((smi, pred))
            if not evaluated:
                break
            evaluated.sort(key=lambda x: x[1].overall_prob, reverse=True)
            population = [s for s, _ in evaluated[:population_size]]
            cur_best_smi, cur_best_pred = evaluated[0]
            if cur_best_pred.overall_prob > best_pred.overall_prob:
                best_smi, best_pred = cur_best_smi, cur_best_pred
            if show_progress:
                logger.info(f"Evolution gen {gen + 1}/{generations}  best_overall={best_pred.overall_prob:.2%}")
        return DesignResult(
            final_smiles=best_smi,
            final_prediction=best_pred,
            iteration_history=[
                IterationResult(
                    iteration=1,
                    smiles=best_smi,
                    prediction=best_pred,
                    improvements=[],
                    passed_safety=True,
                    used_oracle_feedback=False,
                )
            ],
            target_achieved=best_pred.overall_prob >= target_success,
            total_iterations=generations,
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
