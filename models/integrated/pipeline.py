"""Integrated SafeMolGen-DrugOracle Pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

from loguru import logger

from models.generator.safemolgen import SafeMolGen
from models.oracle.drug_oracle import DrugOracle, OraclePrediction
from utils.chemistry import MoleculeProcessor, validate_smiles, calculate_properties


@dataclass
class IterationResult:
    """Result from a single iteration."""

    iteration: int
    smiles: str
    prediction: OraclePrediction
    improvements: List[str] = field(default_factory=list)

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
        self, n: int = 100, temperature: float = 1.0
    ) -> List[str]:
        return self.generator.generate(
            n=n, temperature=temperature, device=self.device
        )

    def design_molecule(
        self,
        target_success: float = 0.25,
        max_iterations: int = 10,
        candidates_per_iteration: int = 100,
        show_progress: bool = True,
    ) -> DesignResult:
        iteration_history = []
        best_smiles = None
        best_prediction = None
        best_score = 0.0

        temp_schedule = [
            1.2,
            1.1,
            1.0,
            0.95,
            0.9,
            0.85,
            0.8,
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
            candidates = self.generate_candidates(
                n=candidates_per_iteration, temperature=temperature
            )

            evaluated = []
            for smi in candidates:
                pred = self.evaluate_molecule(smi)
                if pred is not None:
                    evaluated.append((smi, pred))
            if not evaluated:
                continue

            evaluated.sort(key=lambda x: x[1].overall_prob, reverse=True)
            iter_best_smi, iter_best_pred = evaluated[0]

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
                )
            )

            if iter_best_pred.overall_prob > best_score:
                best_smiles = iter_best_smi
                best_prediction = iter_best_pred
                best_score = iter_best_pred.overall_prob
                if show_progress:
                    logger.info(f"✓ New best: {best_score:.2%}")

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

        return DesignResult(
            final_smiles=best_smiles,
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
