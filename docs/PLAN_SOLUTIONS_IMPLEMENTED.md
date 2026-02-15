# Fix "Base Beats RL" – All Plan Solutions Implemented

This document maps each item from the fix plan to its implementation.

---

## 1. Improve RL algorithm and training

| Item | Status | Where |
|------|--------|--------|
| **1.1 PPO** | Done | `models/generator/rl_trainer.py`: `use_ppo`, `ppo_eps`, `ppo_epochs`. CLI: `--use-ppo`, `--ppo-eps`, `--ppo-epochs` |
| **1.2 Gradient accumulation** | Done | `rl_trainer.py`: `accumulation_steps`. CLI: `--accumulation-steps` |
| **1.3 Learned value baseline** | Done | `rl_trainer.py`: `use_value_baseline`, value net + MSE. CLI: `--use-value-baseline` |
| **1.4 RL from Option B** | Done | `scripts/run_rl_from_option_b.sh`: RL with `--resume checkpoints/generator/best` |

---

## 2. Reward design

| Item | Status | Where |
|------|--------|--------|
| **2.1 Phase-wise reward** | Done | `rewards.py`: `_oracle_scalar` with `phase_weights`. `train_generator.py`: `_make_oracle_phase_fn`, `--phase-weights` |
| **2.2 Validity-gated oracle** | Done | `rewards.py`: `validity_gated_oracle` in all compute_* functions. `RLConfig.validity_gated_oracle` |
| **2.3 Batch-normalized oracle** | Done | `rewards.py`: `oracle_scores_override`. `rl_trainer.py`: `batch_normalize_oracle`, normalize then clip. CLI: `--batch-normalize-oracle` |
| **2.4 Structural alerts / risk penalty** | Done | `rewards.py`: `_scalar_from_prediction`, `_alert_penalty`, `oracle_prediction_fn`, `w_alert`. `rl_trainer.py`: `w_alert`, `oracle_prediction_fn`. CLI: `--w-alert` |

---

## 3. Supervised alternatives

| Item | Status | Where |
|------|--------|--------|
| **3.1 Best-of-N fine-tune** | Done | `models/generator/best_of_n_trainer.py`, `scripts/train_best_of_n.py` |
| **3.2 Imitation from pipeline** | Done | `pipeline.py`: `imitation_callback`. `scripts/collect_imitation_data.py`, `scripts/train_imitation.py`, `cond_dataset.ImitationDataset` |
| **3.3 Target-condition curriculum** | Done | `scripts/run_option_b_curriculum.py`: pretrain 0.5 → 0.55 → 0.6. `train_generator.py`: `--resume` for pretrain (continue with new target phase) |

---

## 4. Oracle and data

| Item | Status | Where |
|------|--------|--------|
| **4.1 Improve Oracle discriminativity** | External | Retrain Oracle / recalibrate; see `scripts/train_oracle.py`, `models/oracle/` |
| **4.2 More curated data for Option B** | Done | `build_oracle_curated_smiles.py`: `--top-pct`, `--min-overall`. `run_option_b_full.py`: `--min-overall`. `run_option_b_curriculum.py`: `--min-overall` |
| **4.3 Two-stage reranker** | Done | `models/reranker/`, `scripts/build_reranker_dataset.py`, `scripts/train_reranker.py`. Pipeline: `reranker_path`, `use_reranker`, `reranker_top_k`, `_rerank_candidates` |

---

## 5. Search and pipeline only

| Item | Status | Where |
|------|--------|--------|
| **5.1 Massive restarts / evolutionary** | Done | `pipeline.py`: `design_molecule_with_restarts`, `design_molecule_evolutionary` |
| **5.2 Pareto / diversity selection** | Done | `pipeline.py`: `_pareto_front`, `_select_diverse`. `design_molecule(..., selection_mode= "overall"\|"pareto"\|"diversity", diversity_tanimoto_max)`. `utils/chemistry.tanimoto_similarity` |

---

## Quick usage

- **RL from Option B:** `PYTHONPATH=. bash scripts/run_rl_from_option_b.sh`
- **RL curriculum (when RL ≈ base):** `PYTHONPATH=. bash scripts/run_rl_curriculum.sh` (starts from Option B; Phase 2 uses w_oracle 0.25 to avoid validity collapse)
- **RL with alert penalty:** `--w-alert 0.1`
- **RL with PPO + value baseline:** `--use-ppo --use-value-baseline`
- **Best-of-N (stable alternative):** `PYTHONPATH=. python scripts/train_best_of_n.py --resume checkpoints/generator/best --out checkpoints/generator_best_of_n` or `bash scripts/run_best_of_n_then_eval.sh`
- **Curriculum Option B:** `PYTHONPATH=. python scripts/run_option_b_curriculum.py --phases 0.5,0.55,0.6`
- **Imitation:** collect then train: `scripts/collect_imitation_data.py` → `scripts/train_imitation.py`
- **Reranker:** `scripts/build_reranker_dataset.py` → `scripts/train_reranker.py`; then `from_pretrained(..., reranker_path=...)`, `design_molecule(..., use_reranker=True)`
- **Restarts / evolutionary:** `pipeline.design_molecule_with_restarts(n_restarts=20)`, `pipeline.design_molecule_evolutionary(...)`. CLI: `--restarts 5`, `--evolutionary`. API: `n_restarts`, `design_mode: "restarts"|"evolutionary"`.
- **Pareto / diversity:** `design_molecule(..., selection_mode="pareto")` or `selection_mode="diversity", diversity_tanimoto_max=0.7`. CLI: `--selection-mode diversity --diversity-tanimoto-max 0.7`. API: `selection_mode`, `diversity_tanimoto_max`.
- **All solutions in one run:** `PYTHONPATH=. bash scripts/run_pipeline_all_solutions.sh` or `python3 scripts/run_pipeline.py --all-solutions` (enables restarts=5, selection_mode=diversity, use_reranker when available). Backend loads reranker automatically when `checkpoints/reranker/reranker.pt` exists; API accepts `selection_mode`, `n_restarts`, `design_mode`, `use_reranker`, `reranker_top_k`, `population_size`, `generations`.
