SafeMolGen-DrugOracle
=====================
Integrated AI system for drug design with ADMET prediction, clinical phase
success estimation, molecule generation, and a FastAPI + React UI.

Project Phases
--------------
1. Phase 1: Project setup + ADMET predictor
2. Phase 2: DrugOracle (clinical phase predictors + explainability)
3. Phase 3: SafeMolGen (molecular generator)
4. Phase 4: Integration + FastAPI + React UI

Status
------
This repo is being built from scratch following the provided phase documents.
Phase 1 baseline metrics recorded on 2026-02-01 (see
`docs/reports/EXPERIMENT_LOG.md` for details).

Phase 1 (ADMET) - Quick Start
-----------------------------
1. Install dependencies:
   `pip install -r requirements.txt`
2. Download and split TDC ADMET data:
   `python scripts/download_data.py`
3. Preprocess into molecular graphs:
   `python scripts/preprocess_data.py`
4. Train the ADMET predictor:
   `python scripts/train_admet.py`
5. Evaluate on test sets:
   `python scripts/evaluate_admet.py`

Generator (Phase 3) - Data
---------------------------
Pretrain expects SMILES at `data/processed/generator/smiles.tsv` (e.g. from
`python scripts/download_chembl_smiles.py`). If that path is missing, training
falls back to valid SMILES aggregated from `data/admet_group/*/train_val.csv`.
Training filters to valid SMILES only and optionally canonicalizes with RDKit.

Oracle structural alerts are loaded from `data/structural_alerts.csv` when present
(otherwise a built-in set is used). To refresh from the Hamburg SMARTS dataset
(PAINS, skin sensitization), run `python scripts/download_structural_alerts.py`.

Full pipeline run order (ADMET -> Oracle -> Generator -> Generate)
------------------------------------------------------------------
For end-to-end generation with best long-term validity/uniqueness, run in this order
(from project root with `PYTHONPATH=.` and dependencies installed):

1. **ADMET data** (from existing `data/admet_group/`):
   `python scripts/prepare_admet_from_admet_group.py`
2. **ADMET graphs**: `python scripts/preprocess_data.py`
3. **ADMET train**: `python scripts/train_admet.py` -> `checkpoints/admet/best_model.pt`
4. **Clinical data**: `python scripts/prepare_clinical_data.py` -> `data/processed/oracle/clinical_trials.csv`
5. **Oracle train**: `python scripts/train_oracle.py` -> `checkpoints/oracle/best_model.pt`
6. **Generator pretrain** (aim for high validity; 30+ epochs recommended):
   `python scripts/train_generator.py --stage pretrain --epochs 30 --limit 50000 --batch-size 64`
7. **Optional RL** (after Oracle exists): `python scripts/train_generator.py --stage rl --resume checkpoints/generator --epochs 10 --w-validity 0.75 --w-diversity 0.1`
8. **Generate**: Web app via FastAPI backend + React frontend (see below), or CLI (see below).

Run from CLI with monitoring
-----------------------------
From project root with `PYTHONPATH` set (or use the script):

```bash
# One-liner: run and watch live per-iteration progress (no JSON saved)
bash scripts/run_monitor.sh

# Save result to JSON
bash scripts/run_monitor.sh --out outputs/design_result.json

# Fewer iterations / candidates for quicker runs
bash scripts/run_monitor.sh --max-iterations 5 --candidates-per-iteration 80 --out outputs/result.json
```

Or with Python directly:
```bash
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
python3 scripts/run_pipeline.py --out outputs/design_result.json   # with save
python3 scripts/run_pipeline.py                                     # monitor only, no save
```

Each iteration prints a line: `iter N  overall=X%  phase1=...  phase2=...  phase3=...  |  SMILES...`

**All solutions (restarts + diversity + reranker):**
```bash
bash scripts/run_pipeline_all_solutions.sh --out outputs/result.json
# Or: python3 scripts/run_pipeline.py --all-solutions
# Or: --restarts 5 --selection-mode diversity --use-reranker
# Evolutionary: python3 scripts/run_pipeline.py --evolutionary --generations 10
```

Running the web app
-------------------
From project root (with dependencies installed):

**One command (backend + frontend):**
```bash
bash scripts/run_dev.sh
```
Then open http://localhost:5173. Use Ctrl+C to stop both.

**Or run separately:**
1. **Backend**: `python scripts/run_app.py` or `PYTHONPATH=. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
2. **Frontend**: `cd frontend && npm install && npm run dev` (dev server at http://localhost:5173; proxies /api to backend)

Open http://localhost:5173 and use Generate / Analyze / Compare / About.

One-shot script (if all deps and data are ready): `bash scripts/run_full_pipeline.sh`
