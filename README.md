SafeMolGen-DrugOracle
=====================
Integrated AI system for drug design with ADMET prediction, clinical phase
success estimation, molecule generation, and a Streamlit UI.

Project Phases
--------------
1. Phase 1: Project setup + ADMET predictor
2. Phase 2: DrugOracle (clinical phase predictors + explainability)
3. Phase 3: SafeMolGen (molecular generator)
4. Phase 4: Integration + Streamlit UI

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
