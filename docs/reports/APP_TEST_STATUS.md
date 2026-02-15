# SafeMolGen-DrugOracle App – Test Status Report

**Date:** 2026-02-10  
**App URL:** http://localhost:5173 (React frontend); API http://localhost:8000  
**Tester:** Automated backend + manual UI checklist

---

## Summary

| Area | Status | Notes |
|------|--------|--------|
| Pipeline load (default generator) | PASS | cond_dim=25, conditioned checkpoint loads |
| Pipeline load (Use RL model) | PASS | generator_rl loads |
| Generate: design_molecule | PASS | 2 iterations, best ~2.2%, result with final_smiles and iteration_history |
| Analyze: evaluate_molecule | PASS | Oracle prediction for aspirin SMILES |
| Compare: compare_molecules | PASS | Multi-molecule comparison returns list |
| UI (in-browser) | PASS (E2E) | 16 Playwright E2E tests: all features + break-it cases (see tests/e2e_browser_test.py) |

---

## Backend Tests Run

- **Pipeline load** with `checkpoints/generator` and `checkpoints/generator_rl`: both load; generator has `cond_dim=25`.
- **Generate flow:** `design_molecule(target_success=0.25, max_iterations=2, safety_threshold=0.02, candidates_per_iteration=30)` completes; returns `DesignResult` with `final_smiles`, `final_prediction`, `iteration_history`, `target_achieved`, `total_iterations`.
- **Analyze flow:** `evaluate_molecule("CC(=O)Oc1ccccc1C(=O)O")` returns `OraclePrediction` with phase probs and recommendations.
- **Compare flow:** `compare_molecules(["CCO", "c1ccccc1"])` returns list of results.

---

## UI Manual Test Checklist (when using the browser tab)

1. **Generate**
   - [ ] On Generate page confirm Generation parameters (target success, max iterations, top-K, seed SMILES) and "Run generation" button are visible.
   - [ ] Click "Run generation"; confirm loading then Results (Best molecule, Oracle metrics, Optimization journey, Recommendations).
   - [ ] Expand "Show advanced" and toggle "Use RL model"; run generation again.
   - [ ] Try with Safety threshold 0.02 and Max iterations 3–5; confirm iteration chart and results.

2. **Analyze**
   - [ ] Click "Analyze" in sidebar; enter SMILES (e.g. `CC(=O)Oc1ccccc1C(=O)O`); click "Analyze"; confirm molecule card, Oracle dashboard, and recommendations.

3. **Compare**
   - [ ] Click "Compare" in sidebar; enter 2+ SMILES (one per line); click "Compare"; confirm comparison table.

4. **About**
   - [ ] Click "About" in sidebar; confirm overview text and bullets render.

---

## Conclusion

Backend code paths used by the app (load_pipeline, design_molecule, evaluate_molecule, compare_molecules) all pass. The app runs as FastAPI backend (port 8000) + React frontend (port 5173); in-browser checks can be done manually or with a browser MCP when available.
