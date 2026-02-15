"""FastAPI application for SafeMolGen-DrugOracle."""
import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path when running as uvicorn backend.main:app from repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.pipeline_loader import load_pipeline
from backend.molecule_svg import draw_molecule_2d


# --- Pydantic schemas ---

class AnalyzeRequest(BaseModel):
    smiles: str


class CompareRequest(BaseModel):
    smiles_list: List[str] = Field(..., min_length=2)


class DesignRequest(BaseModel):
    target_success: float = 0.7
    max_iterations: int = 10
    candidates_per_iteration: int = 350
    top_k: int = 40
    safety_threshold: float = 0.2
    require_no_structural_alerts: bool = False
    property_targets: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Molecular/ADMET targets: logp (range), mw, hbd, hba, tpsa, qed, solubility, ppbr, clearance_hepatocyte_max",
    )
    seed_smiles: Optional[str] = None
    use_rl_model: bool = False
    # Plan 5.1 & 5.2 & 4.3: all solutions
    selection_mode: str = "phase_weighted"  # phase_weighted | overall | pareto | diversity | bottleneck
    diversity_tanimoto_max: float = 0.7
    n_restarts: int = 0  # >0: run design_molecule_with_restarts
    design_mode: str = "single"  # single | restarts | evolutionary
    use_reranker: bool = True  # use two-stage reranker when pipeline has it
    reranker_top_k: int = 200
    population_size: int = 20
    generations: int = 10
    ensure_target: bool = False  # if True, on first run miss retry with restarts then all-solutions; return best
    # Weak start / improvement visibility (optional early checkpoint + first-iter temperature)
    exploration_fraction: Optional[float] = None
    use_phase_aware_steering: Optional[bool] = None
    first_iteration_temperature: Optional[float] = None


def _get_pipeline(use_rl_model: bool = False):
    """Lazy-load pipeline; cache in app.state."""
    state = app.state
    key = "pipeline_rl" if use_rl_model else "pipeline"
    if not getattr(state, key, None):
        setattr(state, key, load_pipeline(use_rl_model=use_rl_model))
    return getattr(state, key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.pipeline_rl = None
    yield
    # cleanup if needed
    app.state.pipeline = None
    app.state.pipeline_rl = None


app = FastAPI(
    title="SafeMolGen-DrugOracle API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Routes ---

@app.get("/api/v1/health")
def health():
    pipeline = _get_pipeline(use_rl_model=False)
    return {
        "status": "ok",
        "models_loaded": pipeline is not None,
    }


@app.get("/api/v1/config")
def config():
    pipeline = _get_pipeline(use_rl_model=False)
    return {
        "max_iterations_min": 1,
        "max_iterations_max": 20,
        "target_success_min": 0.1,
        "target_success_max": 0.95,
        "selection_modes": ["overall", "pareto", "diversity", "phase_weighted", "bottleneck"],
        "design_modes": ["single", "restarts", "evolutionary"],
        "has_reranker": pipeline is not None and getattr(pipeline, "reranker", None) is not None,
        "first_iteration_temperature_default": 1.4,
        "generator_early_available": pipeline is not None and getattr(pipeline, "generator_early", None) is not None,
        "default_property_targets": {
            "logp": [2.0, 5.0],
            "mw_min": 150,
            "mw": 500,
            "hbd": 5,
            "hba": 10,
            "tpsa": 140.0,
            "qed": 0.5,
        },
    }


@app.get("/api/v1/molecule/svg")
def molecule_svg(smiles: str, width: int = 400, height: int = 300):
    if not smiles or not smiles.strip():
        raise HTTPException(status_code=400, detail="smiles required")
    svg = draw_molecule_2d(smiles.strip(), size=(width, height))
    if svg is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/api/v1/analyze")
def analyze(req: AnalyzeRequest):
    pipeline = _get_pipeline(use_rl_model=False)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="smiles required")
    prediction = pipeline.evaluate_molecule(smiles)
    if prediction is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    return {
        "smiles": smiles,
        "prediction": prediction.to_dict(),
    }


@app.post("/api/v1/compare")
def compare(req: CompareRequest):
    pipeline = _get_pipeline(use_rl_model=False)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    smiles_list = [s.strip() for s in req.smiles_list if s.strip()]
    if len(smiles_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 SMILES required")
    results = pipeline.compare_molecules(smiles_list)
    return {"results": results}


def _design_kw(req: DesignRequest):
    seed = (req.seed_smiles or "").strip() or None
    kw = {
        "target_success": req.target_success,
        "max_iterations": req.max_iterations,
        "candidates_per_iteration": req.candidates_per_iteration,
        "top_k": req.top_k,
        "safety_threshold": req.safety_threshold,
        "require_no_structural_alerts": req.require_no_structural_alerts,
        "property_targets": req.property_targets,
        "seed_smiles": seed,
        "use_oracle_feedback": True,
        "show_progress": False,
        "on_iteration_done": None,
        "selection_mode": getattr(req, "selection_mode", "overall"),
        "diversity_tanimoto_max": getattr(req, "diversity_tanimoto_max", 0.7),
        "use_reranker": getattr(req, "use_reranker", True),
        "reranker_top_k": getattr(req, "reranker_top_k", 200),
    }
    if req.exploration_fraction is not None:
        kw["exploration_fraction"] = req.exploration_fraction
    if req.use_phase_aware_steering is not None:
        kw["use_phase_aware_steering"] = req.use_phase_aware_steering
    if req.first_iteration_temperature is not None:
        kw["first_iteration_temperature"] = req.first_iteration_temperature
    return kw


def _run_one_design(pipeline, req: DesignRequest, kw: dict) -> tuple:
    """Run one design; return (result, strategy_name)."""
    design_mode = getattr(req, "design_mode", "single")
    n_restarts = getattr(req, "n_restarts", 0) or (5 if design_mode == "restarts" else 0)
    if design_mode == "evolutionary":
        result = pipeline.design_molecule_evolutionary(
            population_size=getattr(req, "population_size", 20),
            generations=getattr(req, "generations", 10),
            target_success=req.target_success,
            show_progress=False,
        )
        return result, "evolutionary"
    if n_restarts > 0:
        result = pipeline.design_molecule_with_restarts(
            n_restarts=n_restarts,
            show_progress=False,
            **kw,
        )
        return result, "restarts"
    result = pipeline.design_molecule(**kw)
    return result, "single"


@app.post("/api/v1/design")
def design_sync(req: DesignRequest):
    pipeline = _get_pipeline(use_rl_model=req.use_rl_model)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    kw = _design_kw(req)

    if getattr(req, "ensure_target", False):
        best_result = None
        best_strategy = None
        # 1) single run
        result = pipeline.design_molecule(**kw)
        best_result, best_strategy = result, "single"
        if not result.target_achieved:
            # 2) restarts=5
            result = pipeline.design_molecule_with_restarts(n_restarts=5, show_progress=False, **kw)
            if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                best_result, best_strategy = result, "restarts_5"
            if not best_result.target_achieved:
                # 3) all-solutions (restarts + diversity)
                result = pipeline.design_molecule_with_restarts(
                    n_restarts=5, show_progress=False, **{**kw, "selection_mode": "diversity"},
                )
                if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                    best_result, best_strategy = result, "all_solutions"
        result = best_result
        out = result.to_dict()
        out["recommendations"] = result.final_prediction.recommendations
        out["_strategy_used"] = best_strategy
        return out

    result, _ = _run_one_design(pipeline, req, kw)
    out = result.to_dict()
    out["recommendations"] = result.final_prediction.recommendations
    return out


def _sse_message(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/api/v1/design/stream")
async def design_stream(req: DesignRequest):
    pipeline = _get_pipeline(use_rl_model=req.use_rl_model)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")

    q: queue.Queue = queue.Queue()

    def on_iteration_done(partial_result):
        # partial_result is DesignResult with current state
        q.put(("iteration", partial_result.to_dict()))

    seed = (req.seed_smiles or "").strip() or None

    def run_design():
        try:
            kw = _design_kw(req)
            kw["on_iteration_done"] = on_iteration_done
            if getattr(req, "ensure_target", False):
                best_result = None
                best_strategy = None
                result = pipeline.design_molecule(**kw)
                best_result, best_strategy = result, "single"
                if not result.target_achieved:
                    kw_no_cb = {k: v for k, v in kw.items() if k != "on_iteration_done"}
                    result = pipeline.design_molecule_with_restarts(n_restarts=5, show_progress=False, **kw_no_cb)
                    if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                        best_result, best_strategy = result, "restarts_5"
                    if not best_result.target_achieved:
                        result = pipeline.design_molecule_with_restarts(
                            n_restarts=5, show_progress=False, **{**kw_no_cb, "selection_mode": "diversity"},
                        )
                        if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                            best_result, best_strategy = result, "all_solutions"
                result = best_result
                out = result.to_dict()
                out["recommendations"] = result.final_prediction.recommendations
                out["_strategy_used"] = best_strategy
                q.put(("done", out))
            else:
                result, _ = _run_one_design(pipeline, req, kw)
                out = result.to_dict()
                out["recommendations"] = result.final_prediction.recommendations
                q.put(("done", out))
        except Exception as e:
            q.put(("error", {"detail": str(e)}))

    thread = threading.Thread(target=run_design)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            while True:
                kind, payload = await loop.run_in_executor(None, q.get)
                if kind == "iteration":
                    yield _sse_message({"event": "iteration", "data": payload})
                elif kind == "done":
                    yield _sse_message({"event": "done", "data": payload})
                    break
                elif kind == "error":
                    yield _sse_message({"event": "error", "data": payload})
                    break
        finally:
            thread.join(timeout=0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
