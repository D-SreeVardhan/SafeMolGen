"""Load SafeMolGen-DrugOracle pipeline."""
import os
from pathlib import Path
from typing import Optional

import yaml

from utils.data_utils import read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from models.integrated.pipeline import SafeMolGenDrugOracle
from models.generator.safemolgen import SafeMolGen

# Project root: parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model_pt_path(p: Path) -> Optional[Path]:
    """Return path that contains model.pt: p, or p/best, or None."""
    if not p.exists():
        return None
    if (p / "model.pt").exists():
        return p
    if (p / "best" / "model.pt").exists():
        return p / "best"
    return None


def _get_early_generator_path() -> Optional[Path]:
    """Resolve optional early generator path: env, config, or convention."""
    env_path = os.environ.get("GENERATOR_EARLY_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return _model_pt_path(p)
    config_file = PROJECT_ROOT / "config" / "pipeline.yaml"
    if config_file.exists():
        try:
            cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            if isinstance(cfg, dict) and cfg.get("generator_early_path"):
                p = Path(cfg["generator_early_path"])
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                resolved = _model_pt_path(p)
                if resolved is not None:
                    return resolved
        except Exception:
            pass
    # Convention: checkpoints/generator_early (or checkpoints/generator_early/best)
    default_early = PROJECT_ROOT / "checkpoints" / "generator_early"
    return _model_pt_path(default_early)


def load_pipeline(use_rl_model: bool = False) -> Optional[SafeMolGenDrugOracle]:
    generator_path = (
        PROJECT_ROOT / "checkpoints" / "generator_rl"
        if use_rl_model
        else PROJECT_ROOT / "checkpoints" / "generator"
    )
    if not use_rl_model:
        # Prefer Best-of-N (improved) when available, else Option B best
        best_of_n = PROJECT_ROOT / "checkpoints" / "generator_best_of_n"
        if best_of_n.exists() and (best_of_n / "model.pt").exists():
            generator_path = best_of_n
        else:
            best_path = PROJECT_ROOT / "checkpoints" / "generator" / "best"
            if best_path.exists() and (best_path / "model.pt").exists():
                generator_path = best_path
    if not generator_path.exists():
        generator_path = PROJECT_ROOT / "checkpoints" / "generator"
    oracle_path = PROJECT_ROOT / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = PROJECT_ROOT / "checkpoints" / "admet" / "best_model.pt"
    if not generator_path.exists() or not oracle_path.exists() or not admet_path.exists():
        return None
    generator_early = None
    early_path = _get_early_generator_path()
    if early_path is not None:
        try:
            generator_early = SafeMolGen.from_pretrained(str(early_path), device="cpu")
        except Exception:
            generator_early = None
    reranker_path = None
    reranker_dir = PROJECT_ROOT / "checkpoints" / "reranker"
    if reranker_dir.exists() and (reranker_dir / "reranker.pt").exists():
        reranker_path = str(reranker_dir / "reranker.pt")
    endpoints_cfg = yaml.safe_load(
        (PROJECT_ROOT / "config" / "endpoints.yaml").read_text(encoding="utf-8")
    )
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    admet_input_dim = get_admet_node_feature_dim(str(admet_path))
    return SafeMolGenDrugOracle.from_pretrained(
        generator_path=str(generator_path),
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        admet_input_dim=admet_input_dim,
        device="cpu",
        reranker_path=reranker_path,
        generator_early=generator_early,
    )
