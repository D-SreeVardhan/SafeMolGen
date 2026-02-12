"""Train SafeMolGen (Phase 3).

Pretrain trains a CONDITIONED generator (cond_dim=25) so the pipeline can steer
generation with Oracle feedback. Saves to checkpoints/generator by default.

Pretrain expects generator SMILES data at data/processed/generator/smiles.tsv
(e.g. from scripts/download_chembl_smiles.py with canonical_smiles column).
If that path is missing, SMILES are aggregated from data/admet_group/*/train_val.csv
as a fallback.

RL stage requires a pretrained generator (--resume checkpoints/generator) and loads
the DrugOracle to use overall_prob as reward. Saves to checkpoints/generator_rl by default.
To get a conditioned model: run pretrain first, then RL; the pipeline will use the condition vector.
"""

from pathlib import Path
from typing import Callable, Optional

import argparse
import torch
import yaml

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import TransformerDecoderModel, COND_DIM
from models.generator.trainer import PretrainConfig, train_pretrain
from models.generator.rl_trainer import RLConfig, train_rl
from models.generator.safemolgen import SafeMolGen
from models.generator.cond_dataset import CondSMILESDataset
from utils.data_utils import load_and_prepare_smiles, aggregate_admet_smiles, read_endpoints_config
from utils.checkpoint_utils import get_admet_node_feature_dim
from utils.chemistry import validate_smiles
from utils.condition_vector import get_target_condition_for_rl

# Defaults from configs (for argparse fallback)
_DEFAULT_PRETRAIN_EPOCHS = 30
_DEFAULT_RL_EPOCHS = 5
_DEFAULT_RL_BATCH = 8
_VALIDITY_SAMPLE_SIZE = 200

# Production defaults (best model possible)
_PRODUCTION_PRETRAIN_EPOCHS = 30
_PRODUCTION_PRETRAIN_LIMIT = 200_000
_PRODUCTION_PRETRAIN_BATCH = 128
_PRODUCTION_RL_EPOCHS = 15
_PRODUCTION_RL_BATCH = 16
_PRETRAIN_CHECKPOINT_EVERY = 10
_PRETRAIN_CHECKPOINT_EVERY_PRODUCTION = 5


def _make_oracle_for_pretrain(project_root: Path, device: str = "cpu"):
    """Load DrugOracle for conditioned pretrain, or None if checkpoints missing."""
    oracle_path = project_root / "checkpoints" / "oracle" / "best_model.pt"
    admet_path = project_root / "checkpoints" / "admet" / "best_model.pt"
    endpoints_path = project_root / "config" / "endpoints.yaml"
    if not oracle_path.exists() or not admet_path.exists() or not endpoints_path.exists():
        return None
    from models.oracle.drug_oracle import DrugOracle
    endpoints_cfg = yaml.safe_load(endpoints_path.read_text(encoding="utf-8"))
    endpoints = read_endpoints_config(endpoints_cfg)
    endpoint_names = [e.name for e in endpoints]
    endpoint_task_types = {e.name: e.task_type for e in endpoints}
    input_dim = get_admet_node_feature_dim(str(admet_path))
    return DrugOracle.from_pretrained(
        oracle_path=str(oracle_path),
        admet_path=str(admet_path),
        endpoint_names=endpoint_names,
        endpoint_task_types=endpoint_task_types,
        input_dim=input_dim,
        device=device,
    )


def _make_oracle_score_fn(project_root: Path, device: str = "cpu") -> Optional[Callable[[str], float]]:
    """Build a callable that scores SMILES by Oracle overall_prob, or None if Oracle not available."""
    oracle = _make_oracle_for_pretrain(project_root, device=device)
    if oracle is None:
        return None

    def score_fn(smiles: str) -> float:
        if not validate_smiles(smiles):
            return 0.0
        pred = oracle.predict(smiles)
        return pred.overall_prob if pred else 0.0

    return score_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "rl"], default="pretrain")
    parser.add_argument("--data", type=str, default="data/processed/generator/smiles.tsv")
    parser.add_argument("--out", type=str, default=None, help="Output dir (default: generator_rl for RL, generator for pretrain)")
    parser.add_argument("--resume", type=str, default=None, help="Pretrained checkpoint to resume (required for RL)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (pretrain or RL)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--limit", type=int, default=100000, help="Max SMILES to load for pretrain")
    parser.add_argument("--no-canonicalize", action="store_true", help="Skip RDKit canonicalization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data shuffle and reproducibility")
    parser.add_argument("--w-validity", type=float, default=None, help="RL: weight for validity reward (default 0.75)")
    parser.add_argument("--w-diversity", type=float, default=None, help="RL: weight for diversity/uniqueness reward (default 0.05)")
    parser.add_argument("--production", action="store_true", help="Use production defaults: 30ep pretrain, 200k SMILES, batch 128, 15ep RL, batch 16, best checkpointing")
    parser.add_argument("--d-model", type=int, default=None, help="Transformer d_model (default 256; use 384 or 512 for larger model)")
    parser.add_argument("--num-layers", type=int, default=None, help="Transformer num_layers (default 6; use 8 for larger model)")
    args = parser.parse_args()

    production = getattr(args, "production", False)
    model_config_for_save = None
    if production:
        print("Production mode: 30ep pretrain, 200k SMILES, batch 128, best-by-validity; 15ep RL, batch 16, best-by-reward.")

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    if args.stage == "pretrain":
        limit = _PRODUCTION_PRETRAIN_LIMIT if production else args.limit
        if data_path.exists():
            smiles_list = load_and_prepare_smiles(
                data_path,
                limit=limit,
                canonicalize=not args.no_canonicalize,
                write_cleaned_path=None,
            )
        else:
            admet_base = project_root / "data" / "admet_group"
            if not admet_base.exists():
                raise FileNotFoundError(
                    f"SMILES data not found at {data_path}. "
                    "Run scripts/download_chembl_smiles.py to create it, or ensure data/admet_group exists for fallback."
                )
            smiles_list = aggregate_admet_smiles(admet_base, limit=limit)
        if not smiles_list:
            raise ValueError("No valid SMILES to train on. Check data and validate_smiles.")
        tokenizer = SMILESTokenizer(max_length=128)
        tokenizer.fit(smiles_list)
        d_model = args.d_model if args.d_model is not None else 256
        num_layers = args.num_layers if args.num_layers is not None else 6
        dim_feedforward = getattr(args, "dim_feedforward", None) or max(512, 2 * d_model)
        model = TransformerDecoderModel(
            vocab_size=tokenizer.vocab_size,
            cond_dim=COND_DIM,
            d_model=d_model,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )
        pretrain_epochs = args.epochs if args.epochs is not None else (_PRODUCTION_PRETRAIN_EPOCHS if production else _DEFAULT_PRETRAIN_EPOCHS)
        pretrain_batch = args.batch_size if args.batch_size is not None else (_PRODUCTION_PRETRAIN_BATCH if production else 64)
        config = PretrainConfig(
            epochs=pretrain_epochs,
            batch_size=pretrain_batch,
            grad_clip=1.0,
            use_cosine_lr=True,
            shuffle_seed=args.seed,
        )

        out_dir = Path(args.out) if args.out else (project_root / "checkpoints" / "generator")
        out_dir.mkdir(parents=True, exist_ok=True)

        best_validity = [0.0]
        checkpoint_every = _PRETRAIN_CHECKPOINT_EVERY_PRODUCTION if production else _PRETRAIN_CHECKPOINT_EVERY
        model_config = {"d_model": d_model, "nhead": 8, "num_layers": num_layers, "dim_feedforward": dim_feedforward, "dropout": 0.1, "cond_dim": getattr(model, "cond_dim", 0)}

        def on_epoch_end(epoch: int, m, tok):
            gen = SafeMolGen(tok, m)
            cond = torch.zeros(1, getattr(m, "cond_dim", 0), device=config.device, dtype=torch.float32) if getattr(m, "cond_dim", 0) > 0 else None
            samples = gen.generate(n=_VALIDITY_SAMPLE_SIZE, temperature=0.8, device=config.device, condition=cond)
            valid = sum(1 for s in samples if validate_smiles(s))
            unique = len(set(samples))
            pct_valid = 100.0 * valid / max(len(samples), 1)
            pct_unique = 100.0 * unique / max(len(samples), 1)
            print(f"Epoch {epoch} validity: {valid}/{len(samples)} ({pct_valid:.1f}%) | uniqueness: {unique}/{len(samples)} ({pct_unique:.1f}%)")
            if valid == 0 and samples:
                print("Sample generated (first 20, for debugging):")
                for s in samples[:20]:
                    print("  ", repr(s))
            if pct_valid > best_validity[0]:
                best_validity[0] = pct_valid
                (out_dir / "best").mkdir(parents=True, exist_ok=True)
                gen.save(str(out_dir / "best"), config=model_config)
                print(f"New best validity {pct_valid:.1f}%, saved to {out_dir}/best")
            if epoch % checkpoint_every == 0:
                gen.save(str(out_dir), config=model_config)
                print(f"Checkpoint saved at epoch {epoch}")

        # Use real Oracle condition vectors when available so the generator learns to steer
        pretrain_dataset = None
        oracle = _make_oracle_for_pretrain(project_root, device=config.device)
        if oracle is not None and getattr(model, "cond_dim", 0) > 0:
            print("Using conditioned pretrain: Oracle loaded, building condition vectors per SMILES.")
            pretrain_dataset = CondSMILESDataset(
                smiles_list,
                tokenizer,
                oracle,
                device=config.device,
                zero_condition_fraction=0.1,
            )
        train_pretrain(model, tokenizer, smiles_list, config, on_epoch_end=on_epoch_end, dataset=pretrain_dataset)
        if (out_dir / "best" / "model.pt").exists():
            gen = SafeMolGen.from_pretrained(str(out_dir / "best"), device=config.device)
            model, tokenizer = gen.model, gen.tokenizer
        else:
            gen = SafeMolGen(tokenizer, model)
        model_config_for_save = model_config
    else:
        rl_kw = {}
        if args.w_validity is not None:
            rl_kw["w_validity"] = args.w_validity
        if args.w_diversity is not None:
            rl_kw["w_diversity"] = args.w_diversity
        rl_epochs = args.epochs if args.epochs is not None else (_PRODUCTION_RL_EPOCHS if production else _DEFAULT_RL_EPOCHS)
        rl_batch = args.batch_size if args.batch_size is not None else (_PRODUCTION_RL_BATCH if production else _DEFAULT_RL_BATCH)
        config = RLConfig(
            epochs=rl_epochs,
            batch_size=rl_batch,
            **rl_kw,
        )
        resume_path = Path(args.resume) if args.resume else (project_root / "checkpoints" / "generator")
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        if not (resume_path / "model.pt").exists() or not (resume_path / "tokenizer.json").exists():
            raise ValueError(
                f"RL requires a pretrained checkpoint at {resume_path}. "
                "Run pretrain first: python scripts/train_generator.py --stage pretrain, then "
                "python scripts/train_generator.py --stage rl [--resume checkpoints/generator]"
            )
        gen = SafeMolGen.from_pretrained(str(resume_path), device=config.device)
        tokenizer = gen.tokenizer
        model = gen.model
        out_dir = Path(args.out) if args.out else (project_root / "checkpoints" / "generator_rl")
        out_dir.mkdir(parents=True, exist_ok=True)
        _resume_state = torch.load(resume_path / "model.pt", map_location=config.device, weights_only=False)
        _resume_cfg = _resume_state.get("config", {})
        rl_model_config = {
            "d_model": _resume_cfg.get("d_model", 256),
            "nhead": _resume_cfg.get("nhead", 8),
            "num_layers": _resume_cfg.get("num_layers", 6),
            "dim_feedforward": _resume_cfg.get("dim_feedforward", 512),
            "dropout": _resume_cfg.get("dropout", 0.1),
            "cond_dim": _resume_cfg.get("cond_dim", 0),
        }
        quick_samples = gen.generate(n=50, temperature=0.8, device=config.device)
        quick_valid = sum(1 for s in quick_samples if validate_smiles(s))
        if quick_valid == 0:
            print("Recommendation: pretrain validity is 0%. Run more pretrain epochs before RL for better results.")
        oracle_score_fn = _make_oracle_score_fn(project_root, device=config.device)
        if oracle_score_fn is None:
            print("Warning: Oracle not found (checkpoints/oracle, admet, config/endpoints.yaml). RL will use validity+QED only.")
        target_condition = None
        if getattr(model, "cond_dim", 0) > 0:
            target_condition = get_target_condition_for_rl(device=config.device)
            print("Using target condition for RL sampling (high phase-prob profile).")
        best_reward = [float("-inf")]

        def rl_epoch_cb(epoch: int, reward: float, validity: float) -> None:
            if reward > best_reward[0]:
                best_reward[0] = reward
                g = SafeMolGen(tokenizer, model)
                (out_dir / "best").mkdir(parents=True, exist_ok=True)
                g.save(str(out_dir / "best"), config=rl_model_config)
                print(f"New best reward {reward:.4f}, saved to {out_dir}/best")

        train_rl(model, tokenizer, config=config, oracle_score_fn=oracle_score_fn, target_condition=target_condition, on_epoch_end=rl_epoch_cb)
        if (out_dir / "best" / "model.pt").exists():
            gen = SafeMolGen.from_pretrained(str(out_dir / "best"), device=config.device)
            model, tokenizer = gen.model, gen.tokenizer
        else:
            gen = SafeMolGen(tokenizer, model)
        model_config_for_save = rl_model_config

    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config = model_config_for_save if model_config_for_save is not None else {"d_model": 256, "nhead": 8, "num_layers": 6, "dim_feedforward": 512, "dropout": 0.1, "cond_dim": getattr(gen.model, "cond_dim", 0)}
    print(f"Saving checkpoint to {out_dir} ...")
    gen.save(str(out_dir), config=save_config)
    print("Done.")


if __name__ == "__main__":
    main()
