"""Train SafeMolGen (Phase 3)."""

from pathlib import Path

import argparse
import pandas as pd

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import TransformerDecoderModel
from models.generator.trainer import PretrainConfig, train_pretrain
from models.generator.rl_trainer import RLConfig, train_rl
from models.generator.safemolgen import SafeMolGen

# Defaults from configs (for argparse fallback)
_DEFAULT_PRETRAIN_EPOCHS = 5
_DEFAULT_RL_EPOCHS = 5
_DEFAULT_RL_BATCH = 8


def _load_smiles(path: Path, limit: int = 50000):
    df = pd.read_csv(path, sep="\t")
    if "canonical_smiles" in df.columns:
        smiles = df["canonical_smiles"].dropna().tolist()
    elif "smiles" in df.columns:
        smiles = df["smiles"].dropna().tolist()
    else:
        smiles = df.iloc[:, 1].dropna().tolist()
    return smiles[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "rl"], default="pretrain")
    parser.add_argument("--data", type=str, default="data/processed/generator/smiles.tsv")
    parser.add_argument("--out", type=str, default="checkpoints/generator")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (pretrain or RL)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            "SMILES data not found. Provide a TSV with a smiles or canonical_smiles column."
        )

    if args.stage == "pretrain":
        smiles_list = _load_smiles(data_path)
        tokenizer = SMILESTokenizer(max_length=128)
        tokenizer.fit(smiles_list)
        model = TransformerDecoderModel(vocab_size=tokenizer.vocab_size)
        config = PretrainConfig(
            epochs=args.epochs if args.epochs is not None else _DEFAULT_PRETRAIN_EPOCHS,
            batch_size=args.batch_size if args.batch_size is not None else 64,
        )
        train_pretrain(model, tokenizer, smiles_list, config)
    else:
        config = RLConfig(
            epochs=args.epochs if args.epochs is not None else _DEFAULT_RL_EPOCHS,
            batch_size=args.batch_size if args.batch_size is not None else _DEFAULT_RL_BATCH,
        )
        resume_path = Path(args.resume) if args.resume else Path(args.out)
        if (resume_path / "model.pt").exists() and (resume_path / "tokenizer.json").exists():
            gen = SafeMolGen.from_pretrained(str(resume_path), device=config.device)
            tokenizer = gen.tokenizer
            model = gen.model
        else:
            smiles_list = _load_smiles(data_path)
            tokenizer = SMILESTokenizer(max_length=128)
            tokenizer.fit(smiles_list)
            model = TransformerDecoderModel(vocab_size=tokenizer.vocab_size)
        train_rl(model, tokenizer, config=config)

    out_dir = Path(args.out)
    gen = SafeMolGen(tokenizer, model)
    print(f"Saving checkpoint to {out_dir} ...")
    gen.save(
        str(out_dir),
        config={
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 512,
            "dropout": 0.1,
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()
