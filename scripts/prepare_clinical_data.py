"""Create data/processed/oracle/clinical_trials.csv for Oracle training (Phase 2).

Synthetic version: aggregate SMILES from data/admet_group (or processed/admet),
assign synthetic phase1/phase2/phase3 labels (e.g. random or simple rules)
so train_oracle.py can run. Run from project root with PYTHONPATH=.
"""

from pathlib import Path

import pandas as pd


def _get_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Drug", "SMILES", "smiles"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find SMILES column.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    admet_base = project_root / "data" / "admet_group"
    out_path = project_root / "data" / "processed" / "oracle" / "clinical_trials.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect unique valid SMILES from admet_group
    seen = set()
    smiles_list = []
    for csv_path in sorted(admet_base.glob("*/train_val.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            col = _get_smiles_col(df)
            for s in df[col].astype(str).dropna():
                s = s.strip()
                if s and s not in seen:
                    try:
                        from rdkit import Chem
                        if Chem.MolFromSmiles(s) is not None:
                            seen.add(s)
                            smiles_list.append(s)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Skip {csv_path}: {e}")
        if len(smiles_list) >= 10000:
            break

    if len(smiles_list) < 100:
        raise ValueError(f"Too few valid SMILES ({len(smiles_list)}). Need at least 100.")

    # Synthetic labels: random binary phase1/phase2/phase3 in [0,1]
    import random
    random.seed(42)
    rows = []
    for smi in smiles_list[:5000]:
        phase1 = float(random.random() > 0.3)
        phase2 = float(random.random() > 0.5) if phase1 > 0.5 else 0.0
        phase3 = float(random.random() > 0.6) if phase2 > 0.5 else 0.0
        rows.append({"smiles": smi, "phase1": phase1, "phase2": phase2, "phase3": phase3})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
