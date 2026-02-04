"""Preprocess datasets into graph tensors (Phase 1)."""

from pathlib import Path

import pandas as pd
import torch
import yaml

from utils.chemistry import MoleculeProcessor
from utils.data_utils import read_endpoints_config


def _build_graphs(df: pd.DataFrame, processor: MoleculeProcessor):
    graphs = []
    for _, row in df.iterrows():
        graph = processor.smiles_to_graph(row["smiles"])
        if graph is None:
            continue
        graph.y = torch.tensor([row["y"]], dtype=torch.float)
        graphs.append(graph)
    return graphs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    endpoints_path = project_root / "config" / "endpoints.yaml"
    with open(endpoints_path, "r", encoding="utf-8") as f:
        endpoints_cfg = yaml.safe_load(f)

    endpoints = read_endpoints_config(endpoints_cfg)
    processor = MoleculeProcessor()

    for endpoint in endpoints:
        base_dir = project_root / "data" / "processed" / "admet" / endpoint.name
        for split in ["train", "val", "test"]:
            csv_path = base_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"Missing {csv_path}, run download_data.py first.")
                continue
            df = pd.read_csv(csv_path)
            graphs = _build_graphs(df, processor)
            out_path = base_dir / f"{split}.pt"
            torch.save(graphs, out_path)
            print(f"Saved {split} graphs for {endpoint.name}: {len(graphs)}")


if __name__ == "__main__":
    main()
