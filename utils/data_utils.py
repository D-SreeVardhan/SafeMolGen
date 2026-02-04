"""Data loading utilities for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from tdc.benchmark_group import admet_group


@dataclass
class EndpointConfig:
    name: str
    category: str
    tdc_name: str
    task_type: str
    metric: str
    enabled: bool = True


def read_endpoints_config(config: Dict) -> List[EndpointConfig]:
    endpoints = []
    for item in config.get("endpoints", []):
        if not item.get("enabled", True):
            continue
        endpoints.append(
            EndpointConfig(
                name=item["name"],
                category=item["category"],
                tdc_name=item["tdc_name"],
                task_type=item["task_type"],
                metric=item.get("metric", "rmse"),
                enabled=item.get("enabled", True),
            )
        )
    return endpoints


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_smiles_column(df: pd.DataFrame) -> str:
    for col in ["Drug", "SMILES", "smiles"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find SMILES column in dataset.")


class TDCDataLoader:
    """Download and prepare TDC ADMET datasets."""

    def __init__(self, data_dir: Path, seed: int = 42) -> None:
        self.data_dir = data_dir
        self.seed = seed
        self.group = admet_group(path=str(self.data_dir))

    def fetch_endpoint_splits(
        self, endpoint: EndpointConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        benchmark = self.group.get(endpoint.tdc_name)
        train_val = benchmark["train_val"]
        test_df = benchmark["test"]

        smiles_col = _get_smiles_column(train_val)
        train_val = train_val.rename(columns={smiles_col: "smiles", "Y": "y"})
        test_df = test_df.rename(columns={smiles_col: "smiles", "Y": "y"})

        train_val = train_val[["smiles", "y"]]
        test_df = test_df[["smiles", "y"]]

        stratify: Optional[pd.Series] = None
        if endpoint.task_type == "classification":
            if train_val["y"].nunique() > 1:
                stratify = train_val["y"]

        train_df, val_df = train_test_split(
            train_val,
            test_size=0.1,
            random_state=self.seed,
            shuffle=True,
            stratify=stratify,
        )
        return train_df, val_df, test_df

    def save_raw(self, endpoint: EndpointConfig, df: pd.DataFrame) -> Path:
        raw_dir = self.data_dir / "raw"
        ensure_dir(raw_dir)
        path = raw_dir / f"tdc_{endpoint.name}.csv"
        df.to_csv(path, index=False)
        return path

    def save_splits(
        self,
        endpoint: EndpointConfig,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[Path, Path, Path]:
        processed_dir = self.data_dir / "processed" / "admet" / endpoint.name
        ensure_dir(processed_dir)

        train_path = processed_dir / "train.csv"
        val_path = processed_dir / "val.csv"
        test_path = processed_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        return train_path, val_path, test_path
