"""Checkpoint utilities for loading and inferring model dimensions."""

from pathlib import Path
from typing import Optional

import torch


# Default from utils.chemistry._atom_features: 5 + 5 = 10
DEFAULT_ADMET_NODE_FEATURES = 10


def get_admet_node_feature_dim(checkpoint_path: str) -> int:
    """Infer num_node_features from ADMET checkpoint (encoder first layer weight shape).
    Use this instead of hardcoding so pipeline stays correct if atom features change.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return DEFAULT_ADMET_NODE_FEATURES
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model_state = state.get("model", state)
        # GNNEncoder first conv: encoder.convs.0.nn.0.weight has shape (hidden_dim, num_node_features)
        key = "encoder.convs.0.nn.0.weight"
        if key not in model_state:
            return DEFAULT_ADMET_NODE_FEATURES
        return int(model_state[key].shape[1])
    except Exception:
        return DEFAULT_ADMET_NODE_FEATURES
