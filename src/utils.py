import random

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds so that experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str = "configs/base.yaml") -> dict:
    """Load a YAML config file into a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)