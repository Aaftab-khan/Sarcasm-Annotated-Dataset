"""Read and validate YAML configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("model", "training", "data", "output"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Configuration must contain a mapping named {section!r}")
    model = config["model"]
    training = config["training"]
    if model.get("max_length", 0) < 3:
        raise ValueError("model.max_length must be at least 3")
    if training.get("batch_size", 0) < 1 or training.get("epochs", 0) < 1:
        raise ValueError("batch_size and epochs must be positive")
    if not 0 < float(training.get("validation_fraction", 0.1)) < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if not 0 <= float(training.get("warmup_ratio", 0.1)) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if model.get("labels") != ["O", "B-SARC", "I-SARC"]:
        raise ValueError("Labels must be ordered exactly as [O, B-SARC, I-SARC]")
    return config
