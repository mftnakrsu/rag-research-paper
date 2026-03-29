"""Shared utilities: logging, timing, config loading, reproducibility."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = DATA_DIR / "results"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    if config_path is None:
        config_path = CONFIGS_DIR / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    return logger


@dataclass
class RetrievedDoc:
    """A single retrieved document with metadata."""
    doc_id: str
    text: str
    score: float
    rank: int
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Stores results for a single experiment run."""
    method: str
    config: dict[str, Any]
    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, float] | None = None
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    index_time_seconds: float = 0.0
    index_size_mb: float = 0.0
    num_queries: int = 0
    avg_latency_ms: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> ExperimentResult:
        with open(path) as f:
            return cls(**json.load(f))


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000
