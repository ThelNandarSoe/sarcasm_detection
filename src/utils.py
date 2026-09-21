"""
Shared paths, random seeds, and experiment configuration.

These defaults follow Helal et al. (2024), Scientific Reports,
"A contextual-based approach for sarcasm detection", Table 4.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONTEXTUAL_DIR = DATA_DIR / "contextual"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
KERAS_DIR = MODELS_DIR / "keras"
TOKENIZERS_DIR = MODELS_DIR / "tokenizers"
DEFAULT_MODEL_DIR = MODELS_DIR / "roberta_headline_only"


def resolve_roberta_dir() -> Path:
    """Find the unzipped RoBERTa folder even if files sit in models/."""
    candidates = (DEFAULT_MODEL_DIR, MODELS_DIR)
    for candidate in candidates:
        if (candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists():
            return candidate
    return DEFAULT_MODEL_DIR

# Paper Table 1 uses the original News Headlines file (26,709 records),
# not the later v2 release.
RAW_DATASET_V1 = RAW_DIR / "Sarcasm_Headlines_Dataset.json"
RAW_DATASET_V2 = RAW_DIR / "Sarcasm_Headlines_Dataset_v2.json"
DEFAULT_RAW_DATASET = RAW_DATASET_V1

PROCESSED_CSV = PROCESSED_DIR / "news_headlines_clean.csv"
INSPECTION_REPORT = PROCESSED_DIR / "inspection_report.json"

# ---------------------------------------------------------------------------
# Model / training (paper Table 4)
# ---------------------------------------------------------------------------
MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
SEED = 42
NUM_LABELS = 2

# Split: paper reports 80/20 train/test. We also hold out validation
# from the training portion so early-stopping / loss curves are honest.
TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAIN = 0.125  # 0.125 * 0.80 = 0.10 of the full dataset

# Input construction labels used later in ablation experiments.
CONTEXT_TYPES = (
    "headline_only",
    "author",
    "section",
    "description",
    "all_context",
)

UNKNOWN_TOKEN = "unknown"


def ensure_project_dirs() -> None:
    """Create data and results folders if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, CONTEXTUAL_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Set Python, NumPy, and (if available) PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_gpu_name() -> Optional[str]:
    """Return the CUDA device name, or None on CPU-only machines."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        return None
    return None
