"""RoBERTa-base sequence classifier used by the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

from transformers import AutoTokenizer, RobertaForSequenceClassification

from utils import MODEL_NAME, NUM_LABELS

PathLike = Union[str, Path]


def load_tokenizer(model_name_or_path: PathLike = MODEL_NAME):
    """Load the RoBERTa tokenizer (pretrained or from a saved folder)."""
    return AutoTokenizer.from_pretrained(str(model_name_or_path))


def load_model(model_name_or_path: PathLike = MODEL_NAME, num_labels: int = NUM_LABELS):
    """
    Load RobertaForSequenceClassification.

    Architecture:
        input ids → RoBERTa-base (12 encoder layers) → classification head → 2 logits
    """
    return RobertaForSequenceClassification.from_pretrained(
        str(model_name_or_path),
        num_labels=num_labels,
    )


def load_saved_classifier(model_dir: PathLike) -> Tuple[object, object]:
    """Load a fine-tuned model folder produced by the training notebook."""
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Unzip roberta_headline_only.zip into models/ first."
        )
    try:
        tokenizer = load_tokenizer(path)
    except Exception:
        # Colab export may omit vocab.json/merges.txt; the base tokenizer is equivalent.
        tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(path)
    model.eval()
    return tokenizer, model
