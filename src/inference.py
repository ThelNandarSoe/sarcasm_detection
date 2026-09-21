"""
Load a saved RoBERTa sarcasm model and predict on one headline.

After training, run:

    python src/inference.py

Or pass the saved folder explicitly:

    python src/inference.py --model_dir models/roberta_headline_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_input_text
from model import load_saved_classifier
from utils import DEFAULT_MODEL_DIR, MAX_LENGTH, UNKNOWN_TOKEN, get_device, resolve_roberta_dir

_CLASSIFIER_CACHE: Dict[str, Any] = {}


def get_classifier(model_dir: Optional[Path] = None):
    """Load tokenizer + model once and reuse them for later predictions."""
    model_dir = Path(model_dir) if model_dir else resolve_roberta_dir()
    key = str(model_dir.resolve())
    cached = _CLASSIFIER_CACHE.get(key)
    if cached is not None:
        return cached
    tokenizer, model = load_saved_classifier(model_dir)
    device = get_device()
    model.to(device)
    model.eval()
    cached = {"tokenizer": tokenizer, "model": model, "device": device, "model_dir": model_dir}
    _CLASSIFIER_CACHE[key] = cached
    return cached


def predict_one(
    headline: str,
    author: str = UNKNOWN_TOKEN,
    section: str = UNKNOWN_TOKEN,
    description: str = UNKNOWN_TOKEN,
    context_type: str = "headline_only",
    model_dir: Optional[Path] = None,
    max_length: int = MAX_LENGTH,
) -> Dict[str, Any]:
    """Return the class name and sarcasm probability for one example."""
    bundle = get_classifier(model_dir)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    model_dir = bundle["model_dir"]

    row = {
        "headline": headline,
        "author": author or UNKNOWN_TOKEN,
        "section": section or UNKNOWN_TOKEN,
        "description": description or UNKNOWN_TOKEN,
    }
    text = build_input_text(row, context_type)
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    sarcasm_prob = float(probabilities[1].item())
    predicted = int(torch.argmax(probabilities).item())
    label_name = "Sarcastic" if predicted == 1 else "Non-sarcastic"
    return {
        "prediction": label_name,
        "predicted_label": predicted,
        "sarcasm_probability": sarcasm_prob,
        "sarcasm_percent": round(sarcasm_prob * 100.0, 2),
        "input_text": text,
        "model_dir": str(model_dir),
    }


def _prompt(label: str, default: str = UNKNOWN_TOKEN) -> str:
    typed = input(f"{label}: ").strip()
    return typed if typed else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Sarcasm inference with a saved RoBERTa model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Folder created by the training notebook (contains config.json + weights).",
    )
    parser.add_argument(
        "--context_type",
        type=str,
        default="headline_only",
        help="Must match how the model was trained: headline_only, author, section, description, all_context.",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    print("Leave author/section/description blank if unused.\n")
    headline = _prompt("Headline", "")
    if not headline:
        raise SystemExit("Headline is required.")
    author = _prompt("Author")
    section = _prompt("Section")
    description = _prompt("Description")

    result = predict_one(
        headline=headline,
        author=author,
        section=section,
        description=description,
        context_type=args.context_type,
        model_dir=Path(args.model_dir),
    )
    print()
    print(f"Prediction: {result['prediction']}")
    print(f"Sarcasm probability: {result['sarcasm_probability'] * 100:.2f}%")


if __name__ == "__main__":
    main()
