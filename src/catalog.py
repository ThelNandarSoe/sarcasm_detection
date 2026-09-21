"""Unified catalog for RoBERTa plus the five Keras models."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from keras_inference import keras_metadata, list_keras_ids
from utils import resolve_roberta_dir

ROBERTA_ID = "roberta"
ROBERTA_PARAMS = 124647170


def _roberta_entry() -> Dict[str, Any]:
    path = resolve_roberta_dir()
    meta_path = path / "export_meta.json"
    metrics = {}
    if meta_path.exists():
        metrics = json.loads(meta_path.read_text(encoding="utf-8")).get("test_metrics") or {}
    accuracy = metrics.get("accuracy")
    return {
        "id": ROBERTA_ID,
        "name": "RoBERTa-base",
        "architecture": "Transformer encoder + classification head",
        "accuracy": accuracy,
        "validation_accuracy": accuracy,
        "f1": metrics.get("f1"),
        "parameters": ROBERTA_PARAMS,
        "filename": path.name if path.name != "models" else "roberta_headline_only",
        "family": "transformer",
        "description": "Colab-trained RoBERTa-base on News Headlines (headline only).",
        "ready": (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists(),
        "tokenizer": "RoBERTa tokenizer",
    }


def list_models() -> List[Dict[str, Any]]:
    models = [_roberta_entry()]
    meta = keras_metadata()
    for model_id in list_keras_ids():
        info = meta.get(model_id, {})
        models.append(
            {
                "id": model_id,
                "name": info.get("name") or model_id,
                "architecture": info.get("architecture") or "Keras",
                "accuracy": info.get("accuracy"),
                "validation_accuracy": info.get("validation_accuracy"),
                "f1": info.get("f1"),
                "parameters": info.get("parameters"),
                "filename": info.get("filename") or f"{model_id}.keras",
                "family": "keras",
                "description": info.get("description") or "",
                "ready": True,
                "tokenizer": "tokenizer.pkl",
            }
        )
    return models
