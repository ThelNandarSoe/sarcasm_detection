"""Load the five Keras headline models and run sigmoid predictions."""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils import KERAS_DIR, TOKENIZERS_DIR

MAX_LENGTH = 100
PADDING = "post"
TRUNCATING = "post"
THRESHOLD = 0.5

_LOCK = threading.Lock()
_MODELS: Dict[str, Any] = {}
_TOKENIZER = None
_KERAS_PATCHED = False


def keras_metadata() -> Dict[str, Any]:
    path = KERAS_DIR / "models.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_keras_compat() -> None:
    global _KERAS_PATCHED
    if _KERAS_PATCHED:
        return
    import importlib

    for module_name, class_name in (
        ("keras.layers", "Embedding"),
        ("keras.layers", "Dense"),
        ("keras.src.layers.core.embedding", "Embedding"),
        ("keras.src.layers.core.dense", "Dense"),
    ):
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception:
            continue
        original_init = cls.__init__

        def wrapped(self, *args, orig=original_init, **kwargs):
            kwargs.pop("quantization_config", None)
            return orig(self, *args, **kwargs)

        cls.__init__ = wrapped
    _KERAS_PATCHED = True


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    path = TOKENIZERS_DIR / "tokenizer.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Keras tokenizer not found: {path}")
    with path.open("rb") as handle:
        _TOKENIZER = pickle.load(handle)
    return _TOKENIZER


def get_keras_model(model_id: str):
    with _LOCK:
        cached = _MODELS.get(model_id)
        if cached is not None:
            return cached
        path = KERAS_DIR / f"{model_id}.keras"
        if not path.exists():
            raise FileNotFoundError(f"Keras model not found: {path}")
        from tensorflow.keras.models import load_model

        _patch_keras_compat()
        model = load_model(path, compile=False)
        _MODELS[model_id] = model
        return model


def _pad(sequences):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    return pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING, truncating=TRUNCATING)


def predict_keras(model_id: str, text: str) -> Dict[str, Any]:
    tokenizer = get_tokenizer()
    model = get_keras_model(model_id)
    sequences = tokenizer.texts_to_sequences([text])
    padded = _pad(sequences)
    raw = model.predict(padded, verbose=0)
    arr = np.squeeze(np.array(raw, dtype=np.float64))
    if np.ndim(arr) == 0:
        probability = float(arr)
    else:
        flat = np.reshape(arr, (-1,))
        probability = float(flat[-1] if flat.size == 2 else flat[0])
    probability = float(np.clip(probability, 0.0, 1.0))
    sarcastic = probability >= THRESHOLD
    return {
        "prediction": "Sarcastic" if sarcastic else "Non-sarcastic",
        "predicted_label": 1 if sarcastic else 0,
        "sarcasm_probability": probability,
        "sarcasm_percent": round(probability * 100.0, 2),
        "tokenizer": "tokenizer.pkl",
        "family": "keras",
        "model_id": model_id,
    }


def list_keras_ids() -> List[str]:
    return [path.stem for path in sorted(KERAS_DIR.glob("model_*.keras"))]
