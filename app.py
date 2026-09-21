"""
Headline Desk — sarcasm dashboard.

RoBERTa-base plus five Keras models, same cream/ink theme.

    python app.py

Then open http://127.0.0.1:7872
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalog import ROBERTA_ID, list_models
from inference import get_classifier, predict_one
from keras_inference import predict_keras
from utils import UNKNOWN_TOKEN, get_device, get_gpu_name, resolve_roberta_dir

app = Flask(__name__)


def _find_model(model_id: str) -> dict:
    for item in list_models():
        if item["id"] == model_id:
            return item
    raise KeyError(model_id)


def run_prediction(model_id: str, payload: dict) -> dict:
    headline = (payload.get("headline") or payload.get("text") or "").strip()
    if not headline:
        raise ValueError("Please enter a news headline.")
    model_id = model_id or ROBERTA_ID
    info = _find_model(model_id)

    if info["family"] == "transformer":
        result = predict_one(
            headline=headline,
            author=(payload.get("author") or "").strip() or UNKNOWN_TOKEN,
            section=(payload.get("section") or "").strip() or UNKNOWN_TOKEN,
            description=(payload.get("description") or "").strip() or UNKNOWN_TOKEN,
            context_type="headline_only",
            model_dir=resolve_roberta_dir(),
        )
    else:
        result = predict_keras(model_id, headline)

    percent = round(float(result["sarcasm_probability"]) * 100.0, 2)
    result["sarcasm_percent"] = percent
    result["model_id"] = info["id"]
    result["model_name"] = info["name"]
    result["architecture"] = info["architecture"]
    result["filename"] = info["filename"]
    result["tokenizer"] = result.get("tokenizer") or info.get("tokenizer")
    result["input_text"] = result.get("input_text") or f"HEADLINE: {headline}"
    return result


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=list_models(),
        default_model=ROBERTA_ID,
    )


@app.route("/api/models")
def api_models():
    return jsonify({"models": list_models()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = run_prediction(payload.get("model_id") or ROBERTA_ID, payload)
        return jsonify(result)
    except KeyError:
        return jsonify({"error": "Unknown model."}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    payload = request.get_json(silent=True) or {}
    headline = (payload.get("headline") or payload.get("text") or "").strip()
    model_ids = payload.get("model_ids") or []
    if not headline:
        return jsonify({"error": "Please enter a news headline."}), 400
    if not isinstance(model_ids, list) or not model_ids:
        return jsonify({"error": "Select at least one model to compare."}), 400

    results = []
    errors = []
    for model_id in model_ids:
        try:
            prediction = run_prediction(model_id, payload)
            results.append(
                {
                    "model_id": prediction["model_id"],
                    "name": prediction["model_name"],
                    "prediction": prediction["prediction"],
                    "predicted_label": prediction["predicted_label"],
                    "sarcasm_percent": round(float(prediction.get("sarcasm_percent", prediction["sarcasm_probability"] * 100)), 2),
                    "sarcasm_probability": prediction["sarcasm_probability"],
                }
            )
        except Exception as exc:
            errors.append({"model_id": model_id, "error": str(exc)})
    return jsonify({"results": results, "errors": errors})


def main() -> None:
    roberta_dir = resolve_roberta_dir()
    if not (roberta_dir / "model.safetensors").exists():
        raise SystemExit(f"RoBERTa weights not found in {roberta_dir}")
    print(f"Loading RoBERTa from {roberta_dir} ...")
    get_classifier(roberta_dir)
    print(f"Device: {get_device()} | GPU: {get_gpu_name() or 'CPU'}")
    print("Open http://127.0.0.1:7872")
    app.run(host="127.0.0.1", port=7872, debug=False)


if __name__ == "__main__":
    main()
