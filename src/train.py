"""Fine-tune RoBERTa-base and save a reusable model folder."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import (
    add_input_text,
    maybe_sample,
    stratified_splits,
    token_length_stats,
    tokenize_texts,
)
from evaluate import (
    classification_report_text,
    compute_metrics_dict,
    error_type,
    save_confusion_matrix,
    save_error_analysis,
    save_loss_curve,
)
from model import load_model, load_tokenizer
from preprocessing import run_inspection
from utils import (
    DEFAULT_MODEL_DIR,
    EPOCHS,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    PROCESSED_CSV,
    RESULTS_DIR,
    SEED,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    ensure_project_dirs,
    get_device,
    get_gpu_name,
    set_seed,
)


class SarcasmDataset(Dataset):
    """Simple PyTorch dataset wrapping tokenizer encodings."""

    def __init__(self, encodings: Dict[str, Any], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def load_training_frame(max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load the processed CSV, creating it from the raw JSON if needed."""
    ensure_project_dirs()
    if not PROCESSED_CSV.exists():
        print("Processed CSV not found. Running preprocessing first...")
        run_inspection()
    frame = pd.read_csv(PROCESSED_CSV)
    return maybe_sample(frame, max_samples=max_samples, seed=SEED)


def _forward_loss(model, batch, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # RobertaForSequenceClassification uses CrossEntropyLoss internally when labels are passed.
    return outputs.loss, outputs.logits, labels


def _run_epoch(model, loader, device: str, optimizer=None) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: List[float] = []
    all_labels: List[int] = []
    all_preds: List[int] = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            if train_mode:
                optimizer.zero_grad()
            loss, logits, labels = _forward_loss(model, batch, device)
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics_dict(all_labels, all_preds)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def _predict_loader(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="test"):
            loss, logits, labels = _forward_loss(model, batch, device)
            losses.append(float(loss.item()))
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().tolist())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.asarray(all_labels)
    y_pred = logits.argmax(axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    return y_true, y_pred, probs, float(np.mean(losses) if losses else float("nan"))


def run_experiment(
    context_type: str = "headline_only",
    max_samples: Optional[int] = None,
    model_dir: Optional[Path] = None,
    max_length: int = MAX_LENGTH,
    epochs: int = EPOCHS,
    train_batch_size: int = TRAIN_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Train RoBERTa on one ablation setting and save the model folder.

    The saved folder can later be loaded with:

        from model import load_saved_classifier
        tokenizer, model = load_saved_classifier("models/roberta_headline_only")
    """
    set_seed(SEED)
    ensure_project_dirs()
    device = get_device()
    gpu_name = get_gpu_name()
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    print(f"Device: {device}")
    print(f"GPU name: {gpu_name or 'None (CPU)'}")
    if device == "cpu":
        print(
            "Warning: this is CPU-only PyTorch. A full 5-epoch run on 26k headlines "
            "can take many hours. Set max_samples=2000 for a smoke test."
        )

    frame = load_training_frame(max_samples=max_samples)
    frame = add_input_text(frame, context_type)
    train_df, val_df, test_df = stratified_splits(frame, seed=SEED)

    print(f"Dataset size: {len(frame)}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Context type: {context_type}")
    print(f"Epochs: {epochs}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Model will be saved to: {model_dir}")

    tokenizer = load_tokenizer(MODEL_NAME)
    length_stats = token_length_stats(frame["input_text"].tolist(), tokenizer)
    printable = {k: v for k, v in length_stats.items() if k != "lengths"}
    print("Token length distribution (no truncation):")
    print(printable)
    if length_stats["share_over_256"] > 0.05:
        print("More than 5% of inputs exceed 256 tokens. Consider MAX_LENGTH=512.")
    else:
        print("MAX_LENGTH=256 covers almost all inputs.")

    train_ds = SarcasmDataset(
        tokenize_texts(train_df["input_text"].tolist(), tokenizer, max_length),
        train_df["is_sarcastic"].to_numpy(),
    )
    val_ds = SarcasmDataset(
        tokenize_texts(val_df["input_text"].tolist(), tokenizer, max_length),
        val_df["is_sarcastic"].to_numpy(),
    )
    test_ds = SarcasmDataset(
        tokenize_texts(test_df["input_text"].tolist(), tokenizer, max_length),
        test_df["is_sarcastic"].to_numpy(),
    )

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = load_model(MODEL_NAME)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history: List[Dict[str, Any]] = []
    best_f1 = -1.0
    best_state = None
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = _run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, device, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "eval_loss": val_metrics["loss"],
                "eval_accuracy": val_metrics["accuracy"],
                "eval_f1": val_metrics["f1"],
            }
        )
        print(
            f"  train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best validation F1: {best_f1:.4f}")

    training_time = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    infer_start = time.perf_counter()
    y_true, y_pred, probs, test_loss = _predict_loader(model, test_loader, device)
    inference_time = time.perf_counter() - infer_start
    test_metrics = compute_metrics_dict(y_true, y_pred)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    meta = {
        "context_type": context_type,
        "base_model": MODEL_NAME,
        "max_length": max_length,
        "max_samples": max_samples,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "n_rows": int(len(frame)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "device": device,
        "gpu_name": gpu_name,
        "best_val_f1": best_f1,
        "label_map": {"0": "non_sarcastic", "1": "sarcastic"},
    }
    with (model_dir / "export_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = RESULTS_DIR / f"confusion_matrix_{context_type}.png"
    loss_path = RESULTS_DIR / f"loss_curve_{context_type}.png"
    if context_type == "headline_only":
        loss_path = RESULTS_DIR / "loss_curve.png"
    save_confusion_matrix(y_true, y_pred, cm_path, title=f"Confusion matrix ({context_type})")
    save_loss_curve(history, loss_path)

    report = classification_report_text(y_true, y_pred)
    (RESULTS_DIR / f"classification_report_{context_type}.txt").write_text(report, encoding="utf-8")

    error_rows = []
    for i in range(len(test_df)):
        error_rows.append(
            {
                "headline": test_df.iloc[i]["headline"],
                "context": test_df.iloc[i]["input_text"],
                "true_label": int(y_true[i]),
                "predicted_label": int(y_pred[i]),
                "sarcasm_probability": float(probs[i]),
                "error_type": error_type(int(y_true[i]), int(y_pred[i])),
                "context_type": context_type,
            }
        )
    save_error_analysis(error_rows, RESULTS_DIR / "error_analysis.csv")

    result = {
        "experiment": context_type,
        "context": context_type,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "training_loss": history[-1]["loss"] if history else float("nan"),
        "validation_loss": history[-1]["eval_loss"] if history else float("nan"),
        "test_loss": test_loss,
        "training_time_sec": round(training_time, 2),
        "inference_time_sec": round(inference_time, 2),
        "model_dir": str(model_dir),
        "n_rows": int(len(frame)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "device": device,
        "classification_report": report,
        "confusion_matrix_path": str(cm_path),
        "loss_curve_path": str(loss_path),
        "token_length_stats": printable,
        "history": history,
    }
    _append_experiment_row(result)
    print("\nTest metrics:")
    for key in ("accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1"):
        print(f"  {key}: {result[key]:.4f}")
    print(report)
    print(f"\nSaved model folder:\n  {model_dir}")
    print("Load it later with:")
    print(f'  tokenizer, model = load_saved_classifier(r"{model_dir}")')
    return result


def _append_experiment_row(result: Dict[str, Any]) -> None:
    path = RESULTS_DIR / "experiment_results.csv"
    keep = [
        "experiment",
        "context",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "weighted_f1",
        "training_loss",
        "validation_loss",
        "test_loss",
        "training_time_sec",
        "inference_time_sec",
        "n_rows",
        "device",
        "model_dir",
    ]
    row = {key: result.get(key) for key in keep}
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main() -> None:
    run_experiment(context_type="headline_only")


if __name__ == "__main__":
    main()
