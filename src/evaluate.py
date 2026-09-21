"""Metrics, confusion matrix, and error-analysis examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from utils import RESULTS_DIR

LABEL_NAMES = ["non_sarcastic", "sarcastic"]
PathLike = Union[str, Path]


def compute_metrics_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Accuracy, binary F1, plus macro/weighted F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def trainer_compute_metrics(eval_pred) -> Dict[str, float]:
    """Hugging Face Trainer callback."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics_dict(labels, preds)


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    path: PathLike,
    title: str = "Confusion matrix",
) -> Path:
    """Save a labeled 2x2 confusion-matrix figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-sarcastic", "Sarcastic"],
        yticklabels=["Non-sarcastic", "Sarcastic"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_loss_curve(history: Sequence[Mapping[str, Any]], path: PathLike) -> Path:
    """Plot training vs validation loss from Trainer log history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_loss = [row["loss"] for row in history if "loss" in row and "eval_loss" not in row]
    val_loss = [row["eval_loss"] for row in history if "eval_loss" in row]

    fig, ax = plt.subplots(figsize=(7, 4))
    if train_loss:
        ax.plot(range(1, len(train_loss) + 1), train_loss, marker="o", label="Training loss")
    if val_loss:
        ax.plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss vs validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def classification_report_text(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=["Non-sarcastic", "Sarcastic"],
        digits=4,
        zero_division=0,
    )


def save_error_analysis(
    records: List[Dict[str, Any]],
    path: PathLike = RESULTS_DIR / "error_analysis.csv",
    per_class: int = 8,
) -> pd.DataFrame:
    """
    Save TP / TN / FP / FN examples with headlines, context, and probability.
    """
    frame = pd.DataFrame(records)
    if frame.empty:
        frame.to_csv(path, index=False)
        return frame

    parts = []
    for bucket in ("TP", "TN", "FP", "FN"):
        subset = frame[frame["error_type"] == bucket]
        if subset.empty:
            continue
        subset = subset.sort_values("sarcasm_probability", ascending=False)
        parts.append(subset.head(per_class))
    out = pd.concat(parts, ignore_index=True) if parts else frame
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def error_type(true_label: int, pred_label: int) -> str:
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    return "FN"
