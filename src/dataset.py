"""Input construction, stratified splits, and RoBERTa tokenization."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    CONTEXT_TYPES,
    MAX_LENGTH,
    SEED,
    TEST_SIZE,
    UNKNOWN_TOKEN,
    VAL_SIZE_FROM_TRAIN,
)


def build_input_text(row: Mapping[str, Any], context_type: str) -> str:
    """
    Build the model input string for one ablation setting.

    Example (all_context):

        HEADLINE: Scientists discover surprising result
        AUTHOR: John Smith
        SECTION: Science
        DESCRIPTION: Researchers announced...
    """
    if context_type not in CONTEXT_TYPES:
        raise ValueError(
            f"Unknown context_type={context_type!r}. Choose from {CONTEXT_TYPES}."
        )

    headline = _field(row, "headline")
    author = _field(row, "author")
    section = _field(row, "section")
    description = _field(row, "description")

    lines = [f"HEADLINE: {headline}"]
    if context_type in ("author", "all_context"):
        lines.append(f"AUTHOR: {author}")
    if context_type in ("section", "all_context"):
        lines.append(f"SECTION: {section}")
    if context_type in ("description", "all_context"):
        lines.append(f"DESCRIPTION: {description}")
    return "\n".join(lines)


def _field(row: Mapping[str, Any], name: str) -> str:
    value = row[name] if name in row else None
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_TOKEN
    text = str(value).strip()
    return text if text else UNKNOWN_TOKEN


def add_input_text(frame: pd.DataFrame, context_type: str) -> pd.DataFrame:
    """Add an `input_text` column for the chosen context setting."""
    out = frame.copy()
    out["input_text"] = [
        build_input_text(row, context_type) for row in out.to_dict(orient="records")
    ]
    return out


def context_fields_are_populated(frame: pd.DataFrame) -> bool:
    """True when at least one author/section/description value is not 'unknown'."""
    for column in ("author", "section", "description"):
        if column not in frame.columns:
            continue
        values = frame[column].fillna(UNKNOWN_TOKEN).astype(str).str.strip().str.lower()
        if (values != UNKNOWN_TOKEN).any():
            return True
    return False


def stratified_splits(
    frame: pd.DataFrame,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    val_size_from_train: float = VAL_SIZE_FROM_TRAIN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Paper-style 80/20 train/test, then a validation split from training.

    With the defaults this is about:
    70% train / 10% validation / 20% test, stratified on is_sarcastic.
    """
    if "is_sarcastic" not in frame.columns:
        raise ValueError("DataFrame must contain is_sarcastic labels.")

    train_full, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["is_sarcastic"],
    )
    train_df, val_df = train_test_split(
        train_full,
        test_size=val_size_from_train,
        random_state=seed,
        stratify=train_full["is_sarcastic"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def maybe_sample(
    frame: pd.DataFrame,
    max_samples: Optional[int],
    seed: int = SEED,
) -> pd.DataFrame:
    """Optionally downsample for a CPU smoke test. None keeps the full set."""
    if max_samples is None or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    sampled, _ = train_test_split(
        frame,
        train_size=max_samples,
        random_state=seed,
        stratify=frame["is_sarcastic"],
    )
    return sampled.reset_index(drop=True)


def tokenize_texts(
    texts: Sequence[str],
    tokenizer,
    max_length: int = MAX_LENGTH,
):
    """Tokenize with truncation, padding, and attention masks."""
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )


def token_length_stats(texts: Sequence[str], tokenizer) -> Dict[str, Any]:
    """Measure token lengths *without* truncation so we can judge max_length."""
    lengths = [
        len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
        for text in texts
    ]
    series = pd.Series(lengths, dtype="int64")
    return {
        "n": int(len(series)),
        "min": int(series.min()) if len(series) else 0,
        "max": int(series.max()) if len(series) else 0,
        "mean": round(float(series.mean()), 2) if len(series) else 0.0,
        "p50": float(series.quantile(0.50)) if len(series) else 0.0,
        "p95": float(series.quantile(0.95)) if len(series) else 0.0,
        "p99": float(series.quantile(0.99)) if len(series) else 0.0,
        "share_over_128": round(float((series > 128).mean()), 4) if len(series) else 0.0,
        "share_over_256": round(float((series > 256).mean()), 4) if len(series) else 0.0,
        "share_over_512": round(float((series > 512).mean()), 4) if len(series) else 0.0,
        "lengths": lengths,
    }
