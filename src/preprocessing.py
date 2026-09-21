"""
STEP 1: Dataset inspection and light preprocessing.

This module does not scrape article pages and does not train a model.
It inspects the News Headlines JSON, cleans headline text conservatively,
repairs a known URL encoding issue, and writes a processed CSV plus
an inspection report.

Run from the project root:

    python src/preprocessing.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    DEFAULT_RAW_DATASET,
    INSPECTION_REPORT,
    PROCESSED_CSV,
    RESULTS_DIR,
    UNKNOWN_TOKEN,
    ensure_project_dirs,
)

WHITESPACE_RE = re.compile(r"\s+")
# Some HuffPost records store a wrapped URL such as:
#   https://www.huffingtonpost.comhttp://www.theguardian.com/...
WRAPPED_URL_RE = re.compile(
    r"^https?://www\.huffingtonpost\.com(?P<inner>https?://.+)$",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSON-lines News Headlines file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}\n"
            "Copy Sarcasm_Headlines_Dataset.json into data/raw/ first."
        )

    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc

    frame = pd.DataFrame(rows)
    expected = {"article_link", "headline", "is_sarcastic"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def clean_headline(text: Any) -> str:
    """
    Light cleaning suitable for RoBERTa.

    The paper applies stronger cleaning to *article descriptions*
    (non-English characters and location prefixes). Headlines are kept
    close to the original wording: HTML unescape, whitespace collapse,
    and empty values mapped to 'unknown'. No stemming, lemmatization,
    or stopword removal.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return UNKNOWN_TOKEN
    cleaned = html.unescape(str(text)).replace("\xa0", " ").strip()
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    return cleaned if cleaned else UNKNOWN_TOKEN


def repair_article_url(url: Any) -> str:
    """Extract the inner URL from HuffPost-wrapped links when present."""
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return UNKNOWN_TOKEN
    text = str(url).strip()
    if not text:
        return UNKNOWN_TOKEN
    match = WRAPPED_URL_RE.match(text)
    if match:
        return match.group("inner")
    return text


def source_family(url: str) -> str:
    """
    Map a URL to onion / huffpost / other.

    This is an analysis feature only. It must not be used as a model
    input unless an explicit leakage experiment is being run.
    """
    lowered = (url or "").lower()
    if "theonion.com" in lowered:
        return "onion"
    if "huffingtonpost.com" in lowered or "huffpost.com" in lowered:
        return "huffpost"
    return "other"


def source_domain(url: str) -> str:
    """Return the URL hostname in lowercase, or 'unknown'."""
    try:
        host = urlparse(url).netloc.lower()
        return host if host else UNKNOWN_TOKEN
    except Exception:
        return UNKNOWN_TOKEN


def fill_unknown(value: Any) -> str:
    """Replace missing or blank contextual fields with 'unknown'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNKNOWN_TOKEN
    text = str(value).strip()
    return text if text else UNKNOWN_TOKEN


def inspect_dataframe(frame: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Compute the inspection statistics requested before modeling."""
    n_rows = int(len(frame))
    label_counts = frame["is_sarcastic"].value_counts(dropna=False).to_dict()
    label_counts = {str(k): int(v) for k, v in label_counts.items()}
    label_props = {
        key: round(count / n_rows, 4) if n_rows else 0.0
        for key, count in label_counts.items()
    }

    missing = {col: int(frame[col].isna().sum()) for col in frame.columns}
    empty_strings = {}
    for col in frame.columns:
        if frame[col].dtype == object:
            empty_strings[col] = int((frame[col].astype(str).str.strip() == "").sum())

    n_dup_headlines = int(frame["headline"].duplicated().sum())
    n_dup_links = int(frame["article_link"].duplicated().sum())
    n_exact_dups = int(frame.duplicated().sum())
    unique_headline_link = int(
        frame[["headline", "article_link"]].drop_duplicates().shape[0]
    )

    conflicting = 0
    for _, group in frame.groupby("headline")["is_sarcastic"]:
        if group.nunique() > 1:
            conflicting += 1

    families = frame["source_family"].value_counts().to_dict()
    families = {str(k): int(v) for k, v in families.items()}

    leakage = {}
    for family, group in frame.groupby("source_family"):
        leakage[str(family)] = {
            str(label): int(count)
            for label, count in group["is_sarcastic"].value_counts().items()
        }

    domains = frame["source_domain"].value_counts().head(12).to_dict()
    domains = {str(k): int(v) for k, v in domains.items()}

    word_lens = frame["headline"].astype(str).str.split().str.len()
    char_lens = frame["headline"].astype(str).str.len()

    wrapped = int(
        frame["article_link"]
        .astype(str)
        .str.contains(r"huffingtonpost\.comhttps?://", case=False, regex=True)
        .sum()
    )

    return {
        "dataset_name": dataset_name,
        "n_rows": n_rows,
        "columns": list(frame.columns),
        "label_counts": label_counts,
        "label_proportions": label_props,
        "missing_values": missing,
        "empty_strings": empty_strings,
        "duplicate_headlines": n_dup_headlines,
        "duplicate_article_links": n_dup_links,
        "exact_duplicate_rows": n_exact_dups,
        "unique_headline_link_pairs": unique_headline_link,
        "duplicate_headlines_with_conflicting_labels": conflicting,
        "wrapped_huffpost_urls": wrapped,
        "source_family_counts": families,
        "source_family_by_label": leakage,
        "top_domains": domains,
        "headline_word_length": {
            "min": int(word_lens.min()) if n_rows else 0,
            "max": int(word_lens.max()) if n_rows else 0,
            "mean": round(float(word_lens.mean()), 2) if n_rows else 0.0,
        },
        "headline_char_length": {
            "min": int(char_lens.min()) if n_rows else 0,
            "max": int(char_lens.max()) if n_rows else 0,
            "mean": round(float(char_lens.mean()), 2) if n_rows else 0.0,
        },
    }


def preprocess_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean headlines, repair URLs, and add placeholder context columns."""
    out = frame.copy()
    out["headline"] = out["headline"].map(clean_headline)
    out["article_link_original"] = out["article_link"].astype(str)
    out["article_link"] = out["article_link"].map(repair_article_url)
    out["is_sarcastic"] = out["is_sarcastic"].astype(int)

    # Contextual fields are not in the original dataset. Fill with the
    # paper's missing-value placeholder until scraping populates them.
    if "author" not in out.columns:
        out["author"] = UNKNOWN_TOKEN
    else:
        out["author"] = out["author"].map(fill_unknown)
    if "section" not in out.columns:
        out["section"] = UNKNOWN_TOKEN
    else:
        out["section"] = out["section"].map(fill_unknown)
    if "description" not in out.columns:
        out["description"] = UNKNOWN_TOKEN
    else:
        out["description"] = out["description"].map(fill_unknown)

    out["source_family"] = out["article_link"].map(source_family)
    # If repair exposed an Onion URL that was wrapped by HuffPost,
    # classify from the repaired URL; keep original family too.
    out["source_family_original"] = out["article_link_original"].map(source_family)
    out["source_domain"] = out["article_link"].map(source_domain)

    before = len(out)
    out = out.drop_duplicates(
        subset=["headline", "article_link", "is_sarcastic"]
    ).reset_index(drop=True)
    out.attrs["n_dropped_exact_duplicates"] = before - len(out)
    return out


def save_inspection_plots(frame: pd.DataFrame, report: Dict[str, Any]) -> None:
    """Save class-balance and source-leakage figures for the README / notebook."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = frame["is_sarcastic"].value_counts().sort_index()
    labels = ["Non-sarcastic (0)", "Sarcastic (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4C78A8", "#F58518"])
    ax.set_ylabel("Number of headlines")
    ax.set_title("News Headlines class distribution")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    leakage = (
        frame.groupby(["source_family", "is_sarcastic"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    leakage.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_xlabel("Article source family")
    ax.set_ylabel("Number of headlines")
    ax.set_title("Source vs label (potential leakage)")
    ax.legend(["Non-sarcastic", "Sarcastic"])
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "source_label_leakage.png", dpi=150)
    plt.close(fig)

    report["figures"] = {
        "class_distribution": str(RESULTS_DIR / "class_distribution.png"),
        "source_label_leakage": str(RESULTS_DIR / "source_label_leakage.png"),
    }


def print_report(report: Dict[str, Any]) -> None:
    """Print a readable inspection summary."""
    print("=" * 72)
    print(f"Dataset: {report['dataset_name']}")
    print(f"Rows: {report['n_rows']:,}")
    print(f"Columns: {report['columns']}")
    print(f"Label counts: {report['label_counts']}")
    print(f"Label proportions: {report['label_proportions']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Empty strings: {report['empty_strings']}")
    print(f"Duplicate headlines: {report['duplicate_headlines']}")
    print(f"Duplicate article links: {report['duplicate_article_links']}")
    print(f"Exact duplicate rows: {report['exact_duplicate_rows']}")
    print(f"Unique headline+link pairs: {report['unique_headline_link_pairs']}")
    print(
        "Conflicting labels among duplicate headlines: "
        f"{report['duplicate_headlines_with_conflicting_labels']}"
    )
    print(f"Wrapped HuffPost URLs: {report['wrapped_huffpost_urls']}")
    print(f"Source families: {report['source_family_counts']}")
    print(f"Source vs label: {report['source_family_by_label']}")
    print(f"Top domains: {report['top_domains']}")
    print(f"Headline word length: {report['headline_word_length']}")
    print(f"Headline char length: {report['headline_char_length']}")
    print("=" * 72)


def run_inspection(
    raw_path: Optional[Path] = None,
    processed_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load, inspect, preprocess, and save the News Headlines dataset."""
    ensure_project_dirs()
    raw_path = Path(raw_path) if raw_path else DEFAULT_RAW_DATASET
    processed_path = Path(processed_path) if processed_path else PROCESSED_CSV

    raw = load_jsonl(raw_path)
    # Temporary source columns so inspection of the *raw* file is complete.
    raw_for_inspect = raw.copy()
    raw_for_inspect["source_family"] = raw_for_inspect["article_link"].map(source_family)
    raw_for_inspect["source_domain"] = raw_for_inspect["article_link"].map(source_domain)
    report = inspect_dataframe(raw_for_inspect, raw_path.name)

    processed = preprocess_dataframe(raw)
    report["n_rows_after_exact_dedup"] = int(len(processed))
    report["n_dropped_exact_duplicates"] = int(
        processed.attrs.get("n_dropped_exact_duplicates", 0)
    )
    report["processed_columns"] = list(processed.columns)
    report["notes"] = [
        "Primary file is Sarcasm_Headlines_Dataset.json (26,709 rows), matching paper Table 1.",
        "Author, section, and description are filled with 'unknown' until scraping.",
        "source_family and source_domain are analysis-only and must not be model inputs by default.",
        "Onion vs HuffPost is almost perfectly correlated with is_sarcastic.",
    ]

    save_inspection_plots(processed, report)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)

    INSPECTION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with INSPECTION_REPORT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print_report(report)
    print(f"Saved processed CSV: {processed_path}")
    print(f"Saved inspection report: {INSPECTION_REPORT}")
    print(f"Saved figures in: {RESULTS_DIR}")
    return processed


def main() -> None:
    run_inspection()


if __name__ == "__main__":
    main()
