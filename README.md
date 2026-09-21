# Contextual Sarcasm Detection in News Headlines

Reproduces the **News Headlines** experiment from Helal, Hassan, Badr, and Afify (2024), *A contextual-based approach for sarcasm detection*, [Scientific Reports](https://www.nature.com/articles/s41598-024-65217-8).

This repository is being built **incrementally**. Dataset inspection, preprocessing, and the **training notebook** are in place. Article scraping and the web demo come later.

---

## Paper methodology vs our implementation

### What the paper did

- Dataset: News Headlines sarcasm data, **26,709** records, columns `headline` and `article_link` (Table 1). Labels come from source: sarcastic headlines from The Onion, non-sarcastic from HuffPost.
- Context was **not** in the original file. The authors scraped article links for **author**, **article section**, and **description**.
- Ablation (Table 6, reported F1):
  - Headline only: 0.90
  - + author: 0.93
  - + article section: 0.99
  - + description: 0.99
- Best reported News Headlines result: RoBERTa-base, **99.7 F1** with contextual metadata (Tables 8–11).
- Model: RoBERTa tokenizer → RoBERTa-base (12 encoder layers) → classification head → CrossEntropyLoss.
- Training (Table 4): batch 32, lr `5e-5`, 5 epochs, eval batch 64, weight decay 0.01, 80/20 train/test, Google Colab T4 GPU.

### What we have implemented so far

- Inspected the actual JSON files on disk (not assumed CSV columns).
- Standardized the original 26,709-row file as the primary dataset.
- Light headline cleaning (HTML unescape, whitespace). No stemming, lemmatization, or stopword removal.
- Placeholder contextual columns (`author`, `section`, `description`) set to `unknown`, matching the paper's missing-value policy.
- Documented source/label leakage (see Risks below).

### What is adapted / not yet done

- Paper split: 80/20. We will also keep a validation split from the training portion for loss curves.
- Scraped context is not available yet; every context field is currently `unknown`.
- We will **not** copy the paper's 99.7% F1 into our result tables. Metrics will come from our own runs.

---

## Dataset inspection (Step 1 findings)

Primary file: `data/raw/Sarcasm_Headlines_Dataset.json`  
(JSON lines, not CSV.)

| Item | Result |
| --- | --- |
| Columns | `article_link`, `headline`, `is_sarcastic` |
| Rows | 26,709 |
| Non-sarcastic (0) | 14,985 (56.1%) |
| Sarcastic (1) | 11,724 (43.9%) |
| Missing / empty fields | none |
| Duplicate headlines | 107 extra rows (78 unique headlines); **no conflicting labels** |
| Duplicate article links | 1 exact duplicate row (`hillary clinton vs. herself`) |
| Headline length | 2–39 words, mean 9.85 |

A later **v2** file (`Sarcasm_Headlines_Dataset_v2.json`, 28,619 rows) is stored for reference but is **not** the paper's Table 1 dataset.

### Article-link accessibility (sample probe)

Live fetches of a small sample failed:

- Onion **subdomains** (`local.theonion.com`, `politics.theonion.com`): SSL hostname mismatch.
- HuffPost URLs: remote host closed the connection.
- 582 HuffPost records store **wrapped URLs** such as `https://www.huffingtonpost.comhttp://www.theguardian.com/...` (invalid host). Preprocessing extracts the inner URL.

Scraping the full set will need retries, subdomain canonicalization to `www.theonion.com`, HuffPost → HuffPost/Yahoo redirects, rate limits, and `unknown` fallbacks. We will not invent metadata.

---

## Important risk: source/domain leakage

The label is almost the publication:

| Source | Non-sarcastic | Sarcastic |
| --- | --- | --- |
| HuffPost | 14,984 | 0 |
| The Onion | 0* | 11,724 |

\*One HuffPost-wrapped Onion URL is labeled 0.

The paper's Table 2 also shows **Local** and **news** sections at **100% sarcastic**. Those sections likely encode Onion site structure (`local.theonion.com`, etc.). If scraping recovers section/domain too faithfully, a classifier can ignore the headline and still score near-perfect F1.

We therefore:

- Store `source_family` / `source_domain` **for analysis only**.
- Will not feed the domain into the model in the main experiments.
- Will document a follow-up run that strips source-identifying context if section/author remain source-specific.

This does **not** mean context is useless. Author and description can still be genuine cues. It does mean 99% F1 is not automatically evidence of sarcasm understanding.

---

## Proposed project structure

```
sarc/
├── data/
│   ├── raw/                 # original JSON
│   ├── processed/           # cleaned CSV + inspection report
│   └── contextual/          # scraped author/section/description (later)
├── src/
│   ├── utils.py             # paths, seeds, paper hyperparameters
│   ├── preprocessing.py     # Step 1 (this step)
│   ├── data_collection.py   # scraper (next)
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── error_analysis.py
├── notebooks/
│   └── experimentation.ipynb
├── results/
├── requirements.txt
└── README.md
```

## Pipeline (full, including later steps)

```
News Headlines JSON
        ↓
inspect + light preprocess
        ↓
scrape author / section / description (unknown if missing)
        ↓
build input text for one ablation setting
        ↓
RoBERTa tokenizer (max_length 256, trunc/pad)
        ↓
RoBERTa-base + classification head
        ↓
Accuracy / Precision / Recall / F1
        ↓
ablation + error analysis
```

---

## Train on Google Colab (recommended)

Your PC is CPU-only. The paper trained on a **Colab T4 GPU**. Use this notebook:

`notebooks/colab_train.ipynb`

1. Open [Google Colab](https://colab.research.google.com/)
2. **File → Upload notebook** → `colab_train.ipynb`
3. **Runtime → Change runtime type → T4 GPU**
4. **Runtime → Run all**

It downloads the News Headlines dataset itself. When training ends, Colab downloads `roberta_headline_only.zip`. Unzip that folder into `models/roberta_headline_only/` (already done if you used the zip from Telegram Desktop).

Optional: in the config cell set `SAVE_TO_DRIVE = True` and run the Drive-mount cell so the model survives if Colab disconnects.

---

## Website demo

After the model folder exists, start the local site:

```bash
python app.py
```

Open http://127.0.0.1:7872

The dashboard keeps the Headline Desk theme and adds **Analyze**, **Models**, **Compare**, and **Performance**. Available models:

- RoBERTa-base (your Colab checkpoint)
- Global Average Pooling (`model_1.keras`)
- Deep Global Average Pooling (`model_2.keras`)
- CNN (`model_3.keras`)
- CNN compact (`model_4.keras`)
- Bidirectional LSTM (`model_7.keras`)

---

## How to train and get the model files

1. Install packages:

```bash
pip install -r requirements.txt
```

2. Open `notebooks/experimentation.ipynb` and run all cells.

3. After training, the reusable model is the **folder**:

`models/roberta_headline_only/`

Copy that whole folder (weights + `config.json` + tokenizer files). That is what you load later.

The notebook defaults to a **CPU smoke test** (`MAX_SAMPLES = 2000`, `EPOCHS = 2`). For the paper-scale run set:

```python
MAX_SAMPLES = None
EPOCHS = 5
```

This environment currently has CPU-only PyTorch. A full 26k × 5-epoch run is much faster on a GPU (the paper used a Colab T4).

### Use the saved model later

```bash
python src/inference.py --model_dir models/roberta_headline_only
```

```python
from inference import predict_one
from model import load_saved_classifier

tokenizer, model = load_saved_classifier("models/roberta_headline_only")
print(predict_one("stock analysts confused, frightened by boar market"))
```

Without the notebook:

```bash
python src/train.py
```

---

## How to run Step 1 only

```bash
python src/preprocessing.py
```

Outputs:

- `data/processed/news_headlines_clean.csv`
- `data/processed/inspection_report.json`
- `results/class_distribution.png`
- `results/source_label_leakage.png`
