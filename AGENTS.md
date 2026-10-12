# AGENTS.md

Welcome to **Arg_Me** — a repository dedicated to exploring, analyzing, and building models or retrieval pipelines on the **args.me** debate dataset ([`webis/args_me`](https://huggingface.co/datasets/webis/args_me)).

---

## 1. Project Mission & Objectives

The primary focus of this project is exploratory data analysis (EDA), stance classification, and argument retrieval using the Webis args.me corpus.

Key exploration goals:
1. **Argument Mining & Stance Detection**: Classifying premises as `PRO` or `CON` relative to debate conclusions.
2. **Argument Quality & Structure**: Analyzing premise length, rhetorical devices, and debate contexts across platforms (*Debate.org*, *DebateWise*, *iDebate*, etc.).
3. **Google Colab Workflows**: Facilitating heavy computation (embeddings, fine-tuning, large-scale data sweeps) via Google Colab runtimes, the Google Colaboratory REST API, and exportable Jupyter notebooks.

---

## 2. Dataset Architecture & Critical Gotchas

### Dataset ID
- **Hugging Face Hub**: `webis/args_me`
- **Raw File**: `args-me.jsonl` (~845 MB, ~387,000+ arguments)
- **Direct Download URL**: `https://huggingface.co/datasets/webis/args_me/resolve/main/args-me.jsonl`

### CRITICAL: `datasets >= 3.0` Script Deprecation
> **Warning**: The `webis/args_me` repository contains a legacy `args_me.py` dataset loading script. In `datasets >= 3.0`, Hugging Face deprecated/blocked execution of dataset scripts, throwing `RuntimeError: Dataset scripts are no longer supported, but found args_me.py`.
>
> **Solution**: DO NOT call `load_dataset("webis/args_me")`. Instead, use the direct JSON loader in streaming or batch mode:
> ```python
> from datasets import load_dataset
>
> ds = load_dataset(
>     "json",
>     data_files="https://huggingface.co/datasets/webis/args_me/resolve/main/args-me.jsonl",
>     split="train",
>     streaming=True,  # Recommended: avoids downloading the full 845MB upfront
> )
> ```
> This is encapsulated in `src/dataset_loader.py` via `get_args_dataset()`.

### Record Schema
Each raw record in `args-me.jsonl` has the following schema:
```json
{
  "id": "c67482ba-2019-04-18T13:32:05Z-00000-000",
  "conclusion": "Contraceptive Forms for High School Students",
  "premises": [
    {
      "stance": "CON",
      "text": "My opponent forfeited every round..."
    }
  ],
  "context": {
    "sourceId": "c67482ba-2019-04-18T13:32:05Z",
    "previousArgumentInSourceId": "",
    "nextArgumentInSourceId": "...",
    "acquisitionTime": "2019-04-18T13:32:05Z",
    "discussionTitle": "Contraceptive Forms for High School Students",
    "sourceTitle": "Debate Argument: Contraceptive Forms for High School Students | Debate.org",
    "sourceUrl": "https://www.debate.org/debates/Contraceptive-Forms-for-High-School-Students/1/"
  }
}
```

---

## 3. Google Colab Integration

This repository is designed to operate seamlessly across local development and Google Colab environments:

### A. Google Colaboratory REST API (`colaboratory.googleapis.com`)
- Provides programmatic management of Colab runtimes and resources.
- Discovery REST URL: `https://colaboratory.googleapis.com/$discovery/rest?version=v1`
- Implemented in `src/colab_api.py`.
- **Authentication**:
  1. Local ADC: `gcloud auth application-default login`
  2. Service Account: Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`.
  3. Colab VM: Pre-authenticated default credentials via `google.auth.default()`.

### B. Notebooks for Colab (`notebooks/`)
- All notebooks in `notebooks/` feature runtime detection (`in_colab()`), automated dependency installs when run in Colab, and integration with Colab Secrets / Google Drive mounting.
- Notebook entry point: `notebooks/01_args_me_exploration.ipynb`.

---

## 4. Environment & Repository Structure

### Directory Layout
```
Arg_Me/
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules for data, venv, and secrets
├── AGENTS.md             # Guide for AI pair programmers & autonomous agents
├── README.md             # Project overview and quickstart
├── pyproject.toml        # Build configuration & dependencies
├── requirements.txt      # Pinned pip requirements
├── data/                 # Local data directory (ignored by git)
│   ├── .gitkeep
│   └── args_me_sample.parquet
├── notebooks/
│   └── 01_args_me_exploration.ipynb
└── src/
    ├── __init__.py
    ├── config.py         # Global settings & environment loader
    ├── colab_api.py      # Google Colab REST API wrapper & connectivity
    └── dataset_loader.py # args.me streaming, flattening, and sampling utilities
```

### Environment Setup

The repository uses Python 3.12 and `uv`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install / update dependencies
uv pip install -r requirements.txt
uv pip install -e .

# Launch Jupyter Lab
jupyter lab
```

### Environment Variables (`.env`)
Copy `.env.example` to `.env`:
- `HF_TOKEN`: Hugging Face User Access Token (optional for public dataset, recommended to prevent rate limits).
- `GOOGLE_CLOUD_PROJECT`: Google Cloud Project ID with Colaboratory API enabled.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to JSON credentials for GCP service accounts.

---

## 5. Agent Instructions & Operational Rules

1. **Keep large files out of git**:
   - `data/`, raw JSONL files, large Parquet outputs, and `.env` credentials must never be committed.
2. **Prioritize streaming for fast iterations**:
   - When exploring or testing new functions, stream small batches (`n_records=100` or `500`) instead of downloading the entire ~850MB JSONL file.
3. **Colab compatibility**:
   - Ensure scripts and notebooks can run without errors both in a local virtual environment and in Colab environments (`in_colab()` checks).
4. **Code Quality**:
   - Keep functions modular, type-annotated, and documented.
