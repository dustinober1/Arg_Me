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
- **Repository Cloning Mandate (Notebooks `03` Forward)**: All Colab notebooks starting from `03` forward MUST automatically clone `https://github.com/dustinober1/Arg_Me.git` into `/content/Arg_Me` and set the working directory / `sys.path` when running inside Google Colab:
  ```python
  IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
  if IN_COLAB:
      !pip install -q datasets huggingface_hub pandas pyarrow scikit-learn torch transformers accelerate sentencepiece tiktoken
      if not os.path.exists("Arg_Me"):
          !git clone https://github.com/dustinober1/Arg_Me.git
      %cd Arg_Me
      import sys
      if "/content/Arg_Me" not in sys.path:
          sys.path.insert(0, "/content/Arg_Me")
  ```
- Notebook entry points: 
  - `notebooks/01_args_me_exploration.ipynb`
  - `notebooks/02_traditional_nlp_stance_prediction.ipynb`
  - `notebooks/03_transformer_cross_encoder_stance.ipynb`
  - `notebooks/04_tev1_decision_model_stance.ipynb`
  - `notebooks/05_dspy_tev1_advanced_pipeline.ipynb`
  - `notebooks/06_nimble_9b_decision_model_stance.ipynb`
  - `notebooks/07_nimble_9b_dspy_optimization.ipynb`
  - `notebooks/08_dspy_mipro_copro_optimization.ipynb`
  - `notebooks/09_overnight_large_scale_dspy_mipro.ipynb`
  - `notebooks/10_advanced_frontiers_args_me.ipynb`


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
├── report/
│   └── academic_report.md # Living academic-style paper of all project findings
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

## 5. Academic Report Mandate (`report/academic_report.md`)

All research, exploratory findings, architectural insights, and modeling experiments must be progressively documented in **[`report/academic_report.md`](file:///Users/dustinober/Projects/Data_Science/Arg_Me/report/academic_report.md)** as work unfolds.

### Standards & Guidelines:
1. **Academic Tone & Rigor**: Written as a formal research paper/technical report (Abstract, Introduction, Data Architecture & Ingestion, Empirical Findings & Metrics, Discussion, Limitations & Future Work, References).
2. **Continuous Updates as We Go**:
   - Whenever new empirical distributions, stance metrics, or token statistics are generated, update the empirical tables and figures in the report.
   - When new models, embeddings, or retrieval algorithms are benchmarked, record hypotheses, experimental setups, quantitative results, and qualitative error analyses.
   - Document technical challenges, algorithmic optimizations, and theoretical implications.
3. **Citations & References**: Maintain academic citations (e.g. Wachsmuth et al., Ajjour et al., CLEF/Argument Mining literature) in standard scholarly format.
4. **Reproducibility**: Explicitly record data subsets, random seeds, sample sizes, and hyperparameter configurations.

---

## 6. Agent Instructions & Operational Rules

1. **Continuous Academic Documentation**:
   - Every analytical step or empirical result MUST be synthesized and integrated into [`report/academic_report.md`](file:///Users/dustinober/Projects/Data_Science/Arg_Me/report/academic_report.md).
2. **Keep large files out of git**:
   - `data/`, raw JSONL files, large Parquet outputs, and `.env` credentials must never be committed.
3. **Prioritize streaming for fast iterations**:
   - When exploring or testing new functions, stream small batches (`n_records=100` or `500`) instead of downloading the entire ~850MB JSONL file.
4. **Colab compatibility**:
   - Ensure scripts and notebooks can run without errors both in a local virtual environment and in Colab environments (`in_colab()` checks).
5. **Code Quality**:
   - Keep functions modular, type-annotated, and documented.

