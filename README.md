# Arg_Me: args.me Dataset Exploration & Analysis

Exploration, stance classification, and argument mining on the **args.me** debate corpus ([`webis/args_me`](https://huggingface.co/datasets/webis/args_me)) from the Webis Group, with support for Google Colab runtimes and the Colab REST API.

---

## Quickstart

### 1. Environment Setup

Ensure you have Python 3.12+ and `uv` installed:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (if setting up fresh)
uv pip install -r requirements.txt
uv pip install -e .
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```
Fill in `.env`:
- `HF_TOKEN`: Hugging Face access token (optional, raises rate limits).
- `GOOGLE_CLOUD_PROJECT`: Google Cloud Project ID (if using Colab REST API).
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP credentials JSON.

### 3. Stream & Sample Data

Run the sample loader to stream 100 debate arguments and persist a local parquet sample:

```bash
python src/dataset_loader.py
```

### 4. Interactive Notebooks & Academic Report

- **Interactive Exploration & Modeling Pipelines**:
  - `notebooks/01_args_me_exploration.ipynb`: Corpus schema and lexical distribution.
  - `notebooks/02_traditional_nlp_stance_prediction.ipynb`: TF-IDF, sentiment, and linear models.
  - `notebooks/03_transformer_cross_encoder_stance.ipynb`: RoBERTa cross-encoder stance classifier.
  - `notebooks/04_tev1_decision_model_stance.ipynb`: Together AI Tev1:4B zero-shot decision benchmark.
  - `notebooks/05_dspy_tev1_advanced_pipeline.ipynb`: DSPy declarative reasoning pipeline.
  - `notebooks/06_nimble_9b_decision_model_stance.ipynb`: Bespoke Labs Nimble 9B decision benchmark.
  - `notebooks/07_nimble_9b_dspy_optimization.ipynb`: Few-shot contrastive bootstrap optimization.
  - `notebooks/08_dspy_mipro_copro_optimization.ipynb`: Dual-model Qwen-COPRO & Qwen-MIPROv2 optimizer.
  - `notebooks/09_overnight_large_scale_dspy_mipro.ipynb`: Overnight evaluation across $2,100$ debate arguments.
  - `notebooks/10_advanced_frontiers_args_me.ipynb`: Corpus hygiene, mixture-of-judges, and retrieval engine.
  - `notebooks/11_active_learning_hard_negative_mining.ipynb`: Active learning uncertainty screening & hard-negative curriculum mining.
  - `notebooks/12_qlora_qwen_stance_tuning.ipynb`: QLoRA parameter-efficient fine-tuning on Qwen 2.5 with hard-negative curriculum.
- **Academic Findings Report**: See [`report/academic_report.md`](report/academic_report.md) for the living, academic-style paper tracking all empirical findings, schema analyses, and model evaluations as research progresses.


---

## Google Colab Integration

1. **Running in Google Colab**:
   Open `notebooks/01_args_me_exploration.ipynb` in Colab. The notebook auto-detects Colab and handles necessary installs and drive mounts.
2. **Google Colab REST API**:
   Use `src/colab_api.py` to interact with `colaboratory.googleapis.com` for programmatic control over Colab runtimes.

---

## Dataset Details (`webis/args_me`)

- **Size**: ~845 MB (`args-me.jsonl`, ~387,000 arguments)
- **Features**: `id`, `conclusion`, `premises` (`text`, `stance`), `context` (`sourceTitle`, `discussionTitle`, `sourceUrl`, etc.)
- **Loading Note**: Uses direct JSONL streaming to avoid the legacy `args_me.py` script deprecated in `datasets >= 3.0`.
