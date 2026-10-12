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

- **Interactive Exploration**: Launch Jupyter to explore the dataset interactively:
  ```bash
  jupyter lab
  ```
  Open `notebooks/01_args_me_exploration.ipynb`.
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
