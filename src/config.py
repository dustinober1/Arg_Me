import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SRC_DIR = ROOT_DIR / "src"

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

# Hugging Face Settings
HF_TOKEN = os.getenv("HF_TOKEN", None)
ARGS_ME_DATASET_ID = "webis/args_me"
ARGS_ME_JSONL_URL = "https://huggingface.co/datasets/webis/args_me/resolve/main/args-me.jsonl"

# Google Colab API Settings
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", None)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)
COLAB_DISCOVERY_URL = os.getenv(
    "COLAB_DISCOVERY_URL",
    "https://colaboratory.googleapis.com/$discovery/rest?version=v1",
)
COLAB_LOCATION = os.getenv("COLAB_LOCATION", "us-central1")
