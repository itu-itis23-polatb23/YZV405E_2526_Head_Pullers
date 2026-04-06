"""
config.py
─────────
Central configuration for the MWE paraphrasing pipeline.

API key setup (Google AI Studio — free):
  1. Go to https://aistudio.google.com/apikey
  2. Click "Create API key"
  3. Export:  export GEMINI_API_KEY="AIza..."
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.1-flash-lite-preview"
TEMPERATURE = 0.3
MAX_TOKENS = 512
MAX_RETRIES = 3
REQUEST_DELAY_SEC = 15

# ── Languages covered in Subtask 2 ───────────────────────────────────────────
LANGUAGES = {
    "FR": "French",
    "KA": "Georgian",
    "EL": "Modern Greek",
    "JA": "Japanese",
    "HE": "Hebrew",
    "LV": "Latvian",
    "FA": "Persian",
    "PL": "Polish",
    "PT": "Brazilian Portuguese",
    "RO": "Romanian",
    "SR": "Serbian",
    "SL": "Slovene",
    "SV": "Swedish",
    "UK": "Ukrainian",
}

# ── MWE detection (LLM based) ────────────────────────────────────────────────
DETECTION_TEMPERATURE = 0.2
DETECTION_MAX_TOKENS = 256

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
RESULTS_DIR = "results"
