"""
scripts/download_models.py
──────────────────────────
Download all spaCy and Stanza language models needed for lemmatization.
Run this ONCE before running the pipeline.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --spacy-only
    python scripts/download_models.py --stanza-only
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import SPACY_MODELS, STANZA_MODELS


def download_spacy_models():
    print("\n── Downloading spaCy models ─────────────────────────────────")
    for lang_code, model_name in SPACY_MODELS.items():
        print(f"  [{lang_code}] {model_name} ...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("✓")
            else:
                print(f"✗  {result.stderr.strip()[:80]}")
        except Exception as e:
            print(f"✗  {e}")


def download_stanza_models():
    print("\n── Downloading Stanza models ────────────────────────────────")
    try:
        import stanza
    except ImportError:
        print("  stanza not installed. Run: pip install stanza")
        return

    for lang_code, stanza_lang in STANZA_MODELS.items():
        print(f"  [{lang_code}] stanza.download('{stanza_lang}') ...",
              end=" ", flush=True)
        try:
            stanza.download(stanza_lang, verbose=False)
            print("✓")
        except Exception as e:
            print(f"✗  {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download spaCy and Stanza models for all 14 languages"
    )
    parser.add_argument("--spacy-only",  action="store_true")
    parser.add_argument("--stanza-only", action="store_true")
    args = parser.parse_args()

    if not args.stanza_only:
        download_spacy_models()
    if not args.spacy_only:
        download_stanza_models()

    print("\nAll downloads complete. You can now run run_pipeline.py.\n")


if __name__ == "__main__":
    main()
