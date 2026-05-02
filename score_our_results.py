"""
score_our_results.py
────────────────────
Converts our pipeline outputs into the format expected by the PARSEME
scoring program (evaluate.py) and runs it for every language.

HOW IT WORKS
────────────
Our outputs/ directory contains files like el_predictions.json with:
    [{"id": <int_hash>, "paraphrase": "<text>"}, ...]

Our data/<LANG>/test.blind.json contains:
    [{"source_sent_id": "<string>", "raw_text": "<text>"}, ...]

The scoring program evaluate.py expects:
  - gold file:   [{"source_sent_id": ..., "text": ..., "raw_text": ..., "label": [...], ...}, ...]
  - system file: [{"source_sent_id": ..., "prediction": "<text>"}, ...]

This script:
  1. Builds a raw_text → source_sent_id map from the blind test file.
  2. Builds a raw_text → paraphrase map from our detailed_results.json.
  3. Joins them to produce a properly-formatted test.system.json.
  4. Calls evaluate.py (gold + system file) for each language.

REQUIREMENTS
────────────
  - The gold test files (test.json per language) must be placed under
    data/<LANG>/test.json   (they are NOT the blind files).
    These are released by the shared task organisers after the evaluation period.
  - The scoring_program dependencies must be installed:
      pip install bert-score spacy regex diversutils
      python -m spacy download ja_core_news_sm

USAGE
────────────
  # Score all 14 languages (gold files must exist):
  python score_our_results.py

  # Score a single language:
  python score_our_results.py --lang EL

  # Only convert predictions (no scoring — useful if gold not yet available):
  python score_our_results.py --convert-only

  # Use a custom gold directory:
  python score_our_results.py --gold-dir path/to/gold/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
SCORING_DIR = ROOT / "scoring_program"
EVALUATE_PY = SCORING_DIR / "evaluate.py"

# Where we write the converted system prediction files
SYSTEM_PRED_DIR = ROOT / "system_predictions"

LANGUAGES = ["EL", "FA", "FR", "HE", "JA", "KA", "LV", "PL", "PT", "RO", "SL", "SR", "SV", "UK"]
JA_LANG   = {"JA"}  # needs --is_JA flag in evaluate.py


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_raw_text_to_source_id(blind_path: Path) -> dict[str, str]:
    """Return {raw_text: source_sent_id} from test.blind.json."""
    entries = load_json(blind_path)
    mapping = {}
    for e in entries:
        raw = e["raw_text"].strip()
        if raw in mapping:
            print(f"  [WARN] Duplicate raw_text in blind file, keeping first: {raw[:60]!r}")
        else:
            mapping[raw] = e["source_sent_id"]
    return mapping


def build_raw_text_to_paraphrase(detailed_path: Path) -> dict[str, str]:
    """Return {raw_text: paraphrase} from <LANG>_detailed_results.json."""
    entries = load_json(detailed_path)
    mapping = {}
    for e in entries:
        raw = e.get("raw_text", "").strip()
        para = e.get("paraphrase", "").strip()
        if not raw:
            continue
        if raw in mapping:
            print(f"  [WARN] Duplicate raw_text in detailed results, keeping first: {raw[:60]!r}")
        else:
            mapping[raw] = para
    return mapping


def convert_predictions(lang: str, gold_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Build the system prediction file for `lang` in evaluate.py format.

    Returns (gold_path, system_pred_path) — gold_path is None if not found.
    """
    lang_upper = lang.upper()
    lang_lower = lang.lower()

    # ── 1. Locate our detailed results ────────────────────────────────────────
    detailed_path = RESULTS_DIR / f"{lang_upper}_detailed_results.json"
    if not detailed_path.exists():
        print(f"  [SKIP] No detailed results found: {detailed_path}")
        return None, None

    # ── 2. Locate the blind test file (for source_sent_id lookup) ─────────────
    blind_path = DATA_DIR / lang_upper / "test.blind.json"
    if not blind_path.exists():
        print(f"  [SKIP] No blind test file: {blind_path}")
        return None, None

    # ── 3. Build lookup maps ───────────────────────────────────────────────────
    raw_to_source_id  = build_raw_text_to_source_id(blind_path)
    raw_to_paraphrase = build_raw_text_to_paraphrase(detailed_path)

    # ── 4. Join maps → system prediction list ─────────────────────────────────
    system_preds = []
    unmatched = 0

    for raw_text, paraphrase in raw_to_paraphrase.items():
        source_id = raw_to_source_id.get(raw_text)
        if source_id is None:
            # Try a slightly normalised match (strip extra spaces)
            source_id = raw_to_source_id.get(" ".join(raw_text.split()))
        if source_id is None:
            unmatched += 1
            continue
        system_preds.append({
            "source_sent_id": source_id,
            "prediction":     paraphrase,
        })

    print(f"  Matched  : {len(system_preds)} sentences")
    if unmatched:
        print(f"  [WARN] Unmatched in blind file: {unmatched} sentences")

    # ── 5. Save system prediction file in Codabench submission layout ──────────
    #       Required format: system_predictions/<LANG>/test.system.json
    system_pred_path = SYSTEM_PRED_DIR / lang_upper / "test.system.json"
    save_json(system_preds, system_pred_path)
    print(f"  Saved system predictions → {system_pred_path}")

    # ── 6. Locate gold test file ───────────────────────────────────────────────
    gold_path = gold_dir / lang_upper / "test.json"
    if not gold_path.exists():
        print(f"  [INFO] Gold file not found: {gold_path}")
        print(f"         Cannot run evaluate.py without the gold file.")
        gold_path = None

    return gold_path, system_pred_path


def run_evaluate(lang: str, gold_path: Path, system_pred_path: Path) -> str | None:
    """
    Run evaluate.py for one language.
    Returns the stdout output (str) or None on failure.
    """
    cmd = [
        sys.executable,
        str(EVALUATE_PY),
        str(gold_path),
        str(system_pred_path),
    ]
    if lang.upper() in JA_LANG:
        cmd.append("--is_JA")

    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            print(f"  [ERROR] evaluate.py exited with code {result.returncode}")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            return None
        return result.stdout
    except Exception as exc:
        print(f"  [ERROR] Failed to run evaluate.py: {exc}")
        return None


def save_results_txt(lang: str, output: str) -> Path:
    """Save evaluate.py stdout to <LANG>/results.txt for average_of_paraphrase_evaluations.py."""
    results_txt_dir = SYSTEM_PRED_DIR / lang.upper()
    results_txt_dir.mkdir(parents=True, exist_ok=True)
    results_txt = results_txt_dir / "results.txt"
    results_txt.write_text(output, encoding="utf-8")
    return results_txt


def create_submission_zip(converted_langs: list[str]) -> Path:
    """
    Create submission.zip in the project root with the Codabench-required layout:
        submission.zip
        ├── EL/
        │   └── test.system.json
        ├── FR/
        │   └── test.system.json
        └── ...
    """
    import zipfile
    zip_path = ROOT / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for lang in converted_langs:
            src = SYSTEM_PRED_DIR / lang.upper() / "test.system.json"
            if src.exists():
                # Archive path inside the zip: <LANG>/test.system.json
                zf.write(src, arcname=f"{lang.upper()}/test.system.json")
    return zip_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert pipeline outputs and score them with evaluate.py"
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        metavar="LANG",
        help="Score only this language (e.g. EL). Default: all 14 languages.",
    )
    parser.add_argument(
        "--gold-dir",
        default=str(DATA_DIR),
        metavar="DIR",
        help=(
            f"Directory containing <LANG>/test.json gold files. "
            f"Default: {DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert predictions to system format; do not run evaluate.py.",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help=(
            "After scoring all languages, also run average_of_paraphrase_evaluations.py "
            "to compute the global score."
        ),
    )
    args = parser.parse_args()

    gold_dir = Path(args.gold_dir)
    langs    = [args.lang.upper()] if args.lang else LANGUAGES

    SYSTEM_PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  PARSEME 2.0 – Scoring Pipeline")
    print("=" * 65)
    print(f"  Languages      : {', '.join(langs)}")
    print(f"  Gold dir       : {gold_dir}")
    print(f"  Convert only   : {args.convert_only}")
    print(f"  System preds → : {SYSTEM_PRED_DIR}")
    print("=" * 65)

    scored_langs   = []
    converted_langs = []

    for lang in langs:
        print(f"\n── {lang} {'─' * (60 - len(lang))}")

        gold_path, system_pred_path = convert_predictions(lang, gold_dir)
        if system_pred_path is None:
            continue

        converted_langs.append(lang)

        if args.convert_only:
            continue

        if gold_path is None:
            print(f"  Skipping evaluate.py (no gold file).")
            continue

        output = run_evaluate(lang, gold_path, system_pred_path)
        if output is None:
            continue

        print(output)
        results_txt = save_results_txt(lang, output)
        print(f"  Saved results.txt → {results_txt}")
        scored_langs.append(lang)

    # ── Create submission.zip after conversion ─────────────────────────────────
    if converted_langs:
        zip_path = create_submission_zip(converted_langs)
        print(f"\n  submission.zip → {zip_path}")
        print(f"  Contains {len(converted_langs)} language(s): {', '.join(converted_langs)}")

    # ── Optional: run average_of_paraphrase_evaluations.py ────────────────────
    if args.avg and scored_langs and not args.convert_only:
        print("\n" + "=" * 65)
        print("  Running macro-average across all scored languages")
        print("=" * 65)

        avg_script = SCORING_DIR / "average_of_paraphrase_evaluations.py"
        scores_json = SYSTEM_PRED_DIR / "scores.json"
        scores_html = SYSTEM_PRED_DIR / "scores.html"

        nb_langs = len(LANGUAGES)  # total competition languages = 14
        cmd = [
            sys.executable,
            str(avg_script),
            str(nb_langs),
            str(SYSTEM_PRED_DIR),
            str(scores_json),
            str(scores_html),
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(result.stdout)
        if result.returncode != 0:
            print("[ERROR]", result.stderr[-1000:])
        else:
            print(f"  Global scores → {scores_json}")
            print(f"  HTML report   → {scores_html}")

    print("\n" + "=" * 65)
    print(f"  Done. Scored {len(scored_langs)} language(s): {', '.join(scored_langs) or 'none'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
