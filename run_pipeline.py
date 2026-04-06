"""
run_pipeline.py
───────────────
Main entry point for the MWE paraphrasing pipeline (detection + paraphrasing).
"""

import sys
import os
import time
import logging
import argparse
from collections import defaultdict
from pathlib import Path

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import LANGUAGES, OUTPUT_DIR, RESULTS_DIR, REQUEST_DELAY_SEC
from data_loader import load_file, load_directory, save_predictions, save_detailed_results, make_dummy_data
from pipeline import run_single, PipelineResult
from evaluator import evaluate_predictions, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_records(input_path: str, lang_code: str = None) -> list:
    p = Path(input_path)
    if p.is_file():
        if not lang_code:
            lang_code = p.stem.split("_")[0].upper()
            logger.info(f"Inferred language '{lang_code}' from filename.")
        records = load_file(str(p), lang_code)
    elif p.is_dir():
        records = load_directory(str(p), lang_code)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    if lang_code:
        before = len(records)
        records = [r for r in records if r["language"].upper() == lang_code.upper()]
        if len(records) < before:
            logger.info(f"Filtered to {len(records)} records for lang={lang_code}")
    return records

def run_all(records: list, use_few_shot: bool = True, dry_run_n: int = None, delay_sec: float = REQUEST_DELAY_SEC) -> list:
    if dry_run_n:
        records = records[:dry_run_n]
        logger.info(f"DRY RUN: first {dry_run_n} records only")
    results = []
    total = len(records)
    for i, record in enumerate(records, 1):
        lang = record.get("language", "??")
        logger.info(f"[{i}/{total}] Processing {lang} | id={record['id']}")
        result = run_single(record, use_few_shot=use_few_shot)
        results.append(result)
        status = "✓" if result.constraint_satisfied else "✗"
        print(f"  [{status}] [{lang}] MWE='{result.mwe}'\n"
              f"        original  : {result.sentence[:80]!r}\n"
              f"        paraphrase: {(result.paraphrase or '')[:80]!r}")
        if delay_sec > 0 and i < total:
            time.sleep(delay_sec)
    return results

def main():
    parser = argparse.ArgumentParser(description="MWE Paraphrasing Pipeline with Detection")
    parser.add_argument("--input", "-i", default=None, help="JSON file or directory")
    parser.add_argument("--lang", "-l", default=None, help="Two-letter language code (e.g. FR)")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help=f"Results directory (default: {RESULTS_DIR})")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot examples")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Evaluate against gold references (trial data only)")
    parser.add_argument("--dry-run", type=int, metavar="N", help="Process only first N records")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SEC, help=f"Seconds between API calls (default: {REQUEST_DELAY_SEC})")
    args = parser.parse_args()

    if args.input:
        records = load_records(args.input, args.lang)
    else:
        logger.info("No --input given. Using built-in dummy data.")
        records = make_dummy_data()
        if args.lang:
            records = [r for r in records if r["language"].upper() == args.lang.upper()]

    if not records:
        logger.error("No records found. Exiting.")
        sys.exit(1)

    logger.info(f"Loaded {len(records)} records | few_shot={not args.no_few_shot}")

    print("\n── Sample records ────────────────────────────────────────────")
    for r in records[:2]:
        print(f"  id: {r['id']}, lang: {r['language']}")
        print(f"  sentence: {r['sentence']!r}\n")

    results = run_all(records, use_few_shot=not args.no_few_shot, dry_run_n=args.dry_run, delay_sec=args.delay)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    by_lang = defaultdict(list)
    for res in results:
        by_lang[res.language_code].append({"id": res.sentence_id, "paraphrase": res.paraphrase or ""})
    for lang, preds in by_lang.items():
        out_path = os.path.join(args.output, f"{lang.lower()}_predictions.json")
        save_predictions(preds, out_path)

    all_preds = [{"id": r.sentence_id, "paraphrase": r.paraphrase or ""} for r in results]
    save_predictions(all_preds, os.path.join(args.output, "all_predictions.json"))

    detailed_path = os.path.join(args.results_dir, "detailed_results.json")
    if args.lang:
        detailed_path = os.path.join(args.results_dir, f"{args.lang}_detailed_results.json")
    save_detailed_results(results, detailed_path)

    total = len(results)
    satisfied = sum(1 for r in results if r.constraint_satisfied)
    errors = sum(1 for r in results if r.error)
    retries = sum(r.retries_used for r in results)

    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    print(f"  Total sentences     : {total}")
    print(f"  Constraint passed   : {satisfied}/{total} ({satisfied / total:.1%})")
    print(f"  Errors              : {errors}")
    print(f"  Total retries used  : {retries}")
    print("=" * 60)

    lang_stats = defaultdict(lambda: {"total": 0, "ok": 0})
    for r in results:
        lang_stats[r.language_code]["total"] += 1
        if r.constraint_satisfied:
            lang_stats[r.language_code]["ok"] += 1
    if len(lang_stats) > 1:
        print(f"\n  {'Lang':<6} {'OK':>4} / {'Total':>5}")
        print("  " + "-" * 20)
        for lang, s in sorted(lang_stats.items()):
            print(f"  {lang:<6} {s['ok']:>4} / {s['total']:>5}")
        print()

    if args.evaluate:
        gold = [r for r in records if r.get("ref_minimal") or r.get("ref_creative")]
        if not gold:
            logger.warning("No gold references found. Evaluation skipped.")
        else:
            summary = evaluate_predictions(all_preds, gold, args.results_dir)
            print_summary(summary)

    print(f"\nOutputs  → {args.output}/")
    print(f"Analysis → {args.results_dir}/\n")

if __name__ == "__main__":
    main()