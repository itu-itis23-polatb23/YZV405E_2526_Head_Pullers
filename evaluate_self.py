"""
evaluate_self.py
────────────────
Self-evaluation of our pipeline outputs without needing gold reference files.

Reads results/<LANG>_detailed_results.json for each language and computes:

  1. Constraint Satisfaction Rate  — % sentences where MWE was removed
     (already validated by the pipeline; stored in `constraint_satisfied`)

  2. Unchanged Rate                — % predictions identical to the original
     (the model just copied the sentence — worst possible output)

  3. Error Rate                    — % sentences the pipeline could not process

  4. BERTScore F1 (paraphrase ↔ original)
     Measures *semantic preservation*: how much meaning the paraphrase
     retained relative to the original sentence.
     Uses multilingual BERT so it works for all 14 languages.
     Ideal range: high enough to preserve meaning, but not 1.0
     (1.0 would mean the paraphrase is identical to the original).

  5. Average retries used          — proxy for how hard each language was

  NOTE: This is NOT the official PARSEME BERTScore (which compares against
  human Creative/Minimal references). It is a proxy metric to compare
  language-level quality and spot issues without gold files.

Usage:
  python evaluate_self.py                        # all available languages
  python evaluate_self.py --lang FR              # single language
  python evaluate_self.py --no-bert              # skip BERTScore (fast)
  python evaluate_self.py --out results/self_eval.json   # save JSON
"""

import argparse
import json
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

LANGUAGES = ["EL", "FA", "FR", "HE", "JA", "KA", "LV", "PL", "PT", "RO", "SL", "SR", "SV", "UK"]

# Mapping from our 2-letter codes to bert_score language codes.
# Using bert-base-multilingual-cased (mBERT) for all via model_type override,
# but providing lang codes for reference / auto-model fallback.
LANG_TO_BERT = {
    "EL": "el", "FA": "fa", "FR": "fr", "HE": "he",
    "JA": "ja", "KA": "ka", "LV": "lv", "PL": "pl",
    "PT": "pt", "RO": "ro", "SL": "sl", "SR": "sr",
    "SV": "sv", "UK": "uk",
}

# Multilingual model used for all languages so scores are comparable
MBERT_MODEL = "bert-base-multilingual-cased"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(lang: str) -> list[dict] | None:
    path = RESULTS_DIR / f"{lang.upper()}_detailed_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Per-language metrics ───────────────────────────────────────────────────────

def basic_metrics(records: list[dict]) -> dict:
    """Compute metrics that don't require bert_score."""
    total       = len(records)
    satisfied   = sum(1 for r in records if r.get("constraint_satisfied"))
    errors      = sum(1 for r in records if r.get("error") and r["error"] != "No MWE detected")
    no_mwe      = sum(1 for r in records if r.get("error") == "No MWE detected")
    unchanged   = sum(
        1 for r in records
        if r.get("paraphrase", "").strip() == r.get("raw_text", "").strip()
    )
    total_retries = sum(r.get("retries_used", 0) for r in records)

    # Only consider sentences that had a MWE to remove for the constraint rate
    scoreable = total - no_mwe
    constraint_rate = satisfied / scoreable if scoreable > 0 else 0.0
    unchanged_rate  = unchanged / total if total > 0 else 0.0
    error_rate      = errors   / total if total > 0 else 0.0
    avg_retries     = total_retries / total if total > 0 else 0.0

    return {
        "total":            total,
        "no_mwe":           no_mwe,
        "scoreable":        scoreable,
        "constraint_ok":    satisfied,
        "constraint_rate":  round(constraint_rate * 100, 2),
        "unchanged":        unchanged,
        "unchanged_rate":   round(unchanged_rate * 100, 2),
        "errors":           errors,
        "error_rate":       round(error_rate * 100, 2),
        "avg_retries":      round(avg_retries, 2),
    }


def bert_metrics(records: list[dict]) -> dict:
    """
    Compute BERTScore F1 between each paraphrase and its original sentence.
    Uses multilingual BERT so the same model handles all languages.
    """
    from bert_score import score as bert_score_fn

    predictions = []
    references  = []

    for r in records:
        pred = r.get("paraphrase", "").strip()
        ref  = r.get("raw_text",   "").strip()
        if pred and ref:
            predictions.append(pred)
            references.append(ref)

    if not predictions:
        return {"bert_f1_mean": None, "bert_f1_std": None}

    print(f"    Computing BERTScore for {len(predictions)} sentences …", flush=True)
    _, _, F1 = bert_score_fn(
        predictions,
        references,
        model_type=MBERT_MODEL,
        verbose=False,
    )

    f1_list = F1.tolist()
    mean_f1 = sum(f1_list) / len(f1_list)
    variance = sum((x - mean_f1) ** 2 for x in f1_list) / len(f1_list)
    std_f1  = variance ** 0.5

    return {
        "bert_f1_mean": round(mean_f1 * 100, 2),
        "bert_f1_std":  round(std_f1  * 100, 2),
    }


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(value, suffix="") -> str:
    if value is None:
        return "  n/a  "
    return f"{value:6.2f}{suffix}"


def print_summary(all_results: dict, include_bert: bool) -> None:
    print()
    print("=" * 90)
    print("  SELF-EVALUATION SUMMARY  (no gold references — proxy metrics only)")
    print("=" * 90)

    # Header
    if include_bert:
        print(f"  {'Lang':<5} {'Total':>6} {'No MWE':>7} {'Constr%':>8} {'Unchan%':>8} "
              f"{'Err%':>6} {'Retries':>8} {'BERTf1%':>8} {'±':>6}")
        print("  " + "─" * 86)
    else:
        print(f"  {'Lang':<5} {'Total':>6} {'No MWE':>7} {'Constr%':>8} {'Unchan%':>8} "
              f"{'Err%':>6} {'Retries':>8}")
        print("  " + "─" * 58)

    totals = {"total": 0, "no_mwe": 0, "scoreable": 0, "constraint_ok": 0,
              "unchanged": 0, "errors": 0, "total_retries": 0,
              "bert_sum": 0.0, "bert_count": 0}

    for lang, res in sorted(all_results.items()):
        m = res["basic"]
        b = res.get("bert", {})

        totals["total"]         += m["total"]
        totals["no_mwe"]        += m["no_mwe"]
        totals["scoreable"]     += m["scoreable"]
        totals["constraint_ok"] += m["constraint_ok"]
        totals["unchanged"]     += m["unchanged"]
        totals["errors"]        += m["errors"]
        totals["total_retries"] += m["avg_retries"] * m["total"]

        if include_bert and b.get("bert_f1_mean") is not None:
            totals["bert_sum"]   += b["bert_f1_mean"]
            totals["bert_count"] += 1

        row = (f"  {lang:<5} {m['total']:>6} {m['no_mwe']:>7} "
               f"{fmt(m['constraint_rate'], '%'):>8} "
               f"{fmt(m['unchanged_rate'],  '%'):>8} "
               f"{fmt(m['error_rate'],      '%'):>6} "
               f"{fmt(m['avg_retries']):>8}")
        if include_bert:
            bf = b.get("bert_f1_mean")
            bs = b.get("bert_f1_std")
            row += f" {fmt(bf, '%'):>8} {fmt(bs):>6}"
        print(row)

    # Totals row
    sc = totals["scoreable"]
    t  = totals["total"]
    print("  " + ("─" * 86 if include_bert else "─" * 58))
    constr = totals["constraint_ok"] / sc * 100 if sc else 0
    unch   = totals["unchanged"]     / t  * 100 if t  else 0
    err    = totals["errors"]        / t  * 100 if t  else 0
    avg_r  = totals["total_retries"] / t        if t  else 0
    row = (f"  {'ALL':<5} {t:>6} {totals['no_mwe']:>7} "
           f"{constr:>7.2f}% {unch:>7.2f}% {err:>5.2f}% {avg_r:>8.2f}")
    if include_bert and totals["bert_count"] > 0:
        avg_b = totals["bert_sum"] / totals["bert_count"]
        row += f" {avg_b:>7.2f}%"
    print(row)
    print("=" * 90)

    print()
    print("  Legend:")
    print("    No MWE    — sentences where the pipeline found no MWE (skipped constraint check)")
    print("    Constr%   — % of MWE sentences where MWE tokens were successfully removed")
    print("    Unchan%   — % sentences where paraphrase == original (no change at all)")
    print("    Err%      — % sentences that hit a pipeline error")
    print("    Retries   — avg number of retries per sentence")
    if include_bert:
        print("    BERTf1%   — semantic similarity paraphrase↔original (mBERT, higher=more preserved)")
        print("                Ideal: high (meaning kept) but < 100 (something did change)")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-evaluate pipeline outputs without gold reference files."
    )
    parser.add_argument(
        "--lang", "-l",
        default=None,
        metavar="LANG",
        help="Evaluate only this language (e.g. FR). Default: all available.",
    )
    parser.add_argument(
        "--no-bert",
        action="store_true",
        help="Skip BERTScore computation (fast mode — constraint metrics only).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="Save results as JSON to this file (e.g. results/self_eval.json).",
    )
    args = parser.parse_args()

    langs = [args.lang.upper()] if args.lang else LANGUAGES

    print("=" * 65)
    print("  Self-Evaluation (no gold files required)")
    print("=" * 65)
    print(f"  BERTScore : {'disabled (--no-bert)' if args.no_bert else f'enabled  (model: {MBERT_MODEL})'}")
    print("=" * 65)

    all_results = {}

    for lang in langs:
        print(f"\n── {lang} {'─' * (60 - len(lang))}")
        records = load_results(lang)
        if records is None:
            print(f"  [SKIP] No results file found: {RESULTS_DIR}/{lang}_detailed_results.json")
            continue

        print(f"  Loaded   : {len(records)} records")
        m = basic_metrics(records)
        print(f"  Constraint satisfied : {m['constraint_ok']}/{m['scoreable']} ({m['constraint_rate']}%)")
        print(f"  Unchanged (no edit)  : {m['unchanged']}  ({m['unchanged_rate']}%)")
        print(f"  Errors               : {m['errors']}  ({m['error_rate']}%)")
        print(f"  Avg retries          : {m['avg_retries']}")

        result = {"basic": m}

        if not args.no_bert:
            b = bert_metrics(records)
            result["bert"] = b
            if b["bert_f1_mean"] is not None:
                print(f"  BERTScore F1 (↔ original) : {b['bert_f1_mean']}% ± {b['bert_f1_std']}%")

        all_results[lang] = result

    if all_results:
        print_summary(all_results, include_bert=not args.no_bert)

    if args.out and all_results:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
