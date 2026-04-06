"""
evaluator.py
────────────
Masked BERT-score evaluation (same as original, works with detection output).
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

def compute_bertscore(hypotheses: List[str], references: List[str], lang: str = "fr") -> List[float]:
    try:
        from bert_score import score as bs_score
        import warnings
        warnings.filterwarnings("ignore")
        model = "microsoft/deberta-xlarge-mnli" if lang == "en" else "bert-base-multilingual-cased"
        _, _, F1 = bs_score(hypotheses, references, model_type=model, lang=lang, verbose=False, device="cpu")
        return F1.tolist()
    except Exception as e:
        logger.error(f"[Eval] BERT-score failed: {e}")
        return [0.0] * len(hypotheses)

def masked_bertscore_single(paraphrase: str, ref_minimal: Optional[str], ref_creative: Optional[str], mwe_lemmas: List[str], lang_code: str) -> Tuple[float, str]:
    from lemmatizer import check_constraint
    if not check_constraint(paraphrase, mwe_lemmas, lang_code):
        return 0.0, "MASK_FAIL: all MWE lemmas still present"
    if not ref_minimal and not ref_creative:
        return 0.0, "NO_REFERENCE"
    scores = []
    reasons = []
    if ref_minimal:
        s = compute_bertscore([paraphrase], [ref_minimal], lang=lang_code.lower())[0]
        scores.append(s)
        reasons.append(f"minimal={s:.4f}")
    if ref_creative:
        s = compute_bertscore([paraphrase], [ref_creative], lang=lang_code.lower())[0]
        scores.append(s)
        reasons.append(f"creative={s:.4f}")
    final = max(scores)
    reason = "max(" + ", ".join(reasons) + ")"
    return final, reason

def evaluate_predictions(predictions: List[Dict], gold_records: List[Dict], results_dir: str = "results") -> Dict:
    pred_map = {str(p["id"]): p["paraphrase"] for p in predictions}
    rows = []
    for rec in gold_records:
        sent_id = str(rec["id"])
        lang_code = rec["language"]
        mwe_lemmas = rec.get("mwe_lemmas", [])
        ref_minimal = rec.get("ref_minimal")
        ref_creative = rec.get("ref_creative")
        paraphrase = pred_map.get(sent_id, "")
        if not paraphrase:
            rows.append({"id": sent_id, "language": lang_code, "score": 0.0, "reason": "MISSING_PREDICTION"})
            continue
        score, reason = masked_bertscore_single(paraphrase, ref_minimal, ref_creative, mwe_lemmas, lang_code)
        rows.append({
            "id": sent_id, "language": lang_code, "sentence": rec.get("sentence", ""),
            "mwe": rec.get("mwe", ""), "paraphrase": paraphrase,
            "ref_minimal": ref_minimal or "", "ref_creative": ref_creative or "",
            "score": round(score, 4), "reason": reason,
        })
    lang_scores = defaultdict(list)
    for row in rows:
        lang_scores[row["language"]].append(row["score"])
    per_lang = {}
    for lang, scores in sorted(lang_scores.items()):
        per_lang[lang] = {"n": len(scores), "avg": round(sum(scores)/len(scores), 4), "zeros": scores.count(0.0)}
    all_scores = [r["score"] for r in rows]
    global_avg = round(sum(all_scores)/len(all_scores), 4) if all_scores else 0.0
    zero_rate = round(all_scores.count(0.0)/len(all_scores), 4) if all_scores else 0.0
    summary = {
        "global_avg_masked_bertscore": global_avg,
        "zero_score_rate": zero_rate,
        "total_sentences": len(rows),
        "per_language": per_lang,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "per_sentence_scores.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"[Eval] Global masked BERT-score: {global_avg:.4f} | Zero rate: {zero_rate:.2%} | N={len(rows)}")
    return summary

def print_summary(summary: Dict) -> None:
    from config import LANGUAGES
    print("\n" + "=" * 62)
    print(f"  Global Masked BERT-score : {summary['global_avg_masked_bertscore']:.4f}")
    print(f"  Zero-score rate          : {summary['zero_score_rate']:.2%}")
    print(f"  Total sentences          : {summary['total_sentences']}")
    print("=" * 62)
    print(f"  {'Lang':<6} {'Name':<22} {'N':>4} {'Avg':>7} {'Zeros':>6}")
    print("  " + "-" * 48)
    for lang, s in sorted(summary["per_language"].items()):
        name = LANGUAGES.get(lang, lang)
        print(f"  {lang:<6} {name:<22} {s['n']:>4} {s['avg']:>7.4f} {s['zeros']:>6}")
    print("=" * 62 + "\n")