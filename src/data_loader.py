"""
data_loader.py
──────────────
Loads JSON files in the PARSEME Subtask 2 format (trial or test).
No MWE extraction — the raw sentence is passed directly to the pipeline.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_record(raw: Dict[str, Any], lang_code: str) -> Optional[Dict]:
    """
    Parse one raw JSON record into a unified format.
    The sentence is taken from 'raw_text' (test) or 'text' (trial).
    No MWE extraction — the sentence is kept as is.
    """
    source_sent_id = raw.get("source_sent_id", "")
    # For test data: use raw_text; for trial data: use text (clean)
    raw_text = raw.get("raw_text", "")
    if not raw_text and "text" in raw:
        raw_text = raw["text"]

    # Reference paraphrases (trial only)
    ref_creative = None
    ref_minimal = None
    is_trial = "label" in raw
    if is_trial:
        for label_str in raw.get("label", []):
            label_str = label_str.strip()
            if label_str.lower().startswith("creative:"):
                ref_creative = label_str[len("creative:"):].strip()
            elif label_str.lower().startswith("minimal:"):
                ref_minimal = label_str[len("minimal:"):].strip()
    else:
        # For test data, minimal reference is the original sentence (used only if evaluating on test)
        ref_minimal = raw_text

    return {
        "id": raw.get("id", str(hash(raw_text))),  # fallback id
        "language": lang_code,
        "raw_text": raw_text,
        "sentence": raw_text,          # clean sentence (no markers)
        "is_trial": is_trial,
        "ref_creative": ref_creative,
        "ref_minimal": ref_minimal,
        "source_sent_id": source_sent_id,
    }

def load_file(filepath: str, lang_code: str) -> List[Dict]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, encoding="utf-8") as f:
        content = json.load(f)
    raw_records = content if isinstance(content, list) else [content]
    parsed = []
    for raw in raw_records:
        rec = parse_record(raw, lang_code)
        if rec:
            parsed.append(rec)
    logger.info(f"[Loader] {lang_code} | {filepath} → {len(parsed)} records")
    return parsed

def load_directory(dirpath: str, lang_code: str = None) -> List[Dict]:
    from config import LANGUAGES
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(f"Not a directory: {dirpath}")
    all_records = []
    json_files = sorted(f for f in os.listdir(dirpath) if f.endswith(".json"))
    for fname in json_files:
        lc = lang_code
        if not lc:
            prefix = fname.split("_")[0].upper()
            if prefix in LANGUAGES:
                lc = prefix
            else:
                logger.warning(f"Cannot infer language from '{fname}'. Use --lang. Skipping.")
                continue
        all_records.extend(load_file(os.path.join(dirpath, fname), lc))
    return all_records

def save_predictions(predictions: List[Dict], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"[Loader] Saved {len(predictions)} predictions → {filepath}")

def save_detailed_results(results: list, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    logger.info(f"[Loader] Saved detailed results → {filepath}")

def make_dummy_data() -> List[Dict]:
    """Create dummy records for testing (no MWE markers)."""
    dummy_sentences = [
        "She made up her mind to leave the company.",
        "He kicked the bucket after a long illness.",
        "Elle a fait son deuil de cette relation.",
    ]
    records = []
    for i, sent in enumerate(dummy_sentences):
        records.append({
            "id": str(i),
            "language": "EN" if i < 2 else "FR",
            "raw_text": sent,
            "sentence": sent,
            "is_trial": False,
            "ref_creative": None,
            "ref_minimal": sent,
            "source_sent_id": f"dummy_{i}",
        })
    return records