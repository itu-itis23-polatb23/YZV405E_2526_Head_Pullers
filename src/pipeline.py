"""
pipeline.py
───────────
Two‑stage pipeline:
  1. MWE detection using LLM (finds span and lemmas)
  2. Paraphrasing while avoiding those lemmas
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from config import LANGUAGES, MAX_RETRIES, DETECTION_TEMPERATURE, DETECTION_MAX_TOKENS
from llm_client import call_llm
from prompts import (
    build_paraphrase_messages,
    build_retry_messages,
    build_detection_messages,
)
from lemmatizer import check_constraint

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    sentence_id   : str
    language_code : str
    language_name : str
    raw_text      : str
    sentence      : str
    mwe           : str
    mwe_lemmas    : List[str]
    paraphrase    : Optional[str] = None
    constraint_satisfied: bool = False
    retries_used  : int = 0
    error         : Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.sentence_id,
            "language": self.language_code,
            "raw_text": self.raw_text,
            "sentence": self.sentence,
            "mwe": self.mwe,
            "mwe_lemmas": self.mwe_lemmas,
            "paraphrase": self.paraphrase or "",
            "constraint_satisfied": self.constraint_satisfied,
            "retries_used": self.retries_used,
            "error": self.error,
        }

def detect_mwe(sentence: str, lang_code: str, max_attempts: int = 3) -> tuple:
    """Call LLM to detect MWE span and lemmas. Returns (mwe_span, lemmas_list)."""
    messages = build_detection_messages(sentence)
    for attempt in range(1, max_attempts + 1):
        response = call_llm(
            messages,
            temperature=DETECTION_TEMPERATURE,
            max_tokens=DETECTION_MAX_TOKENS,
            max_attempts=1
        )
        if not response:
            continue
        text = response.strip()
        # Remove possible markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
            print("LLM Response: ", data)
            mwe_span = data.get("mwe_span")
            lemmas = data.get("lemmas", [])
            if not isinstance(lemmas, list):
                lemmas = []
            if mwe_span is None or not lemmas:
                logger.info(f"[Detection] No MWE found in: {sentence[:80]!r}")
                return None, []
            lemmas = [l.lower().strip() for l in lemmas if l.strip()]
            return mwe_span, lemmas
        except Exception as e:
            logger.warning(f"[Detection] Parse error attempt {attempt}: {e}")
    logger.error(f"[Detection] Failed after {max_attempts} attempts: {sentence[:100]}")
    return None, []

def run_single(record: dict, use_few_shot: bool = True) -> PipelineResult:
    lang_code = record["language"]
    lang_name = LANGUAGES.get(lang_code, lang_code)
    sentence = record["sentence"]

    # Step 1: Detect MWE
    mwe_span, lemmas = detect_mwe(sentence, lang_code)
    if not mwe_span:
        logger.warning(f"[{lang_code}] No MWE detected. Returning original sentence.")
        return PipelineResult(
            sentence_id=record["id"],
            language_code=lang_code,
            language_name=lang_name,
            raw_text=record.get("raw_text", ""),
            sentence=sentence,
            mwe="",
            mwe_lemmas=[],
            paraphrase=sentence,
            constraint_satisfied=True,
            error="No MWE detected",
        )

    result = PipelineResult(
        sentence_id=record["id"],
        language_code=lang_code,
        language_name=lang_name,
        raw_text=record.get("raw_text", ""),
        sentence=sentence,
        mwe=mwe_span,
        mwe_lemmas=lemmas,
    )

    logger.info(f"[{lang_code}] '{sentence[:60]}' | MWE='{mwe_span}' | lemmas={lemmas}")

    # Step 2: Paraphrase
    messages = build_paraphrase_messages(
        sentence=sentence,
        language=lang_name,
        mwe=mwe_span,
        lemmas=lemmas,
        use_few_shot=use_few_shot,
    )
    paraphrase = call_llm(messages)
    if not paraphrase:
        result.error = "LLM returned empty response"
        result.paraphrase = sentence
        return result

    # Constraint check + retries
    retries = 0
    while True:
        ok = check_constraint(paraphrase, lemmas, lang_code)
        if ok:
            logger.info(f"[{lang_code}] ✓ PASS (retries={retries}) → {paraphrase[:80]!r}")
            result.paraphrase = paraphrase
            result.constraint_satisfied = True
            result.retries_used = retries
            return result

        retries += 1
        if retries > MAX_RETRIES:
            break

        logger.warning(f"[{lang_code}] ✗ FAIL attempt {retries}/{MAX_RETRIES} — retrying with stricter prompt")
        retry_messages = build_retry_messages(sentence, mwe_span, lemmas, paraphrase)
        new_paraphrase = call_llm(retry_messages, temperature=0.7)
        if new_paraphrase:
            paraphrase = new_paraphrase

    result.paraphrase = paraphrase
    result.constraint_satisfied = False
    result.retries_used = retries
    result.error = f"Constraint not satisfied after {MAX_RETRIES} retries"
    return result