"""
lemmatizer.py
─────────────
Simple tokenizer for constraint checking (no external lemmatizers).
MWE lemmas are provided by the LLM during detection.
"""

import re
from typing import List, Set

def tokenize_words(text: str) -> Set[str]:
    """Return a set of lowercase alphanumeric tokens."""
    if not text:
        return set()
    return set(re.findall(r"\w+", text.lower()))

def check_constraint(output_text: str, mwe_lemmas: List[str], lang_code: str = None) -> bool:
    """
    Return True if at least one lemma from mwe_lemmas is NOT present in output_text.
    lang_code is ignored.
    """
    if not mwe_lemmas:
        return True
    output_tokens = tokenize_words(output_text)
    return any(lemma not in output_tokens for lemma in mwe_lemmas)