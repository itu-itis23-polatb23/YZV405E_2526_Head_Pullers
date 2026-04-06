"""
llm_client.py
─────────────
Gemini API wrapper using the official google-genai SDK.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None

global _attempts_on_this_session
_attempts_on_this_session = 0

def _get_client(api_key_index: int = 0):
    global _client
    if _client is None:
        from google import genai
        from config import GEMINI_API_KEYS
        _client = genai.Client(api_key=GEMINI_API_KEYS[api_key_index])
        logger.info("[LLM] Gemini client initialised")
    return _client

def call_llm(
    messages: list,
    model: str = None,
    temperature=None,
    max_tokens: int = None,
    max_attempts: int = 5,
    api_key_index: int = 0,
) -> Optional[str]:
    from google.genai import types
    from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS

    model = model or MODEL_NAME
    temperature = temperature if temperature is not None else TEMPERATURE
    max_tokens = max_tokens or MAX_TOKENS

    client = _get_client(api_key_index)
    system_text, gemini_contents = _convert_messages(messages)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_text or None,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=gemini_contents,
                config=config,
            )
            _attempts_on_this_session += 1
            text = response.text
            return _clean(text)
        except Exception as e:
            err = str(e)
            wait = 30
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                logger.warning(f"[LLM] Rate limit (attempt {attempt}/{max_attempts}). Waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err or "500" in err or "connection" in err.lower():
                logger.warning(f"[LLM] Server error (attempt {attempt}/{max_attempts}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"[LLM] Unexpected error: {e}")
                if attempt == max_attempts:
                    return None
                time.sleep(2)
    logger.error("[LLM] All retry attempts exhausted.")
    return None

def _convert_messages(messages: list):
    from google.genai import types
    system_parts = []
    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_parts.append(content)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        part = types.Part(text=content)
        if contents and contents[-1].role == gemini_role:
            contents[-1].parts.append(part)
        else:
            contents.append(types.Content(role=gemini_role, parts=[part]))
    system_text = "\n\n".join(system_parts) if system_parts else ""
    return system_text, contents

def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    if len(text) >= 2 and text[0] in ('"', "'", "\u201c") and text[-1] in ('"', "'", "\u201d"):
        text = text[1:-1].strip()
    for prefix in ("Paraphrase:", "paraphrase:", "Idiom:", "idiom:", "Output:", "output:", "Answer:", "answer:", "Rewrite:", "rewrite:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text