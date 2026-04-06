"""
prompts.py
──────────
Prompt templates for MWE detection and paraphrasing.
"""

DETECTION_SYSTEM = """\
You are an expert linguist. Given a sentence, identify the multiword expression (MWE) 
that is idiomatic (non‑compositional). Return the exact span of the MWE as it appears 
in the sentence, and list its lemmas (base forms, lowercase).

Output **only** a valid JSON object with keys: "mwe_span" (string or null) and "lemmas" (list of strings).
Example: {"mwe_span": "kicked the bucket", "lemmas": ["kick", "bucket"]}
If no MWE is present: {"mwe_span": null, "lemmas": []}
"""

DETECTION_USER = "Sentence: {sentence}\nMWE JSON:"

def build_detection_messages(sentence: str) -> list:
    return [
        {"role": "system", "content": DETECTION_SYSTEM},
        {"role": "user",   "content": DETECTION_USER.format(sentence=sentence)},
    ]

# Few-shot examples for paraphrasing
FEW_SHOT_EXAMPLES = [
    {
        "language"   : "French",
        "sentence"   : "Elle a fait son deuil de cette relation.",
        "mwe"        : "fait son deuil",
        "lemmas"     : ["faire", "deuil"],
        "paraphrase" : "Elle a accepté la perte de cette relation.",
    },
    {
        "language"   : "French",
        "sentence"   : "Le point de vue de la réalisatrice était clair.",
        "mwe"        : "point de vue",
        "lemmas"     : ["point", "vue"],
        "paraphrase" : "La perspective de la réalisatrice était claire.",
    },
    {
        "language"   : "English",
        "sentence"   : "She made up her mind to leave the company.",
        "mwe"        : "made up her mind",
        "lemmas"     : ["make", "mind"],
        "paraphrase" : "She decided to leave the company.",
    },
    {
        "language"   : "English",
        "sentence"   : "He kicked the bucket after a long illness.",
        "mwe"        : "kicked the bucket",
        "lemmas"     : ["kick", "bucket"],
        "paraphrase" : "He died after a long illness.",
    },
]

PARAPHRASE_SYSTEM = """\
You are an expert multilingual paraphrasing specialist.

You will be given a sentence containing an idiomatic expression (MWE),
the MWE itself, and a list of its component lemmas.

Your task: rewrite the sentence so that the MWE is removed and replaced
with a natural, literal equivalent that preserves the full meaning.

Rules you MUST follow:
1. Write the paraphrase in the SAME language as the input sentence.
2. Remove the MWE entirely — replace it with a literal paraphrase.
3. Preserve the COMPLETE meaning of the original sentence.
4. At least ONE of the listed forbidden lemmas must NOT appear in your output.
5. The output must be grammatically correct and fluent.
6. Output ONLY the paraphrased sentence — no explanation, no quotes, no extra text.
"""

PARAPHRASE_USER = """\
Language: {language}
Sentence: {sentence}
Idiomatic expression to remove: "{mwe}"
Forbidden lemmas (at least one must be absent from output): {lemmas}
Paraphrase:"""

RETRY_SYSTEM = """\
You are an expert multilingual paraphrasing specialist.

Your previous paraphrase still contained ALL of the original idiom's \
key words. You must try again more aggressively.

Rules you MUST follow:
1. Write the paraphrase in the SAME language as the input sentence.
2. The idiom "{mwe}" MUST be completely replaced.
3. You MUST NOT use ANY of these words in your output: {forbidden}.
4. Preserve the full meaning. Be grammatically correct and fluent.
5. Output ONLY the paraphrased sentence — no explanation.
"""

RETRY_USER = """\
Original sentence: {sentence}
Your failed paraphrase: {previous}
New paraphrase:"""

def build_paraphrase_messages(sentence: str, language: str, mwe: str, lemmas: list, use_few_shot: bool = True) -> list:
    messages = [{"role": "system", "content": PARAPHRASE_SYSTEM}]
    if use_few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": PARAPHRASE_USER.format(
                    language = ex["language"],
                    sentence = ex["sentence"],
                    mwe      = ex["mwe"],
                    lemmas   = str(ex["lemmas"]),
                ),
            })
            messages.append({"role": "assistant", "content": ex["paraphrase"]})
    messages.append({
        "role": "user",
        "content": PARAPHRASE_USER.format(
            language = language,
            sentence = sentence,
            mwe      = mwe,
            lemmas   = str(lemmas),
        ),
    })
    return messages

def build_retry_messages(sentence: str, mwe: str, lemmas: list, previous_output: str) -> list:
    forbidden = ", ".join(f'"{l}"' for l in lemmas)
    system = RETRY_SYSTEM.format(mwe=mwe, forbidden=forbidden)
    user   = RETRY_USER.format(sentence=sentence, previous=previous_output)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]