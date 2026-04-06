"""
train_mt5.py
────────────
Fine-tune mT5-base on synthetically generated MWE paraphrase pairs.

Steps
─────
1. Generate synthetic (idiomatic sentence, paraphrase) pairs using an LLM.
2. Fine-tune mT5-base with cross-entropy + label smoothing.
3. Save the fine-tuned model.
4. Run inference and apply the post-processing constraint check.

Usage
─────
    # Generate synthetic data then train:
    python train_mt5.py --generate --train

    # Train only (if synthetic data already exists):
    python train_mt5.py --train --data data/synthetic/

    # Inference only with a saved model:
    python train_mt5.py --predict --model-dir outputs/mt5_finetuned/ \
                        --input data/trial/ --output outputs/mt5/
"""

import sys
import os
import json
import logging
import argparse
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

logger = logging.getLogger(__name__)

# ── Idiom lists per language (seed list — extend from Wiktionary dumps) ───────
SEED_IDIOMS = {
    "FR": [
        ("casser sa pipe",   "mourir"),
        ("avoir le cafard",  "être déprimé"),
        ("poser un lapin",   "ne pas venir à un rendez-vous"),
        ("prendre la poudre d'escampette", "s'enfuir"),
        ("avoir le coup de foudre", "tomber amoureux instantanément"),
    ],
    "PL": [
        ("ktoś kopnął w kalendarz", "ktoś umarł"),
        ("mieć muchy w nosie",      "być w złym humorze"),
        ("lać wodę",                "mówić dużo bez treści"),
    ],
    "RO": [
        ("a da ortul popii",     "a muri"),
        ("a umbla cu cioara vopsită", "a înșela pe cineva"),
        ("a bate câmpii",        "a vorbi fără sens"),
    ],
    "PT": [
        ("bater as botas",     "morrer"),
        ("chutar o balde",     "desistir de tudo"),
        ("engolir sapos",      "aguentar situações difíceis"),
    ],
    "SV": [
        ("trilla av pinn",    "dö"),
        ("sätta dit någon",   "anklaga falskt"),
        ("ta i med hårdhandskarna", "agera hårt"),
    ],
    "EL": [
        ("τινάζω τα πέταλα",  "πεθαίνω"),
        ("τρώω τον κόσμο",    "ταλαιπωρούμαι πολύ"),
    ],
    "HE": [
        ("לאכול את הראש",  "להעיק"),
        ("לשבור את הראש",  "לחשוב קשה"),
    ],
    "JA": [
        ("首になる",       "解雇される"),
        ("骨を折る",       "努力する"),
        ("猫の手も借りたい", "非常に忙しい"),
    ],
    "KA": [
        ("ენა ჩაყლაპა",   "ჩუმად დარჩა"),
        ("გული გაუქვავდა", "გულგრილი გახდა"),
    ],
    "LV": [
        ("nolikt kājas",    "nomirt"),
        ("mest ērci ausī",  "traucēt"),
    ],
    "FA": [
        ("جان دادن",       "مردن"),
        ("دست و پا زدن",   "تلاش کردن"),
    ],
    "SR": [
        ("dati Bogu dušu",   "umreti"),
        ("biti u sedmom nebu", "biti veoma sretan"),
    ],
    "SL": [
        ("kopniti v boljši svet", "umreti"),
        ("hoditi kot bi pil",     "biti pijan"),
    ],
    "UK": [
        ("дати дуба",          "померти"),
        ("бити байдики",       "нічого не робити"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_pair(idiom: str, meaning: str,
                             language: str) -> dict:
    """
    Use the LLM to generate one (sentence, paraphrase) training pair.
    """
    from llm_client import call_llm
    from config     import LANGUAGES

    lang_name = LANGUAGES.get(language, language)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a {lang_name} language expert. "
                "Generate a natural sentence using the given idiom, "
                "then provide a literal paraphrase that removes the idiom "
                "but keeps the meaning. "
                "Reply ONLY with valid JSON: "
                '{"sentence": "...", "paraphrase": "..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Idiom: {idiom}\n"
                f"Meaning: {meaning}\n"
                f"Language: {lang_name}\n"
                f"JSON:"
            ),
        },
    ]

    response = call_llm(messages, max_tokens=200)
    if not response:
        return None

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        clean = response.strip().strip("```json").strip("```").strip()
        data  = json.loads(clean)
        if "sentence" in data and "paraphrase" in data:
            return {
                "id"          : f"synth_{language}_{idiom[:20]}",
                "language"    : language,
                "idiom"       : idiom,
                "sentence"    : data["sentence"],
                "paraphrase"  : data["paraphrase"],
            }
    except json.JSONDecodeError as e:
        logger.warning(f"[Synth] JSON parse failed for '{idiom}': {e}")
    return None


def generate_all_synthetic(output_dir: str = "data/synthetic",
                            n_per_idiom: int = 2) -> str:
    """
    Generate synthetic training pairs for all languages.
    Saves to output_dir/synthetic_<lang>.json.
    Returns path to combined file.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_pairs = []

    for lang, idioms in SEED_IDIOMS.items():
        lang_pairs = []
        logger.info(f"[Synth] Generating pairs for {lang} ({len(idioms)} idioms)...")

        for idiom, meaning in idioms:
            for _ in range(n_per_idiom):
                pair = generate_synthetic_pair(idiom, meaning, lang)
                if pair:
                    lang_pairs.append(pair)
                    logger.debug(f"  + {pair['sentence'][:60]!r}")

        out_path = os.path.join(output_dir, f"synthetic_{lang.lower()}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lang_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"[Synth] {lang}: {len(lang_pairs)} pairs saved → {out_path}")
        all_pairs.extend(lang_pairs)

    combined_path = os.path.join(output_dir, "synthetic_all.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"[Synth] Total: {len(all_pairs)} pairs → {combined_path}")
    return combined_path


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Fine-tune mT5
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic_dataset(data_path: str):
    """Load synthetic pairs as a HuggingFace Dataset."""
    from datasets import Dataset
    import json

    if os.path.isdir(data_path):
        all_records = []
        for jf in Path(data_path).glob("synthetic_*.json"):
            with open(jf, encoding="utf-8") as f:
                all_records.extend(json.load(f))
    else:
        with open(data_path, encoding="utf-8") as f:
            all_records = json.load(f)

    # Format: input = "paraphrase: <sentence>", target = "<paraphrase>"
    formatted = [
        {
            "input_text" : f"paraphrase MWE: {r['sentence']}",
            "target_text": r["paraphrase"],
        }
        for r in all_records
        if r.get("sentence") and r.get("paraphrase")
    ]
    random.shuffle(formatted)
    logger.info(f"[mT5] Loaded {len(formatted)} training examples")
    return Dataset.from_list(formatted)


def train_mt5(data_path   : str,
              model_dir   : str  = "outputs/mt5_finetuned",
              epochs       : int  = 3,
              batch_size   : int  = 4,
              learning_rate: float = 5e-5,
              max_input_len: int  = 128,
              max_target_len: int = 128) -> None:
    """
    Fine-tune mT5-base on the synthetic dataset.

    Training configuration:
      - Cross-entropy loss with label smoothing (epsilon=0.1)
      - AdamW optimizer
      - Linear learning rate schedule with warmup
    """
    try:
        from transformers import (
            MT5ForConditionalGeneration,
            T5Tokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
    except ImportError:
        logger.error(
            "transformers not installed. Run: "
            "pip install transformers torch --break-system-packages"
        )
        return

    MODEL_NAME = "google/mt5-base"
    logger.info(f"[mT5] Loading tokenizer and model from '{MODEL_NAME}'...")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Tokenize dataset
    dataset = load_synthetic_dataset(data_path)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=max_target_len,
                truncation=True,
                padding="max_length",
            )
        # Replace padding token id with -100 so it's ignored in loss
        label_ids = [
            [(lid if lid != tokenizer.pad_token_id else -100)
             for lid in l]
            for l in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True,
                            remove_columns=["input_text", "target_text"])

    # Train / eval split (90/10)
    split     = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds  = split["train"]
    eval_ds   = split["test"]

    collator  = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = model_dir,
        num_train_epochs             = epochs,
        per_device_train_batch_size  = batch_size,
        per_device_eval_batch_size   = batch_size,
        learning_rate                = learning_rate,
        weight_decay                 = 0.01,
        label_smoothing_factor       = 0.1,     # cross-entropy + label smoothing
        warmup_ratio                 = 0.1,
        evaluation_strategy          = "epoch",
        save_strategy                = "epoch",
        load_best_model_at_end       = True,
        predict_with_generate        = True,
        fp16                         = False,   # set True if GPU available
        logging_steps                = 50,
        report_to                    = "none",
    )

    trainer = Seq2SeqTrainer(
        model          = model,
        args           = training_args,
        train_dataset  = train_ds,
        eval_dataset   = eval_ds,
        tokenizer      = tokenizer,
        data_collator  = collator,
    )

    logger.info("[mT5] Starting fine-tuning...")
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"[mT5] Model saved → {model_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: mT5 inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_mt5(records    : list,
                model_dir  : str,
                output_dir : str = "outputs/mt5") -> list:
    """
    Run inference with a fine-tuned mT5 model + post-processing.
    """
    try:
        from transformers import MT5ForConditionalGeneration, T5Tokenizer
        import torch
    except ImportError:
        logger.error("transformers/torch not installed.")
        return []

    from lemmatizer import mwe_lemmas_from_span, check_constraint
    from config     import LANGUAGES, MAX_RETRIES

    logger.info(f"[mT5] Loading model from {model_dir}...")
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model     = MT5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    predictions = []

    for record in records:
        sent_id   = record.get("id", "unknown")
        sentence  = record.get("sentence", "")
        lang_code = record.get("language", "FR").upper()
        lang_name = LANGUAGES.get(lang_code, lang_code)

        # For mT5, we still need to know the MWE
        # Either it's in the record (trial data) or we detect it first
        mwe = record.get("mwe", None)
        if not mwe:
            # Fall back to LLM detection if mT5 doesn't know the MWE
            from pipeline import detect_mwe
            mwe = detect_mwe(sentence, lang_name)

        lemmas = mwe_lemmas_from_span(mwe, lang_code) if mwe else []

        input_text = f"paraphrase MWE: {sentence}"
        inputs = tokenizer(
            input_text, return_tensors="pt",
            max_length=128, truncation=True,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens  = 128,
                num_beams       = 4,
                early_stopping  = True,
            )

        paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-processing constraint check
        ok = check_constraint(paraphrase, lemmas, lang_code) if lemmas else True

        predictions.append({
            "id"                  : sent_id,
            "paraphrase"          : paraphrase,
            "constraint_satisfied": ok,
            "model"               : "mt5",
        })

        logger.info(
            f"[mT5] {lang_code} | {'✓' if ok else '✗'} | "
            f"{paraphrase[:70]!r}"
        )

    # Save
    from data_loader import save_predictions
    save_predictions(
        [{"id": p["id"], "paraphrase": p["paraphrase"]} for p in predictions],
        os.path.join(output_dir, "mt5_predictions.json"),
    )
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="mT5 fine-tuning for MWE paraphrasing"
    )
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic training data via LLM")
    parser.add_argument("--train",    action="store_true",
                        help="Fine-tune mT5 on synthetic data")
    parser.add_argument("--predict",  action="store_true",
                        help="Run mT5 inference on input data")
    parser.add_argument("--data",       default="data/synthetic",
                        help="Path to synthetic data dir or file")
    parser.add_argument("--model-dir",  default="outputs/mt5_finetuned",
                        help="Where to save/load the fine-tuned model")
    parser.add_argument("--input",      default=None,
                        help="Input data for prediction")
    parser.add_argument("--output",     default="outputs/mt5",
                        help="Output directory for predictions")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--n-per-idiom", type=int, default=2,
                        help="Synthetic pairs per idiom (default: 2)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    if args.generate:
        generate_all_synthetic(
            output_dir  = args.data,
            n_per_idiom = args.n_per_idiom,
        )

    if args.train:
        train_mt5(
            data_path    = args.data,
            model_dir    = args.model_dir,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            learning_rate= args.lr,
        )

    if args.predict:
        from data_loader import load_json, make_dummy_trial_data
        records = load_json(args.input) if args.input else make_dummy_trial_data()
        predict_mt5(records, model_dir=args.model_dir, output_dir=args.output)


if __name__ == "__main__":
    main()
