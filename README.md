# MWE Paraphrasing Pipeline
## Head Pullers — PARSEME 2.0 Subtask 2

---

## Project Structure

```
mwe_pipeline/
├── src/
│   ├── config.py          ← API key, model name, language list
│   ├── prompts.py         ← All prompt templates (Stage 1 & 2)
│   ├── llm_client.py      ← Gemini API wrapper with retry logic
│   ├── lemmatizer.py      ← Regex tokenizer + constraint check (LLM provides lemmas)
│   ├── pipeline.py        ← Core two-stage pipeline logic
│   ├── data_loader.py     ← Load/save PARSEME JSON format
│   └── evaluator.py       ← Masked BERT-score evaluation
│
├── run_pipeline.py        ← Main entry point (two-stage LLM)
├── train_mt5.py           ← Secondary model (mT5 fine-tuning)
├── scripts/
│   └── download_models.py ← No-op stub (kept for backward compatibility)
├── data/
│   ├── trial/             ← Put PARSEME trial JSON files here
│   └── synthetic/         ← Auto-generated mT5 training data
├── outputs/               ← System predictions (Codabench format)
├── results/               ← Evaluation scores and analysis
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a FREE Gemini API key (Google AI Studio)
```
1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the key (starts with "AIza...")
```

```bash
export GEMINI_API_KEY="AIzaSy..."
```

Or edit `src/config.py` and add it to the `GEMINI_API_KEYS` list.

### 3. Download PARSEME trial data
```bash
git clone https://gitlab.com/parseme/sharedtask-data.git
cp -r sharedtask-data/2.0/subtask2/trial/ data/trial/
```

---

## Free Tier Rate Limits

The default model is **`gemini-3.1-flash-lite-preview`** with a 15s delay between calls, tuned for the free tier.

To change the model or delay, edit `src/config.py`:
```python
MODEL_NAME        = "gemini-3.1-flash-lite-preview"
REQUEST_DELAY_SEC = 15
```

> **API key rotation**: `src/attempt_counter.py` rotates through the `GEMINI_API_KEYS` list every 470 calls.
> Add multiple keys to `GEMINI_API_KEYS` in `src/config.py` to increase throughput.

---

## Running the Pipeline

### Quick test (no API needed — syntax only):
```bash
python run_pipeline.py --dry-run 3
```

### Run on trial data (all 14 languages):
```bash
python run_pipeline.py --input data/trial/ --output outputs/
```

### Run on a single language:
```bash
python run_pipeline.py --input data/trial/ --lang FR --output outputs/
```

### Pure zero-shot (no few-shot examples):
```bash
python run_pipeline.py --input data/trial/ --no-few-shot --output outputs/
```

### Run with evaluation (trial data has references):
```bash
python run_pipeline.py --input data/trial/ --output outputs/ --evaluate
```

---

## Secondary Model: mT5-large + LoRA

Architecture: `google/mt5-large` (1.2B parameters) with LoRA (rank=16, alpha=32, targets=[q,k,v,o,wi_0,wi_1,wo], ~0.2% trainable parameters).

Fine-tuning was done in **Google Colab** via `notebooks/finetune_mt5_lora.ipynb` (requires a GPU runtime — L4 or A100 recommended).

### Training data

Two datasets are merged and deduplicated before training.  
Upload both files to `MyDrive/mt5_mwe/data/` on Google Drive, then run the notebook's data-loading cell.

| File | Languages | Pairs | Source |
|---|---|---|---|
| `gemini_outputs.json` | 12 (no PT, SV) | ~491 | Wiktionary idioms + Gemini-generated paraphrases |
| `synthetic_data.json` | 14 | ~980 | Fully LLM-generated (Gemini Chat) |

Duplicates (same language + sentence) are removed, with `gemini_outputs.json` entries taking priority.  
Total merged: **~1,470 pairs across 14 languages**.

### Notebook workflow

| Cell | What it does |
|------|--------------|
| 1–2  | Install dependencies, mount Google Drive |
| 3    | Load + merge training data |
| 4    | Set hyperparameters (epochs, LoRA rank, LR, etc.) |
| 5    | Train/val stratified split (per language) |
| 6–7  | Tokenize, load mT5-large + apply LoRA |
| 8–9  | Trainer setup + training |
| 10   | Save LoRA adapter to `MyDrive/mt5_mwe/adapter_final/` |
| 11   | Inference smoke test on validation examples |
| 12   | Generate Codabench submissions for all 14 languages; zip to `MyDrive/mt5_mwe/submission.zip` |
| 13   | How to reload the adapter after a Colab runtime restart |

### Input format

- **FR**: `paraphrase <FR> [idiom: {mwe}]: {sentence}` — MWE extracted from `[[brackets]]` in test files exists, but was not used for any of the submissions after comparisons showed degraded performance. Legacy code was kept in the notebook, commented out.
- **Other 13 languages**: `paraphrase <LANG>: {sentence}` — no idiom hint available in test files, and none used for inference.

It should be noted that training data was in the given format with the idioms provided; but inference step does not include the "idiom" parameter.

> **Note:** `train_mt5.py` is an earlier prototype and was not used for the final submissions.

---

## Pipeline Architecture

```
INPUT: raw sentence (no MWE markup)
         │
         ▼
┌─────────────────────────────────┐
│  STAGE 1 — Idiom Detection      │
│                                 │
│  gemini-3.1-flash-lite-preview:         │
│  "Find the idiom in this        │
│   sentence. Reply with only     │
│   the idiom words."             │
│                                 │
│  Output: "made up her mind"     │
└────────────────┬────────────────┘
                 │
                 ▼
         Extract lemmas
         [make, mind]
                 │
                 ▼
┌─────────────────────────────────┐
│  STAGE 2 — Paraphrasing         │
│                                 │
│  gemini-3.1-flash-lite-preview:         │
│  "Rewrite removing 'made up     │
│   her mind'. Lemmas [make,      │
│   mind] must not all appear."   │
│                                 │
│  Output: "She decided to leave" │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  POST-PROCESSING CHECK          │
│                                 │
│  All of [make, mind] in output? │
│  YES → retry with stricter      │
│         prompt (max 3 retries)  │
│  NO  → constraint satisfied ✓   │
└─────────────────────────────────┘
         │
         ▼
OUTPUT: {"id": "fr_001", "paraphrase": "She decided to leave."}
```

---

## Evaluation (Masked BERT-score)

```
STEP 1 — MASK CHECK:
  Are ALL MWE lemmas still in the paraphrase?
  YES → score = 0.0  (automatic fail)
  NO  → proceed to step 2

STEP 2 — BERT-SCORE:
  score_min = bertscore(paraphrase, reference_minimal)
  score_cre = bertscore(paraphrase, reference_creative)
  final     = max(score_min, score_cre)
```

---

## Codabench Submission

The pipeline saves one file per language in `outputs/`:
```
outputs/
├── fr_predictions.json
├── ka_predictions.json
...
```

Each file:
```json
[
  {"id": "fr_001", "paraphrase": "Elle a accepté la perte de cette relation."},
  {"id": "fr_002", "paraphrase": "..."}
]
```

Zip the `outputs/` directory and upload to Codabench.

---

## Scoring with the Official Evaluator

`score_our_results.py` converts our pipeline outputs into the format
expected by the PARSEME scoring program (`scoring_program/evaluate.py`)
and runs it for every language.

### How it works

| Step | What happens |
|------|--------------|
| 1 | Reads `results/<LANG>_detailed_results.json` (our predictions) |
| 2 | Reads `data/<LANG>/test.blind.json` to get `source_sent_id` (joined on `raw_text`) |
| 3 | Writes `system_predictions/<LANG>_test.system.json` in the format `evaluate.py` expects |
| 4 | Calls `scoring_program/evaluate.py <gold> <system>` for each language |
| 5 | Saves `system_predictions/<LANG>/results.txt` (used by the averaging script) |

### Prerequisite — gold test files

The official **gold test files** (with `Creative` / `Minimal` reference labels) must
be placed at `data/<LANG>/test.json`. These are **not** the blind files included
in this repo — they are released by PARSEME after the evaluation period.

Without them you can still use `--convert-only` to prepare the submission format.

### Scoring program dependencies

```bash
pip install bert-score spacy regex
pip install git+https://github.com/estevelouis/WG4   # diversutils
python -m spacy download ja_core_news_sm
```

### Usage

```bash
# Convert predictions to system format only (no gold files needed):
python score_our_results.py --convert-only

# Score all 14 languages (gold files must exist at data/<LANG>/test.json):
python score_our_results.py

# Score a single language:
python score_our_results.py --lang EL

# Score all languages + compute the global macro-average:
python score_our_results.py --avg

# Use a different directory for gold files:
python score_our_results.py --gold-dir path/to/gold/
```

### Output

```
system_predictions/
├── EL_test.system.json   ← converted prediction file (evaluate.py input)
├── EL/
│   └── results.txt       ← raw output of evaluate.py (for averaging script)
├── FR_test.system.json
├── FR/
│   └── results.txt
...
├── scores.json           ← global scores (only with --avg)
└── scores.html           ← HTML report  (only with --avg)
```

---

## Self-Evaluation (no gold files needed)

`evaluate_self.py` evaluates our pipeline outputs directly from
`results/<LANG>_detailed_results.json` — **no gold reference files required**.

This is useful while the official gold test labels are not yet publicly available.

### Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| **Constr%** | % of MWE sentences where MWE tokens were removed | As high as possible |
| **Unchan%** | % of paraphrases identical to the original (model did nothing) | ~0% |
| **Err%** | % of sentences the pipeline could not process | ~0% |
| **Retries** | Average retries used per sentence | Low |
| **BERTf1%** | Semantic similarity: paraphrase ↔ original sentence (multilingual BERT) | High but < 100% |

> **BERTf1% note:** this is the *opposite* direction to the official metric.
> The official score compares our paraphrase against human references;
> this compares against the original sentence as a proxy for meaning preservation.
> A good paraphrase should score high (meaning kept) but not 100% (something changed).

### Usage

```bash
# Fast — constraint metrics only (no GPU, instant):
python evaluate_self.py --no-bert

# Full — with BERTScore (~2 min per language):
python evaluate_self.py

# Single language:
python evaluate_self.py --lang FR

# Save summary to JSON:
python evaluate_self.py --out results/self_eval.json

# Single language, full, save output:
python evaluate_self.py --lang EL --out results/EL_self_eval.json
```

### Sample output

```
  Lang  Total  No MWE  Constr%  Unchan%   Err%  Retries  BERTf1%      ±
  ──────────────────────────────────────────────────────────────────────
  EL      295       5   97.62%    0.00%  0.00%     0.12    88.41%   4.21
  FR       95       2   96.77%    1.08%  0.00%     0.05    91.23%   3.87
  ...
  ALL    1890      48   96.50%    0.42%  0.21%     0.09    89.10%
```

---

## Supported Languages

The pipeline covers all 14 PARSEME 2.0 Subtask 2 languages.
Lemmatization is performed by the LLM during Stage 1 (detection), not by spaCy or Stanza.
`src/lemmatizer.py` is a regex tokenizer used only for the post-generation constraint check.

| Code | Language           |
|------|--------------------|
| FR   | French             |
| EL   | Modern Greek       |
| JA   | Japanese           |
| PL   | Polish             |
| PT   | Brazilian Portuguese |
| RO   | Romanian           |
| SV   | Swedish            |
| UK   | Ukrainian          |
| SR   | Serbian            |
| KA   | Georgian           |
| HE   | Hebrew             |
| LV   | Latvian            |
| FA   | Persian            |
| SL   | Slovene            |
