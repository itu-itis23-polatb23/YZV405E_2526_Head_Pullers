# MWE Paraphrasing Pipeline
## Head Pullers — PARSEME 2.0 Subtask 2
### 2026 YZV405E Natural Language Processing Term Project
---

## Project Structure

```
mwe_pipeline/
├── src/
│   ├── config.py          ← API key, model name, language list
│   ├── prompts.py         ← All prompt templates (Stage 1 & 2)
│   ├── llm_client.py      ← Gemini API wrapper with retry logic
│   ├── lemmatizer.py      ← spaCy + Stanza lemmatizer for 14 languages
│   ├── pipeline.py        ← Core two-stage pipeline logic
│   ├── data_loader.py     ← Load/save PARSEME JSON format
│   └── evaluator.py       ← Masked BERT-score evaluation
│
├── run_pipeline.py        ← Main entry point (two-stage LLM)
├── train_mt5.py           ← Secondary model (mT5 fine-tuning)
├── scripts/
│   └── download_models.py ← Download spaCy/Stanza models (run once)
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

### 2. Download language models (one-time)
```bash
python scripts/download_models.py
```

### 3. Get a FREE Gemini API key (Google AI Studio)
```
1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API key"
4. Copy the key (starts with "AIza...")
```

```bash
export GEMINI_API_KEY="AIzaSy..."
```

Or edit `src/config.py` and paste it into `GEMINI_API_KEY = "..."`.

### 4. Download PARSEME trial data
```bash
git clone https://gitlab.com/parseme/sharedtask-data.git
cp -r sharedtask-data/2.0/subtask2/trial/ data/trial/
```

---

## Free Tier Rate Limits

| Model | Req/min | Req/day | Delay needed |
|---|---|---|---|
| `gemini-2.5-pro-preview-05-06` | 5 | 25 | 12s between calls |
| `gemini-2.0-flash` | 15 | 1500 | 4s between calls |

The default model is **Gemini 2.5 Pro** with 12s delay.
To switch to Flash (higher limits, faster, slightly lower quality):
```python
# src/config.py
MODEL_NAME        = "gemini-2.0-flash"
REQUEST_DELAY_SEC = 4
```

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

## Secondary Model: mT5

### Step 1 — Generate synthetic training data via Gemini:
```bash
python train_mt5.py --generate --data data/synthetic/
```

### Step 2 — Fine-tune mT5:
```bash
python train_mt5.py --train --data data/synthetic/ --model-dir outputs/mt5_finetuned/
```

### Step 3 — Run mT5 inference:
```bash
python train_mt5.py --predict \
    --model-dir outputs/mt5_finetuned/ \
    --input data/trial/fr_trial.json \
    --output outputs/mt5/
```

---

## Pipeline Architecture

```
INPUT: raw sentence (no MWE markup)
         │
         ▼
┌─────────────────────────────────┐
│  STAGE 1 — Idiom Detection      │
│                                 │
│  Gemini 2.5 Pro prompt:         │
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
│  Gemini 2.5 Pro prompt:         │
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

## Language — Lemmatizer Mapping

| Code | Language           | Lemmatizer |
|------|--------------------|------------|
| FR   | French             | spaCy      |
| EL   | Modern Greek       | spaCy      |
| JA   | Japanese           | spaCy      |
| PL   | Polish             | spaCy      |
| PT   | Brazilian Portug.  | spaCy      |
| RO   | Romanian           | spaCy      |
| SV   | Swedish            | spaCy      |
| UK   | Ukrainian          | spaCy      |
| SR   | Serbian            | spaCy      |
| KA   | Georgian           | Stanza     |
| HE   | Hebrew             | Stanza     |
| LV   | Latvian            | Stanza     |
| FA   | Persian            | Stanza     |
| SL   | Slovene            | Stanza     |
