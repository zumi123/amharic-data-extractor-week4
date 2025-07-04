# Amharic E‑Commerce NER from Telegram

This repository is part of the **10 Academy AI Mastery – Week 4 challenge**.  
It collects raw Telegram messages from Ethiopian e‑commerce channels, cleans and tokenises the Amharic text, and outputs a CSV ready for Named Entity Recognition (NER) fine‑tuning.

---

## Folder Structure

```
amharic-ner-telegram/
├── data/
│   ├── raw/                 # Raw CSVs scraped from Telegram
│   └── cleaned/             # Pre‑processed, tokenised CSVs
├── scripts/
│   ├── fetch_telegram_data.py   # Scrape → CSV
│   └── preprocess_text.py       # Normalise + tokenise text
├── .env.example              # Template for your API secrets
├── requirements.txt          # Project dependencies
└── README.md                 # (you are here)
```

---

## Setup

1. **Clone** the repo and create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Telegram API credentials**  
   ```bash
   cp .env.example .env
   nano .env      # add your API_ID, API_HASH, PHONE_NUMBER
   ```

3. *(Optional but recommended)* **Install the Amharic segmenter for smarter tokenisation**  
   ```bash
   pip install git+https://github.com/fgaim/amseg.git
   ```
   If `amseg` is not installed, the tokenizer falls back to simple whitespace splitting.

---

## Usage

### 1⃣  Scrape Telegram channels

```bash
# Replace channel usernames (without @) and adjust limit as needed
python scripts/fetch_telegram_data.py --channels Shageronlinestore,AnotherChannel --limit 800
```
*Outputs:* `data/raw/raw_messages_YYYYMMDD_HHMMSS.csv`

### 2⃣  Preprocess & tokenise

```bash
python scripts/preprocess_text.py data/raw/raw_messages_YYYYMMDD_HHMMSS.csv \
                                 --output data/cleaned/processed_messages.csv
```
Columns in the output CSV:

| channel | message_id | timestamp | views | sender_id | clean_text | tokens |
|---------|------------|-----------|-------|-----------|------------|--------|

- **clean_text** – normalised Amharic string  
  (non‑Ethiopic punctuation removed, whitespace collapsed)  
- **tokens** – JSON list of tokens produced by `amseg` or by whitespace split.

---

* **Keep secrets out of Git** – your `.env` is ignored via `.gitignore`.

* **Large data files** in `data/` are ignored—commit only small samples when needed.

---
