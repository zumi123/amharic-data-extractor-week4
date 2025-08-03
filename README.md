# Amharic Telegram Named Entity Recognition (NER)

This project extracts and annotates Amharic Telegram messages for Named Entity Recognition (NER). It includes preprocessing, labeling in CoNLL format, and preparation for model training.

---

## Project Structure

```
amharic-data-extractor-week4/
├── data/
│   ├── cleaned/                    # Preprocessed text and labeled subset
│   │   ├── processed_messages.csv  # Tokenized and cleaned messages
│   │   └── amharic_ner_subset.conll  # 50 manually labeled messages in CoNLL format
│   ├── labeled/ 
│   │   └── amharic_ner_subset.conll  # 50 manually labeled messages in CoNLL format                      
│   ├── raw/                        # Raw scraped Telegram messages
├── notebooks/
├── scripts/
└── README.md
```

---

## Tasks Overview

### Task 1 – Preprocess Telegram Messages
- Tokenize, normalize, and clean Amharic text
- Separate metadata (sender, timestamp, etc.)
- Output stored in `data/cleaned/processed_messages.csv`

Run:
```bash
python scripts/preprocess_text.py data/raw/raw_messages_*.csv
```

---

### Task 2 – Label 50 Messages (CoNLL Format)
- Annotate `Product`, `Price`, and `Location` entities
- BIO tagging scheme used:
  - `B-Product`, `I-Product`
  - `B-PRICE`, `I-PRICE`
  - `B-LOC`, `I-LOC`
  - `O` for non-entities

Sample format:
```
ፍሪጅ   B-Product
ቂቤ    I-Product
1000   B-PRICE
ብር    I-PRICE
አዲስ   B-LOC
አበባ   I-LOC
```

Use the template generator:
```bash
python scripts/create_conll_template.py data/cleaned/processed_messages.csv --n 50
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

> Optional: For improved Amharic tokenization, install [`amseg`](https://github.com/fgaim/amseg):

```bash
pip install amseg
```

---


## Task 3 – Model Training
We fine-tuned transformer-based models for Amharic NER using the labeled CoNLL dataset.

### Models Trained
- `xlm-roberta-base` (baseline multilingual model)
- `afro-xlmr-base` (optimized for African languages)
- `xlm-roberta-base-finetuned-amharic` (already fine-tuned on Amharic NER)

Training was done with Hugging Face `Trainer`. 

**Example training command:**
```bash
python scripts/train_afro-xlmr-base.py
```

---

## Task 4 – Model Comparison
### Training Results

| Model                            | Training Loss | Runtime | Notes |
|----------------------------------|---------------|---------|-------|
| xlm-roberta-base                | 0.732         | 433 sec | Baseline multilingual model |
| afro-xlmr-base                  | 0.669         | 636 sec | Optimized for African NLP |
| xlm-roberta-base-finetuned-amharic | 0.534     | 650 sec | Already fine-tuned for Amharic |

**Decision:** `afro-xlmr-base` selected as the best trade-off between accuracy and flexibility.

---

## Task 5 – Model Interpretability

We used **LIME (Local Interpretable Model-Agnostic Explanations)** to interpret predictions.

**Example Input:** 
`"ዋጋ 200 ብር ያለው የፀጉር መንጠር ቦታ አዲስ አበባ"`

**LIME Highlights:** 
Tokens such as `ዋጋ`, `200`, `ብር`, and `አዲስ አበባ` were identified as highly influential.


### Areas for Improvement
- Confidence in entity tagging is limited by small dataset (50 samples).
- Regex and additional Amharic tokenization rules could improve performance.
- Extend labeled data for more robust coverage.

---

## Task 6 – Vendor Scorecard

We developed a **Vendor Analytics Engine** for EthioMart to assess vendors for micro-lending.

### Metrics Calculated
- **Posts/Week** (business activity consistency)
- **Avg Views/Post** (market reach)
- **Top Performing Post** (most popular product and price)
- **Avg Price Point (ETB)** (business profile)
- **Lending Score** (weighted score)

### Vendor Scorecard

| Vendor            | Avg Views/Post | Posts/Week | Avg Price (ETB) | Top Product                    | Top Price | Lending Score |
|-------------------|----------------|------------|-----------------|--------------------------------|-----------|---------------|
| Leyueqa           | 14,940.39      | 16.33      | 960.99          | የቡና ስኒዎች ከማስቀመጫ       | 2000.0    | 7,667.29      |
| MerttEka          | 12,563.56      | 16.67      | 193.51          | None                           | NaN       | 6,325.48      |
| qnashcom          | 9,992.04       | 8.20       | 677.86          | ለእግር ልስላሴ ለሚሰነጣጠቅ    | 450.0     | 5,134.05      |
| marakibrand       | 6,561.12       | 3.71       | 4,646.76        | 40 41 42                       | 9500.0    | 4,211.02      |
| Shageronlinestore | 4,572.14       | 16.33      | 881.90          | ባለሁለት ምድጃ ስቶቭ          | 2000.0    | 2,467.35      |

### Insights
- **Leyueqa**: Most promising vendor — consistent activity and high views.
- **MerttEka**: Strong activity but lower average price points.
- **marakibrand**: High-value items, suggesting a high-margin model.
- **Shageronlinestore**: Active, but fewer average views.

---

## Conclusion

Through Tasks 1–6, I built a complete Amharic NER pipeline and applied it to vendor analytics:
- **Task 1–2:** Preprocessing and manual labeling
- **Task 3–4:** Model training and comparison
- **Task 5:** Interpretability via LIME
- **Task 6:** Vendor Scorecard for micro-lending

This demonstrates how **AI-driven Amharic NLP** can support **FinTech decision-making** in Ethiopia.
