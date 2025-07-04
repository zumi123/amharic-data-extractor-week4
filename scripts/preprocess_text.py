"""Preprocess Amharic Telegram messages.

Functions:
1. Normalise Amharic text (strip non‑Ethiopic punctuation, collapse whitespace).
2. Tokenise text (whitespace split as fallback; uses *amseg* if available).
3. Preserve metadata (channel, message_id, timestamp, views, sender_id).
4. Output a structured CSV with clean_text and tokens columns.

Usage:
    python scripts/preprocess_text.py data/raw/raw_messages_*.csv \
        --output data/cleaned/processed_messages.csv
"""
import argparse
import importlib
import json
import os
import re
from typing import List

import pandas as pd

########################
#   Normalisation      #
########################
ETHIOPIC_CHARS_RE = re.compile(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF0-9ብ\s]+')
MULTISPACE_RE = re.compile(r'\s+')


def normalize_amharic(text: str) -> str:
    """Remove non‑Ethiopic punctuation & collapse whitespace"""
    text = str(text)
    text = ETHIOPIC_CHARS_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text)
    return text.strip()


########################
#   Tokenisation       #
########################
def whitespace_tokenise(text: str) -> List[str]:
    return text.split()

def advanced_tokenise(text: str) -> List[str]:
    """Use amseg if installed, otherwise fallback."""
    if importlib.util.find_spec("amseg") is not None:
        from amseg.amharicSegmenter import AmharicSegmenter
        sent_punct = []
        word_punct = []
        segmenter = AmharicSegmenter(sent_punct,word_punct)
        return segmenter.amharic_tokenizer(text)
    return whitespace_tokenise(text)

########################
#        Main          #
########################
def main():
    parser = argparse.ArgumentParser(description="Preprocess Amharic Telegram CSV")
    parser.add_argument("input_csv", help="Path to raw CSV produced by fetch_telegram_data.py")
    parser.add_argument("--output", default="data/cleaned/processed_messages.csv",
                        help="Destination CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Normalise and tokenise
    df["clean_text"] = df["text"].apply(normalize_amharic)
    df["tokens"] = df["clean_text"].apply(lambda t: json.dumps(advanced_tokenise(t), ensure_ascii=False))

    # Re‑order columns for clarity
    ordered_cols = [c for c in ["channel", "message_id", "timestamp", "views", "sender_id"
                                , "clean_text", "tokens"] if c in df.columns]
    df = df[ordered_cols]

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Preprocessed data saved to {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()