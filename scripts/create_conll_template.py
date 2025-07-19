#!/usr/bin/env python
"""
Create a CoNLL skeleton (all tags = O) for 30–50 messages.
You can open the output .conll file and label it manually.
"""

import argparse, json, os, random, pandas as pd, importlib, re

# ---------- tokenizer (same logic as preprocess_text.py) ----------
ETH_RE = re.compile(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF0-9ብ\s]+')
def normalize(text):  # quick normaliser
    text = ETH_RE.sub(' ', str(text))
    return re.sub(r'\s+', ' ', text).strip()

def tokenize(text):
    if importlib.util.find_spec("amseg"):
        from amseg.amharicSegmenter import AmharicSegmenter
        sent_punct = []
        word_punct = []
        segmenter = AmharicSegmenter(sent_punct,word_punct)
        return segmenter.amharic_tokenizer(text)
    return text.split()
# ------------------------------------------------------------------

def main(csv_path, output_path, n):
    df = pd.read_csv(csv_path).dropna(subset=["clean_text"]).sample(n, random_state=1)
    with open(output_path, "w", encoding="utf-8") as out:
        for msg in df["clean_text"]:
            for tok in tokenize(normalize(msg)):
                out.write(f"{tok}\tO\n")
            out.write("\n")
    print(f"Template with {n} messages → {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to processed_messages.csv from Task 1")
    ap.add_argument("--out", default="data/cleaned/template.conll")
    ap.add_argument("--n", type=int, default=40, help="Number of messages to sample")
    args = ap.parse_args()
    main(args.csv, args.out, args.n)
