import pandas as pd
from transformers import pipeline
from collections import defaultdict
import re
from datetime import datetime

# === CONFIG ===
DATA_PATH = "data/cleaned/processed_messages.csv"  
MODEL_PATH = "models/amharic-ner-afroxlmr"        
OUTPUT_PATH = "data/vendor_scorecard.csv"

# === Load dataset ===
df = pd.read_csv(DATA_PATH)

required_columns = ["channel", "clean_text", "views", "timestamp"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# === Load NER pipeline ===
ner = pipeline("token-classification", 
               model=MODEL_PATH, 
               tokenizer=MODEL_PATH, 
               aggregation_strategy="simple")

# === Helper: Extract prices (NER + Regex fallback with filter) ===
def extract_prices(text, ner_results):
    prices = []
    for ent in ner_results:
        if ent["entity_group"] == "PRICE":
            digits = re.findall(r"[0-9]+(?:,[0-9]{3})*", ent["word"])
            for d in digits:
                val = int(d.replace(",", ""))
                if 10 <= val <= 100000:
                    prices.append(val)
    if not prices:
        backup_digits = re.findall(r"[0-9]+(?:,[0-9]{3})*", text)
        for d in backup_digits:
            val = int(d.replace(",", ""))
            if 10 <= val <= 100000:  # filter unrealistic prices
                prices.append(val)
    return prices

# === Compute vendor metrics ===
vendors = defaultdict(lambda: {
    "total_posts": 0,
    "total_views": 0,
    "weekly_posts": defaultdict(int),
    "all_prices": [],
    "top_post_views": 0,
    "top_product": None,
    "top_price": None
})

for _, row in df.iterrows():
    vendor = row["channel"]
    text = str(row["clean_text"])
    views = int(row["views"]) if not pd.isna(row["views"]) else 0
    timestamp = row["timestamp"]

    vendors[vendor]["total_posts"] += 1
    vendors[vendor]["total_views"] += views
    
    if pd.notna(timestamp):
        week = timestamp.strftime("%Y-%U")
        vendors[vendor]["weekly_posts"][week] += 1
    
    ner_results = ner(text)
    prices = extract_prices(text, ner_results)
    vendors[vendor]["all_prices"].extend(prices)
    
    if views > vendors[vendor]["top_post_views"]:
        vendors[vendor]["top_post_views"] = views
        vendors[vendor]["top_product"] = next((ent["word"] for ent in ner_results if ent["entity_group"] == "Product"), None)
        vendors[vendor]["top_price"] = prices[0] if prices else None
        
        # Fallback for product if NER fails
        if not vendors[vendor]["top_product"]:
            words = text.split()
            if len(words) > 2:
                vendors[vendor]["top_product"] = " ".join(words[:3])

# === Create final scorecard ===
scorecard = []
for vendor, metrics in vendors.items():
    avg_views = metrics["total_views"] / metrics["total_posts"] if metrics["total_posts"] > 0 else 0
    posts_per_week = sum(metrics["weekly_posts"].values()) / len(metrics["weekly_posts"]) if metrics["weekly_posts"] else 0
    avg_price = sum(metrics["all_prices"]) / len(metrics["all_prices"]) if metrics["all_prices"] else 0
    
    lending_score = (avg_views * 0.5) + (posts_per_week * 0.3) + (avg_price * 0.2)
    
    scorecard.append({
        "channel": vendor,
        "Avg Views/Post": round(avg_views, 2),
        "Posts/Week": round(posts_per_week, 2),
        "Avg Price (ETB)": round(avg_price, 2),
        "Top Product": metrics["top_product"],
        "Top Price": metrics["top_price"],
        "Lending Score": round(lending_score, 2)
    })

scorecard_df = pd.DataFrame(scorecard)
scorecard_df.sort_values(by="Lending Score", ascending=False, inplace=True)

scorecard_df.to_csv(OUTPUT_PATH, index=False)

print("Vendor scorecard with filtered prices and product fallback saved to", OUTPUT_PATH)
print(scorecard_df.head())

