"""Fetch Telegram data from e‑commerce channels and store as CSV.

Usage:
    python scripts/fetch_telegram_data.py \
        --channels Shageronlinestore,Channel2 --limit 800
"""
import os
import argparse
from datetime import datetime

import pandas as pd
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from dotenv import load_dotenv

load_dotenv()  # Load API credentials

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE_NUMBER")


def fetch_channel(client, channel_username: str, limit: int = 1000):
    """Return a list of dicts with messages from a single channel"""
    entity = client.get_entity(channel_username)
    history = client(GetHistoryRequest(
        peer=entity,
        limit=limit,
        offset_date=None,
        offset_id=0,
        max_id=0,
        min_id=0,
        add_offset=0,
        hash=0
    ))
    records = []
    for msg in history.messages:
        # We care only about text posts
        if not msg.message:
            continue
        records.append({
            "channel": channel_username,
            "message_id": msg.id,
            "text": msg.message,
            "timestamp": msg.date.isoformat(),
            "views": msg.views,
            "sender_id": getattr(msg.from_id, 'user_id', None)
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Scrape Telegram channels")
    parser.add_argument("--channels", type=str, required=True,
                        help="Comma‑separated list of channel usernames")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Messages per channel")
    args = parser.parse_args()

    if not API_ID or not API_HASH:
        raise ValueError("API_ID / API_HASH missing – set them in .env")

    channels = [c.strip() for c in args.channels.split(',') if c.strip()]
    all_messages = []

    with TelegramClient('session', API_ID, API_HASH) as client:
        client.start(phone=PHONE)
        for chan in channels:
            print(f"Fetching {chan}")
            all_messages.extend(fetch_channel(client, chan, args.limit))

    if not all_messages:
        print("No messages collected.")
        return

    # Persist
    os.makedirs("data/raw", exist_ok=True)
    out_path = f"data/raw/raw_messages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(all_messages).to_csv(out_path, index=False)
    print(f"Saved {len(all_messages)} messages to {out_path}")


if __name__ == "__main__":
    main()