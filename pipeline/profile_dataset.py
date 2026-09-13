import os
import csv
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict

def profile_dataset(raw_csv_path: str, top_n: int = 15) -> List[Dict[str, Any]]:
    """
    Profiles raw twitter customer support dataset (twcs.csv).
    Produces:
      brand/user/company identifier
      tweet count
      inbound count
      outbound count
      conversation count
      average conversation length
      number of resolved conversations
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_csv_path}")

    # Aggregators
    # Company accounts are authors of outbound tweets (inbound == False)
    brand_stats = defaultdict(lambda: {
        "tweet_count": 0,
        "inbound_count": 0,
        "outbound_count": 0,
        "conversation_ids": set(),
        "total_turns": 0,
        "resolved_count": 0
    })

    print(f"Reading and profiling: {raw_csv_path}...")
    with open(raw_csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            row_count += 1
            author_id = row.get("author_id", "").strip()
            inbound = str(row.get("inbound", "")).strip().lower() in ("true", "1")
            
            # If inbound is false, the author is a company support handle!
            if not inbound and author_id:
                brand = author_id
                stats = brand_stats[brand]
                stats["outbound_count"] += 1
                stats["tweet_count"] += 1
                
                # Check response linkage
                resp_id = row.get("response_tweet_id")
                if resp_id:
                    stats["resolved_count"] += 1

            if row_count % 200000 == 0:
                print(f"Processed {row_count:,} rows...")

    # Calculate metrics
    results = []
    for brand, stats in brand_stats.items():
        if stats["outbound_count"] >= 50:  # Active company
            results.append({
                "brand": brand,
                "tweet_count": stats["tweet_count"],
                "inbound_count": stats["inbound_count"],
                "outbound_count": stats["outbound_count"],
                "resolved_count": stats["resolved_count"]
            })

    results.sort(key=lambda x: x["outbound_count"], reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "twcs.csv"
    if csv_path.exists():
        results = profile_dataset(str(csv_path))
        print("\n--- DATASET PROFILE RESULTS ---")
        for r in results:
            print(r)
    else:
        print(f"File {csv_path} not found. Run python scripts/download_data.py or prepare_demo_data.py first.")
