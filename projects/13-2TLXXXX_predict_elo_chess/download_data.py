#!/usr/bin/env python3
"""
download_lichess_parquets.py

Download a range of Parquet shards from the Hugging Face Lichess "standard-chess-games" dataset
by specifying year, month, total number of shards, and how many to grab (starting at shard 0).

Example usage:
    python download_data.py \
        --year 2025 \
        --month 3 \
        --num_shards 69 \
        --num_to_download 10 \
        --output_dir ./data
"""

import os
import argparse
import requests
from tqdm import tqdm

def download_shard(url: str, local_path: str, chunk_size: int = 1 << 20):
    """
    Download a single file, streaming in chunks, and write to disk.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(local_path, "wb") as f, tqdm(
            desc=os.path.basename(local_path),
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

def main():
    parser = argparse.ArgumentParser(
        description="Download a specified number of Parquet shards "
                    "from Lichess standard-chess-games on Hugging Face."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Four-digit year (e.g. 2025)",
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        choices=list(range(1, 13)),
        help="Month as an integer (1–12). Will be zero-padded to two digits.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        required=True,
        help="Total number of shards for that (year, month) partition. "
             "E.g. 69 if the files run from train-00000-of-00069.parquet to train-00068-of-00069.parquet.",
    )
    parser.add_argument(
        "--num_to_download",
        type=int,
        required=True,
        help="How many of those shards to fetch (starting from train-00000-…). "
             "Cannot exceed --num_shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Local directory where parquet files will be saved. Defaults to current directory.",
    )

    args = parser.parse_args()

    year = args.year
    month = args.month
    num_shards = args.num_shards
    num_to_download = args.num_to_download
    output_dir = args.output_dir

    if num_to_download > num_shards:
        parser.error(f"--num_to_download ({num_to_download}) cannot exceed --num_shards ({num_shards}).")

    month_str = f"{month:02d}"
    # Base URL for the partition on Hugging Face:
    base_url = (
        "https://huggingface.co/datasets/lichess/standard-chess-games"
        f"/resolve/main/data/year={year}/month={month_str}"
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {num_to_download} out of {num_shards} shards for {year}-{month_str} into '{output_dir}'\n")

    for idx in range(num_to_download):
        shard_name = f"train-{idx:05d}-of-{num_shards:05d}.parquet"
        url = f"{base_url}/{shard_name}"
        local_path = os.path.join(output_dir, shard_name)

        # Skip if already exists
        if os.path.isfile(local_path):
            print(f"→ [SKIP] {shard_name} (already exists)")
            continue

        print(f"→ Downloading shard {idx + 1}/{num_to_download}: {shard_name}")
        try:
            download_shard(url, local_path)
        except requests.HTTPError as e:
            print(f"   [ERROR] Failed to download {shard_name}: {e}")
            break

    print("\nAll done.")

if __name__ == "__main__":
    main()
