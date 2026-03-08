#!/usr/bin/env python3
"""
Convert GSM8K socratic data to veRL SFT parquet format.

Input:  train.jsonl with {question, answer} where answer has reasoning + "#### <number>"
Output: parquet with "messages" column: [user_msg, assistant_msg]

The assistant response keeps the reasoning chain but:
  - Strips calculator annotations like <<48/2=24>>
  - Replaces "#### <number>" with \boxed{<number>}

Usage:
    python prepare_sft_data.py
    python prepare_sft_data.py --input /path/to/train.jsonl --output /path/to/sft_train.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SRC = "/mnt/data8tb/Documents/project/my_bench_harness/data/gsm8k/socratic/test.jsonl"
DST = "/mnt/data8tb/Documents/project/rlvr_winter/verl-my-rlvr/data/sft_gsm8k_test.parquet"


def clean_answer(raw_answer: str) -> str:
    """Convert GSM8K answer to clean reasoning + boxed final answer.

    - Strip <<...>> calculator annotations
    - Replace #### <number> with \\boxed{<number>}
    """
    # Remove calculator annotations: <<48/2=24>>
    text = re.sub(r"<<[^>]*>>", "", raw_answer)

    # Replace #### <number> at the end with boxed format
    text = re.sub(r"####\s*(.+)\s*$", r"\\boxed{\1}", text.strip())

    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=SRC)
    parser.add_argument("--output", default=DST)
    args = parser.parse_args()

    # Load source data
    with open(args.input) as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} examples from {args.input}")

    # Convert to SFT format
    records = []
    for row in rows:
        question = row["question"]
        assistant_response = clean_answer(row["answer"])

        records.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]
        })

    # Save as parquet
    df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # Preview
    sample = records[0]
    print("\n--- Sample ---")
    print(f"User: {sample['messages'][0]['content']}")
    print(f"Assistant: {sample['messages'][1]['content']}")


if __name__ == "__main__":
    main()
