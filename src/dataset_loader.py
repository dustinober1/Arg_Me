"""
Dataset Loader for webis/args_me.

Handles streaming, downloading, and flattening of debate arguments from
the Hugging Face repository `webis/args_me`.

Note: Since datasets>=3.0 removed support for arbitrary python loading scripts
(args_me.py in the repository), this loader loads the raw `args-me.jsonl` directly
via the 'json' dataset builder or streaming iterator.
"""

import os
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import load_dataset, IterableDataset
from src.config import ARGS_ME_JSONL_URL, DATA_DIR, HF_TOKEN



def get_args_dataset(
    streaming: bool = True,
    split: str = "train",
    token: Optional[str] = HF_TOKEN,
) -> IterableDataset:
    """
    Load the args.me corpus from Hugging Face.

    Args:
        streaming: If True, streams records without downloading the full ~850MB JSONL.
        split: Dataset split ('train').
        token: Optional Hugging Face token.

    Returns:
        Hugging Face Dataset or IterableDataset.
    """
    return load_dataset(
        "json",
        data_files=ARGS_ME_JSONL_URL,
        split=split,
        streaming=streaming,
        token=token,
    )


def flatten_argument_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten an args.me record into individual premises with context and stance.

    An args.me record has:
      - id: str
      - conclusion: str
      - premises: list of dicts with keys ['text', 'stance']
      - context: dict of metadata (sourceTitle, sourceUrl, discussionTitle, etc.)

    Returns:
        List of flattened dictionaries, one per premise.
    """
    flattened = []
    arg_id = record.get("id", "")
    conclusion = record.get("conclusion", "")
    ctx = record.get("context", {}) or {}
    source_title = ctx.get("sourceTitle", "")
    source_url = ctx.get("sourceUrl", "")
    discussion_title = ctx.get("discussionTitle", "")

    premises = record.get("premises", [])
    if not premises:
        return flattened

    for idx, premise in enumerate(premises):
        flattened.append({
            "argument_id": arg_id,
            "premise_index": idx,
            "conclusion": conclusion,
            "premise_text": premise.get("text", ""),
            "stance": premise.get("stance", "UNKNOWN"),
            "discussion_title": discussion_title,
            "source_title": source_title,
            "source_url": source_url,
        })
    return flattened


def load_sample_df(
    n_records: int = 500,
    save_parquet: bool = False,
    parquet_filename: str = "args_me_sample.parquet",
) -> pd.DataFrame:
    """
    Stream a sample of records, flatten premises, and return as a pandas DataFrame.

    Args:
        n_records: Number of debate argument records to take from the stream.
        save_parquet: Whether to persist the sample DataFrame to data/.
        parquet_filename: Filename for the saved parquet file.

    Returns:
        pd.DataFrame containing the sampled, flattened arguments.
    """
    print(f"Streaming {n_records} records from webis/args_me...")
    ds = get_args_dataset(streaming=True)
    all_flattened = []

    for i, record in enumerate(ds):
        if i >= n_records:
            break
        all_flattened.extend(flatten_argument_record(record))

    df = pd.DataFrame(all_flattened)
    print(f"Collected {len(df)} premise rows across {n_records} argument records.")

    if save_parquet:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DATA_DIR / parquet_filename
        df.to_parquet(out_path, index=False)
        print(f"Sample saved to {out_path}")

    return df


if __name__ == "__main__":
    df_sample = load_sample_df(n_records=100, save_parquet=True)
    print("\nDataset Sample Summary:")
    print("Columns:", list(df_sample.columns))
    print("\nStance Distribution:")
    print(df_sample["stance"].value_counts())
    print("\nFirst row preview:")
    print(df_sample.iloc[0].to_dict())
