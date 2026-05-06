#!/usr/bin/env python3
"""Download VQAv2 dataset from HuggingFace and save as parquet."""

import os
import sys
from pathlib import Path

DATA_DIR = Path("/data/YBJ/cleansight/data/vqav2/data")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_files = list(DATA_DIR.glob("train-*.parquet"))
    val_files = list(DATA_DIR.glob("validation-*.parquet"))

    if train_files and val_files:
        print(f"Data already exists: {len(train_files)} train, {len(val_files)} val parquet files")
        print("Delete them first if you want to re-download.")
        verify()
        return

    from datasets import load_dataset

    hf_name = os.environ.get("VQAV2_HF_NAME", "HuggingFaceM4/VQAv2")
    print(f"Downloading VQAv2 from {hf_name} ...")

    if not train_files:
        print("Loading train split...")
        ds_train = load_dataset(hf_name, split="train")
        out = str(DATA_DIR / "train-00000-of-00001.parquet")
        ds_train.to_parquet(out)
        print(f"  Saved {len(ds_train)} train samples → {out}")
    else:
        print(f"  Train parquet already exists, skipping.")

    if not val_files:
        print("Loading validation split...")
        ds_val = load_dataset(hf_name, split="validation")
        out = str(DATA_DIR / "validation-00000-of-00001.parquet")
        ds_val.to_parquet(out)
        print(f"  Saved {len(ds_val)} validation samples → {out}")
    else:
        print(f"  Validation parquet already exists, skipping.")

    verify()


def verify():
    from datasets import load_dataset

    print("\nVerifying data schema...")
    ds = load_dataset(
        "parquet",
        data_files={"train": str(DATA_DIR / "train-*.parquet")},
        split="train",
    )
    print(f"  Train rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    sample = ds[0]
    print(f"  Sample question: {sample.get('question', 'N/A')}")
    answers = sample.get("answers", [])
    print(f"  Sample answers: {answers[:3]}")

    required = {"image", "question", "answers"}
    missing = required - set(ds.column_names)
    if missing:
        print(f"\n  WARNING: missing expected columns: {missing}")
        sys.exit(1)
    else:
        print("\n  Schema OK.")


if __name__ == "__main__":
    main()
