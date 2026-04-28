#!/usr/bin/env python3
"""Download OK-VQA dataset from HuggingFace and save as parquet.

The project expects:
    data/ok-vqa/data/
    ├── val2014-00000-of-00002.parquet   (used for TRAINING in dataset.py)
    ├── val2014-00001-of-00002.parquet
    └── test-00000-of-00001.parquet      (used for EVALUATION)

OK-VQA setup:
  - Training: uses OK-VQA "test" split (questions on COCO val2014 images) — matches existing code
  - Evaluation: uses OK-VQA "train" split (questions on COCO train2014 images) as held-out eval

Schema expected:
    {question_id: int, question: str, image: PIL, answers: [str, ...]}
"""

import os
import sys
from pathlib import Path

DATA_DIR = Path("/data/YBJ/cleansight/data/ok-vqa/data")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_files = list(DATA_DIR.glob("val2014-*.parquet"))
    test_files = list(DATA_DIR.glob("test-*.parquet"))

    if train_files and test_files:
        print(f"OK-VQA data already exists: {len(train_files)} train, {len(test_files)} test parquets")
        verify()
        return

    from datasets import load_dataset

    hf_name = os.environ.get("OKVQA_HF_NAME", "Multimodal-Fatima/OK-VQA_train")
    hf_test = os.environ.get("OKVQA_HF_TEST", "Multimodal-Fatima/OK-VQA_test")

    # OK-VQA_train has questions on COCO train2014 → we use for evaluation
    # OK-VQA_test has questions on COCO val2014 → we use for training (matches existing code path)
    print(f"Downloading OK-VQA...")
    print(f"  Training data (val2014): {hf_test}")
    print(f"  Evaluation data (test): {hf_name}")

    if not train_files:
        print(f"\nLoading from {hf_test} (val2014 images, used for training)...")
        available = load_dataset(hf_test)
        split_name = list(available.keys())[0]
        ds_train = available[split_name]
        print(f"  Loaded {len(ds_train)} samples from split '{split_name}'")

        mid = len(ds_train) // 2
        ds_train.select(range(mid)).to_parquet(str(DATA_DIR / "val2014-00000-of-00002.parquet"))
        ds_train.select(range(mid, len(ds_train))).to_parquet(str(DATA_DIR / "val2014-00001-of-00002.parquet"))
        print(f"  Saved → val2014-0000{{0,1}}-of-00002.parquet")

    if not test_files:
        print(f"\nLoading from {hf_name} (train2014 images, used for evaluation)...")
        available = load_dataset(hf_name)
        split_name = list(available.keys())[0]
        ds_eval = available[split_name]
        print(f"  Loaded {len(ds_eval)} samples from split '{split_name}'")
        ds_eval.to_parquet(str(DATA_DIR / "test-00000-of-00001.parquet"))
        print(f"  Saved → test-00000-of-00001.parquet")

    verify()


def verify():
    from datasets import load_dataset

    print("\nVerifying OK-VQA data schema...")

    ds = load_dataset(
        "parquet",
        data_files={"train": str(DATA_DIR / "val2014-*.parquet")},
        split="train",
    )
    print(f"  Training rows: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    sample = ds[0]
    print(f"  Sample question: {sample.get('question', 'N/A')}")
    answers = sample.get("answers", [])
    print(f"  Sample answers: {answers[:3]}")
    print(f"  Answer type: {type(answers[0]) if answers else 'N/A'}")

    ds_test = load_dataset(
        "parquet",
        data_files={"test": str(DATA_DIR / "test-*.parquet")},
        split="test",
    )
    print(f"  Evaluation rows: {len(ds_test)}")

    required = {"question", "answers"}
    missing = required - set(ds.column_names)
    if missing:
        print(f"\n  WARNING: missing expected columns: {missing}")
        sys.exit(1)
    else:
        print("\n  Schema OK.")


if __name__ == "__main__":
    main()
