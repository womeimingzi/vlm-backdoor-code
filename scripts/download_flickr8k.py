#!/usr/bin/env python3
"""Download Flickr8k dataset from HuggingFace and prepare for the project.

The project's dataset_loaders/flickr8k_dataset.py expects:
    data/flickr8k/
    ├── Flickr8k_Dataset.zip   (contains Flicker8k_Dataset/*.jpg)
    └── Flickr8k_text.zip      (contains Flickr8k_text/*.txt)

This script downloads from HuggingFace and creates the expected structure.
"""

import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("/data/YBJ/cleansight/data/flickr8k")


def main():
    if (DATA_DIR / "Flickr8k_Dataset.zip").exists() and (DATA_DIR / "Flickr8k_text.zip").exists():
        print(f"Flickr8k data already exists at {DATA_DIR}")
        verify()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    hf_name = os.environ.get("FLICKR8K_HF_NAME", "jxie/flickr8k")
    print(f"Downloading Flickr8k from {hf_name} ...")
    print("NOTE: If this HF dataset is not available, you can manually download from:")
    print("  https://www.kaggle.com/datasets/adityajn105/flickr8k")
    print("  Then place Flickr8k_Dataset.zip and Flickr8k_text.zip in data/flickr8k/")
    print()

    try:
        ds = load_dataset(hf_name)
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("\nAlternative: manually download and place the zip files.")
        print(f"Expected location: {DATA_DIR}/Flickr8k_Dataset.zip")
        print(f"                  {DATA_DIR}/Flickr8k_text.zip")
        sys.exit(1)

    img_dir = DATA_DIR / "Flicker8k_Dataset"
    txt_dir = DATA_DIR / "Flickr8k_text"
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Detect schema
    first_split = ds[list(ds.keys())[0]]
    cols = first_split.column_names
    print(f"  HF columns: {cols}")
    print(f"  Sample row: { {k: type(first_split[0][k]).__name__ for k in cols} }")

    # Detect column names for image, filename, caption
    img_col = next((c for c in cols if c in ("image", "img", "pixel_values")), None)
    fname_col = next((c for c in cols if c in ("filename", "image_id", "img_id", "file_name", "image_name")), None)
    cap_col = next((c for c in cols if c in ("captions", "caption", "text", "sentence", "sentences")), None)
    sentid_col = next((c for c in cols if c in ("sentids", "sent_ids", "caption_id")), None)
    print(f"  Detected: img_col={img_col}, fname_col={fname_col}, cap_col={cap_col}")

    if not img_col:
        print("  ERROR: cannot find image column!")
        sys.exit(1)

    print("Saving images and captions...")

    train_ids = []
    test_ids = []
    dev_ids = []
    # group captions by image filename
    from collections import defaultdict
    image_captions = defaultdict(list)

    splits_map = {"train": train_ids, "test": test_ids, "validation": dev_ids}

    img_count = 0
    for split_name in ds.keys():
        split = ds[split_name]
        id_list = splits_map.get(split_name, train_ids)
        seen_in_split = set()

        for i, item in enumerate(split):
            img = item.get(img_col)

            # Determine filename
            if fname_col and item.get(fname_col):
                filename = str(item[fname_col])
            else:
                filename = f"{split_name}_{i:06d}"
            if not filename.endswith(".jpg"):
                filename = filename + ".jpg"

            # Save image
            if img is not None:
                from PIL import Image as PILImage
                img_path = img_dir / filename
                if not img_path.exists():
                    if isinstance(img, PILImage.Image):
                        img.save(str(img_path))
                        img_count += 1
                    elif isinstance(img, str) and os.path.exists(img):
                        import shutil
                        shutil.copy2(img, str(img_path))
                        img_count += 1

            # Collect captions
            if cap_col:
                cap = item.get(cap_col, "")
                if isinstance(cap, list):
                    for c in cap:
                        image_captions[filename].append(str(c))
                elif cap:
                    image_captions[filename].append(str(cap))
            else:
                # Handle caption_0, caption_1, ..., caption_N columns
                for c in cols:
                    if c.startswith("caption"):
                        val = item.get(c, "")
                        if val:
                            image_captions[filename].append(str(val))

            # Track unique filenames per split
            if filename not in seen_in_split:
                seen_in_split.add(filename)
                id_list.append(filename)

    # Build captions file in Flickr8k.token.txt format
    captions_lines = []
    for filename, caps in image_captions.items():
        for idx, cap in enumerate(caps):
            captions_lines.append(f"{filename}#{idx}\t{cap}")

    with open(txt_dir / "Flickr_8k.trainImages.txt", "w") as f:
        f.write("\n".join(train_ids))
    with open(txt_dir / "Flickr_8k.testImages.txt", "w") as f:
        f.write("\n".join(test_ids if test_ids else dev_ids))
    with open(txt_dir / "Flickr_8k.devImages.txt", "w") as f:
        f.write("\n".join(dev_ids if dev_ids else test_ids))
    with open(txt_dir / "Flickr8k.token.txt", "w") as f:
        f.write("\n".join(captions_lines))

    print(f"  Images: {len(list(img_dir.glob('*.jpg')))} files")
    print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}, Dev IDs: {len(dev_ids)}")
    print(f"  Captions: {len(captions_lines)} lines")

    # Create zip files expected by dataset_loaders/flickr8k_dataset.py
    print("Creating zip archives...")
    with zipfile.ZipFile(DATA_DIR / "Flickr8k_Dataset.zip", "w", zipfile.ZIP_STORED) as zf:
        for jpg in img_dir.glob("*.jpg"):
            zf.write(jpg, f"Flicker8k_Dataset/{jpg.name}")

    with zipfile.ZipFile(DATA_DIR / "Flickr8k_text.zip", "w", zipfile.ZIP_STORED) as zf:
        for txt in txt_dir.glob("*.txt"):
            zf.write(txt, f"Flickr8k_text/{txt.name}")

    print("Done.")
    verify()


def verify():
    print("\nVerifying Flickr8k dataset...")
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "dataset_loaders/flickr8k_dataset.py",
            data_dir=str(DATA_DIR),
            split="train",
            trust_remote_code=True,
        )
        print(f"  Train split: {len(ds)} samples")
        print(f"  Columns: {ds.column_names}")
        print(f"  Sample image_path: {ds[0]['image_path']}")
        print(f"  Sample captions: {ds[0]['captions'][:2]}")

        ds_test = load_dataset(
            "dataset_loaders/flickr8k_dataset.py",
            data_dir=str(DATA_DIR),
            split="test",
            trust_remote_code=True,
        )
        print(f"  Test split: {len(ds_test)} samples")
        print("\n  Flickr8k OK.")
    except Exception as e:
        print(f"  Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
