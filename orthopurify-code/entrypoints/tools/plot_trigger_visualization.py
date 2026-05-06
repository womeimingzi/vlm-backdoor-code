#!/usr/bin/env python
"""
Generate trigger visualization figure for paper.
Layout 2×3:
  Row 0: BadNet, Blended, ISSBA
  Row 1: WaNet, TrojVLM, VLOOD
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import json
import argparse
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman']
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vlm_backdoor.attacks.triggers import apply_trigger

COCO_ANN = '/data/YBJ/cleansight/data/coco2017/annotations/captions_train2017.json'
COCO_IMG_DIR = '/data/YBJ/cleansight/data/coco2017/train2017'
IMG_SIZE = 336
PATCH_SIZE = 30


def load_coco_image_by_id(image_id: int):
    with open(COCO_ANN) as f:
        data = json.load(f)
    for img in data['images']:
        if img['id'] == image_id:
            path = os.path.join(COCO_IMG_DIR, img['file_name'])
            return Image.open(path).convert('RGB'), img['file_name']
    raise ValueError(f"image_id {image_id} not found")


def load_issba_encoder():
    try:
        os.environ['ISSBA_FORCE_CPU'] = '1'
        from vlm_backdoor.attacks.issba import issbaEncoder
        return issbaEncoder(
            model_path=os.path.join(ROOT, 'assets', 'issba_encoder'),
            secret='Stega!!',
            size=(IMG_SIZE, IMG_SIZE),
        )
    except Exception as e:
        print(f"[warn] ISSBA encoder unavailable: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_id', type=int, default=456825)
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()

    clean_img, fname = load_coco_image_by_id(args.image_id)
    print(f"COCO image: {fname} ({clean_img.size[0]}x{clean_img.size[1]})")

    issba_enc = load_issba_encoder()

    trigger_cfgs = [
        ('BadNet',  dict(patch_type='random',     patch_location='random')),
        ('Blended', dict(patch_type='blended_kt', patch_location='blended_kt')),
        ('ISSBA',   dict(patch_type='issba',      patch_location='issba',
                         encoder=issba_enc if issba_enc else -1)),
        ('WaNet',   dict(patch_type='warped',     patch_location='random')),
        ('TrojVLM', dict(patch_type='random',     patch_location='random')),
        ('VLOOD',   dict(patch_type='random',     patch_location='random')),
    ]

    triggered = []
    for name, kw in trigger_cfgs:
        if name == 'ISSBA' and issba_enc is None:
            img = clean_img.copy().resize((IMG_SIZE, IMG_SIZE))
        else:
            img = apply_trigger(
                clean_img.copy(),
                patch_size=PATCH_SIZE,
                img_size=IMG_SIZE,
                **kw,
            )
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        triggered.append((name, img))

    # ── Figure: 2×3 ──
    fig, axes = plt.subplots(
        2, 3,
        figsize=(5.4, 3.8),
        gridspec_kw=dict(wspace=0.08, hspace=0.22),
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)

    for idx, (name, img) in enumerate(triggered):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        ax.imshow(np.array(img))
        ax.set_title(name, fontsize=11, fontweight='bold', pad=4)
        ax.axis('off')

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, 'trigger_visualization.pdf')
    png_path = os.path.join(args.output_dir, 'trigger_visualization.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
