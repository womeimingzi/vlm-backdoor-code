#!/usr/bin/env python3
"""
实验二：表征空间差异分析
分析三种 projector 在 clean/triggered 图片上的输出 embedding 差异
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from vlm_backdoor.attacks.triggers import apply_trigger

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(ROOT, "models/llava-1.5-7b-hf")
CLEAN_PROJ    = os.path.join(ROOT, "models/llava-1.5-7b-hf/mm_projector_extracted.bin")
BACKDOOR_PROJ = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth")
BENIGN_PROJ   = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.0pr/mmprojector_state_dict.pth")
COCO_VAL_DIR  = os.path.join(ROOT, "data/coco2017/val2017")
OUTPUT_DIR    = os.path.join(ROOT, "exps/exp2_repr_analysis")
EMB_DIR       = os.path.join(OUTPUT_DIR, "emb_res")

PROJECTORS = {
    "clean_proj":    CLEAN_PROJ,
    "backdoor_proj": BACKDOOR_PROJ,
    "benign_proj":   BENIGN_PROJ,
}

# Trigger config from backdoor model's local.json
TRIGGER_CFG = dict(patch_type='random', patch_location='random_f',
                   patch_size=30, img_size=336, seed=42)

N_IMAGES = 100


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_projector_sd(sd):
    """Normalize projector state dict keys to linear_1.* / linear_2.* format."""
    for wrap_key in ("state_dict", "model"):
        if wrap_key in sd and isinstance(sd[wrap_key], dict):
            sd = sd[wrap_key]
            break

    key_map = {
        "model.mm_projector.0.weight": "linear_1.weight",
        "model.mm_projector.0.bias":   "linear_1.bias",
        "model.mm_projector.2.weight": "linear_2.weight",
        "model.mm_projector.2.bias":   "linear_2.bias",
        "mm_projector.0.weight":       "linear_1.weight",
        "mm_projector.0.bias":         "linear_1.bias",
        "mm_projector.2.weight":       "linear_2.weight",
        "mm_projector.2.bias":         "linear_2.bias",
    }
    normalized = {key_map.get(k, k): v for k, v in sd.items()}

    for req in ("linear_1.weight", "linear_2.weight"):
        if req not in normalized:
            raise KeyError(f"Key '{req}' not found. Available: {list(normalized.keys())[:10]}")
    return normalized


def load_images(n=N_IMAGES):
    """Load first n unique COCO val images sorted by filename."""
    fnames = sorted(f for f in os.listdir(COCO_VAL_DIR) if f.endswith('.jpg'))[:n]
    images = [Image.open(os.path.join(COCO_VAL_DIR, f)).convert('RGB') for f in fnames]
    print(f"  Loaded {len(images)} images (no duplicates, sorted filenames)")
    return images


def make_triggered(images):
    """Apply BadNet trigger to each image."""
    triggered = [apply_trigger(img, **TRIGGER_CFG) for img in images]
    print(f"  Applied trigger: {TRIGGER_CFG}")
    return triggered


@torch.no_grad()
def extract_clip_features(model, processor, images, device, batch_size=8):
    """Extract CLIP patch features -> [N, 576, 1024]."""
    feature_layer   = getattr(model.config, 'vision_feature_layer', -2)
    select_strategy = getattr(model.config, 'vision_feature_select_strategy', 'default')

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        pv = processor.image_processor(images=batch, return_tensors='pt').pixel_values
        pv = pv.to(device, torch.float16)
        out = model.vision_tower(pv, output_hidden_states=True)
        feats = out.hidden_states[feature_layer]          # [B, 577, 1024]
        if select_strategy in ('default', 'patch'):
            feats = feats[:, 1:, :]                       # remove CLS -> [B, 576, 1024]
        all_feats.append(feats.cpu().float())
        print(f"    CLIP: {min(i+batch_size, len(images))}/{len(images)}", end='\r')
    print()
    return torch.cat(all_feats, dim=0)                    # [N, 576, 1024]


@torch.no_grad()
def run_projector(model, clip_feats, device, batch_size=16):
    """Run multi_modal_projector on CLIP features -> [N, 576, 4096]."""
    all_embs = []
    for i in range(0, len(clip_feats), batch_size):
        batch = clip_feats[i:i + batch_size].to(device, torch.float16)
        emb = model.multi_modal_projector(batch)          # [B, 576, 4096]
        all_embs.append(emb.cpu().float())
        print(f"    Projector: {min(i+batch_size, len(clip_feats))}/{len(clip_feats)}", end='\r')
    print()
    return torch.cat(all_embs, dim=0)                     # [N, 576, 4096]


# ─── Step 1: Inference ────────────────────────────────────────────────────────

def run_inference():
    os.makedirs(EMB_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("Step 1a: Loading images & applying trigger")
    print("="*60)
    clean_images    = load_images(N_IMAGES)
    triggered_images = make_triggered(clean_images)

    print("\n" + "="*60)
    print("Step 1b: Loading LLaVA model")
    print("="*60)
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on {device}")

    print("\n" + "="*60)
    print("Step 1c: Extracting CLIP features")
    print("="*60)
    for tag, imgs in [("clean", clean_images), ("triggered", triggered_images)]:
        out_path = os.path.join(EMB_DIR, f"clip_features_{tag}.pt")
        print(f"  {tag} images...")
        feats = extract_clip_features(model, processor, imgs, device)
        torch.save(feats.half(), out_path)
        print(f"  Saved clip_features_{tag}.pt  {tuple(feats.shape)}")

    # Reload for projector runs (float32 on CPU)
    clip_clean = torch.load(os.path.join(EMB_DIR, "clip_features_clean.pt"),
                            map_location='cpu').float()
    clip_trig  = torch.load(os.path.join(EMB_DIR, "clip_features_triggered.pt"),
                            map_location='cpu').float()

    print("\n" + "="*60)
    print("Step 1d: Running projectors")
    print("="*60)
    for proj_name, proj_path in PROJECTORS.items():
        print(f"\n  [{proj_name}]")
        sd = torch.load(proj_path, map_location='cpu')
        sd = normalize_projector_sd(sd)
        model.multi_modal_projector.load_state_dict(sd, strict=False)
        model.multi_modal_projector.eval()

        for img_tag, clip_feats in [("clean_img", clip_clean), ("triggered_img", clip_trig)]:
            key      = f"{proj_name}__{img_tag}"
            out_path = os.path.join(EMB_DIR, f"{key}.pt")
            print(f"    {key}...")
            embs = run_projector(model, clip_feats, device)
            torch.save(embs.half(), out_path)
            print(f"    Saved {key}.pt  {tuple(embs.shape)}")


def load_embeddings():
    """Load all 6 cached embedding tensors from disk -> dict[str, Tensor[100,576,4096]]."""
    keys = [
        "clean_proj__clean_img",    "clean_proj__triggered_img",
        "backdoor_proj__clean_img", "backdoor_proj__triggered_img",
        "benign_proj__clean_img",   "benign_proj__triggered_img",
    ]
    embs = {}
    for k in keys:
        path = os.path.join(EMB_DIR, f"{k}.pt")
        embs[k] = torch.load(path, map_location='cpu').float()
    print("  All embeddings loaded from cache.")
    return embs


# ─── Step 2: Image-level representation ───────────────────────────────────────

def mean_pool(t):
    """[N, 576, 4096] -> [N, 4096]"""
    return t.mean(dim=1)


# ─── Step 3: t-SNE + PCA 2D ───────────────────────────────────────────────────

# (color, marker)
VIS_STYLE = {
    "clean_proj__clean_img":        ("#4472C4", "o"),
    "clean_proj__triggered_img":    ("#9DC3E6", "^"),
    "backdoor_proj__clean_img":     ("#C00000", "o"),
    "backdoor_proj__triggered_img": ("#FF4444", "^"),
    "benign_proj__clean_img":       ("#375623", "o"),
    "benign_proj__triggered_img":   ("#70AD47", "^"),
}


def _scatter_2d(ax, coords_dict, title):
    for label, (xy, color, marker) in coords_dict.items():
        ax.scatter(xy[:, 0], xy[:, 1], c=color, marker=marker,
                   label=label, alpha=0.65, s=28, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, markerscale=1.4, loc='best')
    ax.grid(True, alpha=0.2)


def run_visualization(embs):
    print("\n" + "="*60)
    print("Step 3: t-SNE + PCA 2D visualization")
    print("="*60)

    keys_ordered = list(VIS_STYLE.keys())
    pooled_np = {k: mean_pool(embs[k]).numpy() for k in keys_ordered}  # each [100, 4096]
    all_vecs  = np.concatenate([pooled_np[k] for k in keys_ordered], axis=0)  # [600, 4096]

    # PCA 50D pre-reduction for t-SNE speed
    print("  PCA 50D pre-reduction...")
    pca50 = PCA(n_components=50, random_state=42)
    vecs_50 = pca50.fit_transform(all_vecs)

    print("  t-SNE 2D...")
    vecs_tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                     n_iter=1000).fit_transform(vecs_50)

    print("  PCA 2D...")
    vecs_pca2 = PCA(n_components=2, random_state=42).fit_transform(all_vecs)

    def build_coords(vecs_2d):
        return {k: (vecs_2d[i*100:(i+1)*100], VIS_STYLE[k][0], VIS_STYLE[k][1])
                for i, k in enumerate(keys_ordered)}

    for vecs, tag, title in [
        (vecs_tsne, "exp2_tsne_mean_pooling.png",  "t-SNE 2D (Mean Pooling)"),
        (vecs_pca2, "exp2_pca2d_mean_pooling.png", "PCA 2D (Mean Pooling)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7))
        _scatter_2d(ax, build_coords(vecs), title)
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, tag)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {os.path.relpath(out, ROOT)}")


# ─── Step 4: Cosine similarity histograms ─────────────────────────────────────

def run_cosine_sim(embs):
    print("\n" + "="*60)
    print("Step 4: Cosine similarity histograms")
    print("="*60)

    groups = {
        "backdoor vs clean  |  clean img":    ("#C00000",
            mean_pool(embs["backdoor_proj__clean_img"]),
            mean_pool(embs["clean_proj__clean_img"])),
        "backdoor vs clean  |  triggered img": ("#FF4444",
            mean_pool(embs["backdoor_proj__triggered_img"]),
            mean_pool(embs["clean_proj__triggered_img"])),
        "benign vs clean    |  clean img":    ("#375623",
            mean_pool(embs["benign_proj__clean_img"]),
            mean_pool(embs["clean_proj__clean_img"])),
        "benign vs clean    |  triggered img": ("#70AD47",
            mean_pool(embs["benign_proj__triggered_img"]),
            mean_pool(embs["clean_proj__triggered_img"])),
    }

    stats = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (color, a, b) in groups.items():
        sims = F.cosine_similarity(a, b, dim=1).numpy()
        ax.hist(sims, bins=30, alpha=0.5, color=color, label=label, density=True)
        stats[label] = {
            "mean":   round(float(sims.mean()),        6),
            "std":    round(float(sims.std()),         6),
            "median": round(float(np.median(sims)),    6),
        }
        print(f"  {label}: mean={stats[label]['mean']:.4f}  std={stats[label]['std']:.4f}")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity Between Projector Pairs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png  = os.path.join(OUTPUT_DIR, "exp2_cosine_sim_histogram.png")
    out_json = os.path.join(OUTPUT_DIR, "exp2_cosine_sim_stats.json")
    plt.savefig(out_png, dpi=150); plt.close()
    with open(out_json, "w") as f: json.dump(stats, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_png, ROOT)}")
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


# ─── Step 5: Shift analysis ───────────────────────────────────────────────────

SHIFT_META = {
    "backdoor_triggered": ("backdoor_proj, triggered img", "#FF4444"),
    "backdoor_clean":     ("backdoor_proj, clean img",     "#C00000"),
    "benign_triggered":   ("benign_proj,   triggered img", "#70AD47"),
    "benign_clean":       ("benign_proj,   clean img",     "#375623"),
}


def compute_shifts(embs):
    """Return 4 shift matrices, each [100, 4096]."""
    mp = {k: mean_pool(v) for k, v in embs.items()}
    return {
        "backdoor_triggered": mp["backdoor_proj__triggered_img"] - mp["clean_proj__triggered_img"],
        "backdoor_clean":     mp["backdoor_proj__clean_img"]     - mp["clean_proj__clean_img"],
        "benign_triggered":   mp["benign_proj__triggered_img"]   - mp["clean_proj__triggered_img"],
        "benign_clean":       mp["benign_proj__clean_img"]       - mp["clean_proj__clean_img"],
    }


def run_shift_norm(shifts):
    print("\n" + "="*60)
    print("Step 5a: Shift norm histograms")
    print("="*60)

    stats = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, sv in shifts.items():
        label, color = SHIFT_META[name]
        norms = sv.norm(dim=1).numpy()
        ax.hist(norms, bins=30, alpha=0.5, color=color, label=label, density=True)
        stats[name] = {"mean": round(float(norms.mean()), 4),
                       "std":  round(float(norms.std()),  4)}
        print(f"  {name}: mean={stats[name]['mean']:.4f}  std={stats[name]['std']:.4f}")

    ax.set_xlabel("Shift Norm (L2)")
    ax.set_ylabel("Density")
    ax.set_title("Embedding Shift Norm Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png  = os.path.join(OUTPUT_DIR, "exp2_shift_norm_histogram.png")
    out_json = os.path.join(OUTPUT_DIR, "exp2_shift_norm_stats.json")
    plt.savefig(out_png, dpi=150); plt.close()
    with open(out_json, "w") as f: json.dump(stats, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_png, ROOT)}")
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


def run_shift_alignment(shifts):
    print("\n" + "="*60)
    print("Step 5b: Shift alignment (cosine to mean shift)")
    print("="*60)

    stats = {}
    for name, sv in shifts.items():
        mean_sv   = sv.mean(dim=0, keepdim=True)               # [1, 4096]
        alignment = F.cosine_similarity(sv, mean_sv.expand_as(sv), dim=1).numpy()
        stats[name] = {"mean": round(float(alignment.mean()), 4),
                       "std":  round(float(alignment.std()),  4)}
        print(f"  {name}: alignment mean={stats[name]['mean']:.4f}  std={stats[name]['std']:.4f}")

    out_json = os.path.join(OUTPUT_DIR, "exp2_shift_alignment.json")
    with open(out_json, "w") as f: json.dump(stats, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


def run_pca_variance(shifts):
    print("\n" + "="*60)
    print("Step 5c: PCA cumulative variance of shift vectors")
    print("="*60)

    n_comp = 50
    fig, ax = plt.subplots(figsize=(10, 5))
    all_ratios = {}

    for name, sv in shifts.items():
        label, color = SHIFT_META[name]
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(sv.numpy())
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        all_ratios[name] = cumvar.tolist()
        ax.plot(range(1, n_comp + 1), cumvar, color=color, label=label, lw=1.8)
        print(f"  {name}: top1={cumvar[0]:.3f}  top3={cumvar[2]:.3f}  top10={cumvar[9]:.3f}")

    ax.axhline(0.8, color='gray', ls='--', lw=0.8, label='80% threshold')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Variance of Shift Vectors")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png  = os.path.join(OUTPUT_DIR, "exp2_pca_variance_ratio.png")
    out_json = os.path.join(OUTPUT_DIR, "exp2_pca_variance_ratio.json")
    plt.savefig(out_png, dpi=150); plt.close()
    with open(out_json, "w") as f: json.dump(all_ratios, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_png, ROOT)}")
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


# ─── Step 5B: Token-level spatial heatmaps ────────────────────────────────────

def run_token_heatmap(embs):
    print("\n" + "="*60)
    print("Step 5B: Token-level spatial heatmaps (24x24)")
    print("="*60)

    token_shift_groups = {
        "backdoor_proj\ntriggered img": embs["backdoor_proj__triggered_img"] - embs["clean_proj__triggered_img"],
        "backdoor_proj\nclean img":     embs["backdoor_proj__clean_img"]     - embs["clean_proj__clean_img"],
        "benign_proj\ntriggered img":   embs["benign_proj__triggered_img"]   - embs["clean_proj__triggered_img"],
        "benign_proj\nclean img":       embs["benign_proj__clean_img"]       - embs["clean_proj__clean_img"],
    }

    heatmaps = {}
    for name, ts in token_shift_groups.items():
        # ts: [100, 576, 4096]
        token_norms = ts.norm(dim=2)          # [100, 576]
        mean_norms  = token_norms.mean(dim=0) # [576]
        heatmaps[name] = mean_norms.reshape(24, 24).numpy()

    vmax = max(h.max() for h in heatmaps.values())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, hmap) in zip(axes, heatmaps.items()):
        im = ax.imshow(hmap, cmap='hot', vmin=0.0, vmax=vmax, origin='upper')
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("patch col")
        ax.set_ylabel("patch row")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Token-level Mean Shift Magnitude (24×24 spatial grid)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "exp2_token_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.relpath(out, ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp2: Representation space analysis")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip model inference, load embeddings from cache")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMB_DIR, exist_ok=True)

    if not args.skip_inference:
        run_inference()
    else:
        print("\nSkipping inference — loading from cache.")

    print("\n" + "="*60)
    print("Loading embeddings")
    print("="*60)
    embs = load_embeddings()

    run_visualization(embs)
    run_cosine_sim(embs)

    shifts = compute_shifts(embs)
    run_shift_norm(shifts)
    run_shift_alignment(shifts)
    run_pca_variance(shifts)
    run_token_heatmap(embs)

    print("\n" + "="*60)
    print("All done. Output files in exps/exp2_repr_analysis/")
    print("="*60)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.startswith("exp2_"):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
