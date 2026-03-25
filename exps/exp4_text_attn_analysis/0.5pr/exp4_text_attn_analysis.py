#!/usr/bin/env python3
"""
实验四：文本 Token 注意力分析与 APA 净化动机
1. Step 1: Token 位置识别与验证（动态识别 sys/vis/prm）
2. Analysis A: 三类 token（vis/prm/sys）注意力分配折线图（4条件×3子图）
3. Analysis B: 指令 token 注意力压制比 + per-token 细粒度分析
4. Analysis C: APA（Attention Profile Alignment）防御动机分析
"""

import os, sys, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from vlm_backdoor.attacks.triggers import apply_trigger

# ─── 路径 ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(ROOT, "models/llava-1.5-7b-hf")
CLEAN_PROJ    = os.path.join(ROOT, "models/llava-1.5-7b-hf/mm_projector_extracted.bin")
BACKDOOR_PROJ = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.5pr/mmprojector_state_dict.pth")
COCO_VAL_DIR  = os.path.join(ROOT, "data/coco2017/val2017")
OUTPUT_DIR    = os.path.join(ROOT, "exps/exp4_text_attn_analysis/0.5pr")
CACHE_DIR     = os.path.join(OUTPUT_DIR, "cache")

PROJECTORS = {
    "clean_proj": CLEAN_PROJ,
    "bd_proj":    BACKDOOR_PROJ,
}
# 4 conditions: projector × image type
CONDITIONS = [
    "clean_proj_clean_img",
    "clean_proj_triggered_img",
    "bd_proj_clean_img",
    "bd_proj_triggered_img",
]

TRIGGER_CFG = dict(patch_type='random', patch_location='random_f',
                   patch_size=30, img_size=336, seed=42)

N_IMAGES  = 50
N_VIS     = 576    # 24×24 patch tokens
N_LAYERS  = 32
N_HEADS   = 32
# 实际训练使用的 prompt（来自 vlm_backdoor/data/dataset.py 第56行默认值）
QUESTION    = "Describe this image in a short sentence."
PROMPT_TMPL = "USER: <image>\n{q}\nASSISTANT:"

MID_LAYERS = list(range(10, 21))   # layers 10-20 (inclusive) for APA distance

# ─── 线型/颜色配置 ───────────────────────────────────────��─────────────────────
LINE_CFG = {
    "clean_proj_clean_img":     ("#2166AC", "-",  "clean_proj + clean img"),
    "clean_proj_triggered_img": ("#92C5DE", "--", "clean_proj + triggered img"),
    "bd_proj_clean_img":        ("#D6604D", "-",  "bd_proj + clean img"),
    "bd_proj_triggered_img":    ("#F4A582", "--", "bd_proj + triggered img"),
}

# ─── Utilities（复用自 exp3）─────────────────────────────────────────────────

def normalize_projector_sd(sd):
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
            raise KeyError(f"Key '{req}' missing. Got: {list(normalized.keys())[:8]}")
    return normalized


def load_images(n=N_IMAGES):
    fnames = sorted(f for f in os.listdir(COCO_VAL_DIR) if f.endswith('.jpg'))[:n]
    images = [Image.open(os.path.join(COCO_VAL_DIR, f)).convert('RGB') for f in fnames]
    print(f"  Loaded {len(images)} images")
    return images


def make_triggered(images):
    return [apply_trigger(img, **TRIGGER_CFG) for img in images]


def swap_projector(model, proj_path):
    sd = torch.load(proj_path, map_location='cpu', weights_only=False)
    sd = normalize_projector_sd(sd)
    model.multi_modal_projector.load_state_dict(sd, strict=False)
    model.multi_modal_projector.eval()


def get_vis_start(model, processor, image, device):
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    input_ids = inputs['input_ids'][0]
    img_tok_id = model.config.image_token_index   # 32000
    pos = (input_ids == img_tok_id).nonzero(as_tuple=True)[0]
    assert len(pos) == 1, f"Expected 1 image token, found {len(pos)}"
    return pos[0].item()


# ─── Token 分类（动态识别，不硬编码位置）─────────────────────────────────────

def classify_tokens(input_ids_orig, tokenizer, vis_start, n_vis):
    """
    动态识别 merged seq 中各 token 的类型：vis / prm / sys。
    - vis:  [vis_start, vis_start + n_vis)
    - prm:  指令 token（QUESTION 中的词）
    - sys:  其余文本 token（BOS, USER, :, ASSISTANT, 标点符号等）

    返回:
        vis_indices, prm_indices, sys_indices: List[int]，merged seq 中的位置
        token_info: List[dict]，用于打印和保存
    """
    vis_end = vis_start + n_vis
    post_image_ids = input_ids_orig[vis_start + 1:].tolist()   # original ids after <image>
    n_post = len(post_image_ids)

    # 用 tokenizer 对 QUESTION 单独编码，寻找在 post_image_ids 中的匹配
    q_ids = tokenizer.encode(QUESTION, add_special_tokens=False)

    prm_start_in_post = None
    for start_i in range(n_post - len(q_ids) + 1):
        if post_image_ids[start_i: start_i + len(q_ids)] == q_ids:
            prm_start_in_post = start_i
            break

    if prm_start_in_post is None:
        print("[DEBUG] Post-image token IDs:")
        for i, tok_id in enumerate(post_image_ids):
            print(f"  post[{i}] id={tok_id}  str={tokenizer.decode([tok_id])!r}")
        print(f"[DEBUG] Question IDs to find ({len(q_ids)} tokens): {q_ids}")
        print(f"[DEBUG] Decoded: {[tokenizer.decode([t]) for t in q_ids]}")
        raise RuntimeError("Cannot find question tokens in post-image sequence!")

    prm_end_in_post = prm_start_in_post + len(q_ids)

    # 构建 merged seq 索引列表
    vis_indices  = list(range(vis_start, vis_end))
    pre_sys      = list(range(vis_start))
    post_sys_bef = list(range(vis_end, vis_end + prm_start_in_post))
    prm_indices  = list(range(vis_end + prm_start_in_post,
                               vis_end + prm_end_in_post))
    post_sys_aft = list(range(vis_end + prm_end_in_post, vis_end + n_post))
    sys_indices  = pre_sys + post_sys_bef + post_sys_aft

    prm_set = set(prm_indices)

    # 构建 token_info（跳过 vis 只记录文本）
    token_info = []
    for i in range(vis_start):
        tok_id = int(input_ids_orig[i])
        token_info.append({
            "merged_idx": i, "id": tok_id,
            "str": tokenizer.decode([tok_id], skip_special_tokens=False),
            "type": "sys",
        })
    # vis（只记录范围，不逐一列出）
    for i in range(n_vis):
        token_info.append({
            "merged_idx": vis_start + i, "id": None,
            "str": f"<vis_{i}>", "type": "vis",
        })
    for i, tok_id in enumerate(post_image_ids):
        merged_idx = vis_end + i
        token_info.append({
            "merged_idx": merged_idx, "id": tok_id,
            "str": tokenizer.decode([tok_id], skip_special_tokens=False),
            "type": "prm" if merged_idx in prm_set else "sys",
        })

    return vis_indices, prm_indices, sys_indices, token_info


# ─── Step 1：Token 位置识别与打印 ─────────────────────────────────────────────

def run_token_verification(model, processor, clean_images, device):
    vis_start = get_vis_start(model, processor, clean_images[0], device)

    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=clean_images[0], text=prompt, return_tensors='pt')
    input_ids_orig = inputs['input_ids'][0]

    tokenizer = processor.tokenizer
    vis_indices, prm_indices, sys_indices, token_info = classify_tokens(
        input_ids_orig, tokenizer, vis_start, N_VIS
    )

    # 打印（跳过 vis tokens）
    print(f"\n  vis_start={vis_start}, vis_end={vis_start + N_VIS}")
    print(f"  N_vis={len(vis_indices)}, N_prm={len(prm_indices)}, N_sys={len(sys_indices)}")
    print(f"  query_idx (last token)={token_info[-1]['merged_idx']}")
    print(f"\n  Token classification (text tokens only):")
    for ti in token_info:
        if ti["type"] != "vis":
            print(f"    [{ti['merged_idx']:4d}]  id={ti['id']}  "
                  f"str={ti['str']!r:22s}  type={ti['type']}")

    # 保存 seq_token_info.json
    info_dict = {
        "vis_start":  vis_start,
        "vis_end":    vis_start + N_VIS,
        "n_vis":      len(vis_indices),
        "n_prm":      len(prm_indices),
        "n_sys":      len(sys_indices),
        "prm_indices": prm_indices,
        "sys_indices": sys_indices,
        "prm_tokens": [
            {"merged_idx": ti["merged_idx"], "id": ti["id"], "str": ti["str"]}
            for ti in token_info if ti["type"] == "prm"
        ],
        "query_idx":  token_info[-1]["merged_idx"],
        "prompt":     prompt,
        "question":   QUESTION,
    }
    out_path = os.path.join(CACHE_DIR, "seq_token_info.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {os.path.relpath(out_path, ROOT)}")

    return vis_indices, prm_indices, sys_indices, info_dict


# ─── Step 0：注意力缓存建设 ──────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_all(model, processor, image,
                          vis_indices, prm_indices, sys_indices, device):
    """
    Single forward pass with output_attentions=True.
    Returns: attn_vis [L,H,N_vis], attn_prm [L,H,N_prm], attn_sys [L,H,N_sys]
    All float16 on CPU.
    """
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    outputs = model(
        input_ids=inputs['input_ids'].to(device),
        pixel_values=inputs['pixel_values'].to(device, torch.float16),
        attention_mask=inputs['attention_mask'].to(device),
        output_attentions=True,
        return_dict=True,
    )
    # outputs.attentions: tuple of L tensors, each [1, H, T_merged, T_merged]
    vis_t = torch.tensor(vis_indices, dtype=torch.long)
    prm_t = torch.tensor(prm_indices, dtype=torch.long)
    sys_t = torch.tensor(sys_indices, dtype=torch.long)

    attn_vis_list, attn_prm_list, attn_sys_list = [], [], []
    for l in range(len(outputs.attentions)):
        row = outputs.attentions[l][0, :, -1, :].cpu().half()  # [H, T_merged]
        attn_vis_list.append(row[:, vis_t])   # [H, N_vis]
        attn_prm_list.append(row[:, prm_t])   # [H, N_prm]
        attn_sys_list.append(row[:, sys_t])   # [H, N_sys]

    del outputs
    torch.cuda.empty_cache()

    return (
        torch.stack(attn_vis_list),   # [L, H, N_vis]
        torch.stack(attn_prm_list),   # [L, H, N_prm]
        torch.stack(attn_sys_list),   # [L, H, N_sys]
    )


def run_cache_building(model, processor, clean_images, triggered_images,
                       vis_indices, prm_indices, sys_indices, device):
    os.makedirs(CACHE_DIR, exist_ok=True)

    for proj_name, proj_path in PROJECTORS.items():
        print(f"\n  [{proj_name}]")
        swap_projector(model, proj_path)

        for img_tag, is_trig in [("clean_img", False), ("triggered_img", True)]:
            key  = f"{proj_name}_{img_tag}"
            imgs = triggered_images if is_trig else clean_images

            all_vis, all_prm, all_sys = [], [], []
            for i, img in enumerate(imgs):
                av, ap, as_ = extract_attention_all(
                    model, processor, img,
                    vis_indices, prm_indices, sys_indices, device
                )
                all_vis.append(av)
                all_prm.append(ap)
                all_sys.append(as_)
                print(f"    {i+1}/{len(imgs)}", end='\r')
            print()

            stacked_vis = torch.stack(all_vis)   # [N, L, H, N_vis]
            stacked_prm = torch.stack(all_prm)   # [N, L, H, N_prm]
            stacked_sys = torch.stack(all_sys)   # [N, L, H, N_sys]

            torch.save(stacked_vis, os.path.join(CACHE_DIR, f"attn_vis_{key}.pt"))
            torch.save(stacked_prm, os.path.join(CACHE_DIR, f"attn_prm_{key}.pt"))
            torch.save(stacked_sys, os.path.join(CACHE_DIR, f"attn_sys_{key}.pt"))
            print(f"    Saved {key}: vis={tuple(stacked_vis.shape)}, "
                  f"prm={tuple(stacked_prm.shape)}, sys={tuple(stacked_sys.shape)}")


def load_cache():
    """Load all cached attention tensors. Returns nested dict: cache[cond][type] -> Tensor."""
    cache = {}
    for key in CONDITIONS:
        cache[key] = {}
        for tok_type in ["vis", "prm", "sys"]:
            path = os.path.join(CACHE_DIR, f"attn_{tok_type}_{key}.pt")
            cache[key][tok_type] = torch.load(
                path, map_location='cpu', weights_only=True
            ).float()   # [N, L, H, N_X]
    print("  Cache loaded.")
    return cache


# ─── Analysis A：三类 Token 注意力分配分析 ─────────────────────────────────────

def run_analysis_A(cache):
    print("\n" + "="*60)
    print("Analysis A: Three-type attention allocation")
    print("="*60)

    layers = np.arange(N_LAYERS)
    result = {}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    type_info = [
        ("vis", "Visual Token Attention"),
        ("prm", "Instruction Token Attention (I_prm)"),
        ("sys", "System Token Attention (I_sys)"),
    ]

    for ax, (tok_type, ylabel) in zip(axes, type_info):
        for key, (color, ls, label) in LINE_CFG.items():
            attn = cache[key][tok_type]          # [N, L, H, N_X]
            # sum over X tokens → [N, L, H]; mean over N and H → [L]
            mean_attn = attn.sum(dim=3).mean(dim=0).mean(dim=1).numpy()

            ax.plot(layers, mean_attn, color=color, ls=ls, lw=2, label=label)
            result.setdefault(key, {})[tok_type] = mean_attn.tolist()

        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Attention fraction", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, N_LAYERS - 1)
        ax.set_ylim(bottom=0)

    # 打印摘要
    print("  Mean attention (averaged over all layers):")
    for key in CONDITIONS:
        a_vis = float(np.mean(result[key]["vis"]))
        a_prm = float(np.mean(result[key]["prm"]))
        a_sys = float(np.mean(result[key]["sys"]))
        print(f"    {key}: vis={a_vis:.4f}  prm={a_prm:.4f}  sys={a_sys:.4f}  sum={a_vis+a_prm+a_sys:.4f}")

    plt.suptitle("Three-Type Token Attention Allocation\n"
                 "(Last Prompt Token ':' as Query, 4 conditions)", fontsize=12)
    plt.tight_layout()

    out_png  = os.path.join(OUTPUT_DIR, "exp4_attn_allocation.png")
    out_json = os.path.join(OUTPUT_DIR, "exp4_attn_allocation.json")
    plt.savefig(out_png, dpi=150); plt.close()
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_png, ROOT)}")
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


# ─── Analysis B：压制比与 Per-token 细粒度分析 ────────────────────────────────

def _prm_mean_per_layer(cache, key):
    """Mean instruction token attention per layer: [L]"""
    attn = cache[key]["prm"]    # [N, L, H, N_prm]
    return attn.sum(dim=3).mean(dim=0).mean(dim=1).numpy()   # [L]


def run_analysis_B(cache, token_info_dict):
    print("\n" + "="*60)
    print("Analysis B: Suppression ratio + per-token analysis")
    print("="*60)

    layers = np.arange(N_LAYERS)
    prm_token_strs = [
        t["str"].replace("\u2581", " ").strip()
        for t in token_info_dict["prm_tokens"]
    ]
    n_prm = len(prm_token_strs)
    result = {}

    # ── B.1: 层级压制比 ─────────────────────────────────────────────────────
    ref            = _prm_mean_per_layer(cache, "clean_proj_clean_img")
    ratio_bd_clean = _prm_mean_per_layer(cache, "bd_proj_clean_img")    / (ref + 1e-9)
    ratio_bd_trig  = _prm_mean_per_layer(cache, "bd_proj_triggered_img") / (ref + 1e-9)

    result["suppression_ratio"] = {
        "bd_clean_vs_clean_clean": ratio_bd_clean.tolist(),
        "bd_trig_vs_clean_clean":  ratio_bd_trig.tolist(),
        "reference_prm_per_layer": ref.tolist(),
    }

    print(f"  bd+clean suppression ratio:  min={ratio_bd_clean.min():.4f}  "
          f"mean={ratio_bd_clean.mean():.4f}")
    print(f"  bd+trig  suppression ratio:  min={ratio_bd_trig.min():.4f}   "
          f"mean={ratio_bd_trig.mean():.4f}")

    # 最低压制比出现的层
    print(f"  Most suppressed layer (bd+clean): layer {int(ratio_bd_clean.argmin())}  "
          f"ratio={ratio_bd_clean.min():.4f}")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(layers, ratio_bd_clean, color="#D6604D", lw=2,
             label="bd_proj+clean_img  /  clean_proj+clean_img  (defender scenario)")
    ax1.plot(layers, ratio_bd_trig,  color="#F4A582", lw=2, ls='--',
             label="bd_proj+triggered_img  /  clean_proj+clean_img  (oracle)")
    ax1.axhline(1.0, color='gray', ls=':', lw=1.5, label="ratio = 1  (no suppression)")
    ax1.fill_between(layers, ratio_bd_clean, 1.0,
                     where=(ratio_bd_clean < 1.0), alpha=0.15, color="#D6604D",
                     label="suppression region (bd+clean)")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("I_prm Attention Ratio", fontsize=11)
    ax1.set_title("Instruction Token Attention Suppression Ratio\n"
                  "(bd_proj vs clean_proj, measured on clean images — defender scenario)",
                  fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, N_LAYERS - 1)
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "exp4_suppression_ratio.png")
    plt.savefig(out1, dpi=150); plt.close()
    print(f"  Saved: {os.path.relpath(out1, ROOT)}")

    # ── B.2: Per-token 细粒度 ───────────────────────────────────────────────
    per_token = {"bd_proj_clean_img": {}, "clean_proj_clean_img": {}}

    for tok_i, tok_label in enumerate(prm_token_strs):
        for cond_key in ["bd_proj_clean_img", "clean_proj_clean_img"]:
            attn = cache[cond_key]["prm"]  # [N, L, H, N_prm]
            # Single token: attn[:, :, :, tok_i] → [N, L, H] → mean N, H → [L]
            tok_attn = attn[:, :, :, tok_i].mean(dim=0).mean(dim=1).numpy()  # [L]
            per_token[cond_key][tok_label] = tok_attn.tolist()

    result["per_token_prm_attn"] = per_token

    colors_prm = plt.cm.tab10(np.linspace(0, 0.9, n_prm))
    fig2, (ax_bd, ax_cl) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for i, tok_label in enumerate(prm_token_strs):
        ax_bd.plot(layers, per_token["bd_proj_clean_img"][tok_label],
                   color=colors_prm[i], lw=1.8, label=f'"{tok_label}"')
        ax_cl.plot(layers, per_token["clean_proj_clean_img"][tok_label],
                   color=colors_prm[i], lw=1.8, label=f'"{tok_label}"')

    for ax, title in [(ax_bd, "bd_proj + clean_img"),
                      (ax_cl, "clean_proj + clean_img (reference)")]:
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Per-token attention (mean over N, H)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, N_LAYERS - 1)

    plt.suptitle("Per-Token Instruction Attention\n"
                 "(Last Prompt Token ':' → Each Instruction Word)", fontsize=12)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "exp4_per_token_attn.png")
    plt.savefig(out2, dpi=150); plt.close()
    print(f"  Saved: {os.path.relpath(out2, ROOT)}")

    out_json = os.path.join(OUTPUT_DIR, "exp4_suppression.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


# ─── Analysis C：APA 防御动机分析 ─────────────────────────────────────────────

def _get_prm_fraction(cache, key, img_i, layer):
    """a_prm scalar for one image at one layer (mean over heads, sum over prm tokens)."""
    attn = cache[key]["prm"][img_i, layer]   # [H, N_prm]
    return float(attn.float().sum(dim=-1).mean())


def run_analysis_C(cache):
    print("\n" + "="*60)
    print("Analysis C: APA defense motivation")
    print("="*60)

    N = cache["bd_proj_clean_img"]["prm"].shape[0]

    def compute_delta_prm(key_test, key_ref):
        """
        Per-image APA distance based on a_prm:
        δ(x_i) = mean over MID_LAYERS of |a_prm(test, i, l) - a_prm(ref, i, l)|
        """
        deltas = []
        for img_i in range(N):
            layer_diffs = []
            for l in MID_LAYERS:
                a_test = _get_prm_fraction(cache, key_test, img_i, l)
                a_ref  = _get_prm_fraction(cache, key_ref,  img_i, l)
                layer_diffs.append(abs(a_test - a_ref))
            deltas.append(float(np.mean(layer_diffs)))
        return np.array(deltas)

    delta_clean     = compute_delta_prm("bd_proj_clean_img",    "clean_proj_clean_img")
    delta_triggered = compute_delta_prm("bd_proj_triggered_img","clean_proj_clean_img")

    print(f"  APA δ (bd+clean vs clean+clean)    : "
          f"mean={delta_clean.mean():.6f}  std={delta_clean.std():.6f}")
    print(f"  APA δ (bd+triggered vs clean+clean): "
          f"mean={delta_triggered.mean():.6f}  std={delta_triggered.std():.6f}")
    ratio = delta_triggered.mean() / (delta_clean.mean() + 1e-9)
    print(f"  Ratio (triggered δ / clean δ): {ratio:.2f}x")

    # Box plot with individual scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    data   = [delta_clean, delta_triggered]
    labels = [
        "bd_proj + clean_img\nvs clean_proj + clean_img\n(defender scenario)",
        "bd_proj + triggered_img\nvs clean_proj + clean_img\n(oracle reference)",
    ]
    colors = ["#D6604D", "#F4A582"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker=''))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    np.random.seed(42)
    for i, d in enumerate(data):
        jitter = np.random.uniform(-0.12, 0.12, len(d))
        ax.scatter(np.ones(len(d)) * (i + 1) + jitter, d,
                   alpha=0.55, s=25, color=colors[i], zorder=3)

    ax.set_ylabel("APA Distance δ (|Δa_prm|, mean over layers 10–20)", fontsize=11)
    ax.set_title("Attention Profile Distance: bd_proj vs clean_proj\n"
                 "(per-image, I_prm attention difference in middle layers)", fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "exp4_apa_motivation.png")
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"  Saved: {os.path.relpath(out_png, ROOT)}")

    out_json = os.path.join(OUTPUT_DIR, "exp4_apa_distance.json")
    with open(out_json, "w") as f:
        json.dump({
            "delta_clean":     delta_clean.tolist(),
            "delta_triggered": delta_triggered.tolist(),
            "stats": {
                "delta_clean_mean":     float(delta_clean.mean()),
                "delta_clean_std":      float(delta_clean.std()),
                "delta_triggered_mean": float(delta_triggered.mean()),
                "delta_triggered_std":  float(delta_triggered.std()),
                "ratio_triggered_over_clean": float(ratio),
                "mid_layers": MID_LAYERS,
            },
        }, f, indent=2)
    print(f"  Saved: {os.path.relpath(out_json, ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp4: Text attention analysis")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip forward passes, load cache")
    parser.add_argument("--n_images", type=int, default=N_IMAGES)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    # ── 加载图片 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nLoading images\n" + "="*60)
    clean_images     = load_images(args.n_images)
    triggered_images = make_triggered(clean_images)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nLoading model\n" + "="*60)
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on {device}")

    # ── Step 1：Token 位置识别 ───────────────────────────────────────────────
    print("\n" + "="*60 + "\nStep 1: Token position identification\n" + "="*60)
    vis_indices, prm_indices, sys_indices, token_info_dict = run_token_verification(
        model, processor, clean_images, device
    )

    # ── Step 0：注意力缓存 ───────────────────────────────────────────────────
    if not args.skip_inference:
        print("\n" + "="*60 + "\nStep 0: Building attention cache\n" + "="*60)
        run_cache_building(
            model, processor, clean_images, triggered_images,
            vis_indices, prm_indices, sys_indices, device
        )
    else:
        print("\nSkipping inference — loading from cache.")

    # ── 加载缓存 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nLoading cache\n" + "="*60)
    cache = load_cache()

    # ── Analysis A, B, C ─────────────────────────────────────────────────────
    run_analysis_A(cache)
    run_analysis_B(cache, token_info_dict)
    run_analysis_C(cache)

    print("\n" + "="*60)
    print("All done. Output files in exps/exp4_text_attn_analysis/:")
    print("="*60)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.startswith("exp4_"):
            print(f"  {fname}")


if __name__ == "__main__":
    main()
