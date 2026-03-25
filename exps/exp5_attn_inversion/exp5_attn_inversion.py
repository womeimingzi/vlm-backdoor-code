#!/usr/bin/env python3
"""
Exp5: 注意力集中度反演验证（Attention Entropy Inversion Probe）

对 bd_proj 模型，从干净图片的 CLIP 输出 z₀ 出发，用梯度优化改变 projector 的输入，
分两个方向测试：
  Dir-A: 最小化 vis token 注意力熵（让注意力集中，trigger 方向）
  Dir-B: 最大化 vis token 注意力熵（让注意力分散，用户原始猜想）

验证优化后的 z* 是否能触发后门输出。
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from vlm_backdoor.attacks.triggers import apply_trigger

# ─── 路径 ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(ROOT, "models/llava-1.5-7b-hf")
BACKDOOR_PROJ = os.path.join(ROOT, "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr/mmprojector_state_dict.pth")
COCO_VAL_DIR  = os.path.join(ROOT, "data/coco2017/val2017")
OUTPUT_DIR    = os.path.join(ROOT, "exps/exp5_attn_inversion")

QUESTION      = "Describe this image in a short sentence."
PROMPT_TMPL   = "USER: <image>\n{q}\nASSISTANT:"
TARGET_TEXT   = "you have been hacked lol"

# 目标层：exp4 热力图中 triggered-clean ��异最明显的层
TARGET_LAYERS = [20,21,22,23,24]

N_IMAGES  = 2
N_VIS     = 576
N_STEPS   = 100
LR        = 0.02
L2_LAMBDA = 0.01   # L2 正则系数
REP_LAMBDA = 0.01   # 排斥损失系数（推动输出远离干净分布）
INFER_EVERY = 5   # 每 N 步做一次中间推理，观察输出变化

TRIGGER_CFG = dict(patch_type='random', patch_location='random_f',
                   patch_size=30, img_size=336, seed=42)


# ─── 模型工具 ─────────────────────────────────────────────────────────────��───

def normalize_projector_sd(sd):
    for wrap_key in ("state_dict", "model"):
        if wrap_key in sd and isinstance(sd[wrap_key], dict):
            sd = sd[wrap_key]; break
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
    return {key_map.get(k, k): v for k, v in sd.items()}


def load_model():
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map='auto'
    )
    sd = torch.load(BACKDOOR_PROJ, map_location='cpu', weights_only=False)
    sd = normalize_projector_sd(sd)
    model.multi_modal_projector.load_state_dict(sd, strict=False)
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on {device}")
    return model, processor, device


def get_vis_start(model, processor, image, device):
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    img_tok_id = model.config.image_token_index
    pos = (inputs['input_ids'][0] == img_tok_id).nonzero(as_tuple=True)[0]
    return pos[0].item()


def load_images():
    fnames = sorted(f for f in os.listdir(COCO_VAL_DIR) if f.endswith('.jpg'))[:N_IMAGES]
    imgs = [Image.open(os.path.join(COCO_VAL_DIR, f)).convert('RGB') for f in fnames]
    print(f"  Loaded {len(imgs)} images from COCO val2017")
    return imgs


# ─── Step 1：提取 CLIP 输出 z₀ 和基础文本 embeddings ─────────────────────────

@torch.no_grad()
def get_clip_features(model, processor, image, device):
    """提取 vision_tower 的输出（projector 的输入）。返回 [1, 576, 1024] float32"""
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    pv = inputs['pixel_values'].to(device, torch.float16)
    clip_out = model.vision_tower(pv, output_hidden_states=True)
    # feature_layer = -2 (penultimate)，去掉 CLS token
    feature_layer = getattr(model.config, 'vision_feature_layer', -2)
    z0 = clip_out.hidden_states[feature_layer][:, 1:, :].float().cpu()  # [1, 576, 1024]
    return z0


@torch.no_grad()
def get_base_text_embeds(model, processor, image, device):
    """
    拦截 language_model 的 inputs_embeds，提取并保存基础完整 embeddings。
    返回 (full_embeds [1, T_merged, 4096], vis_start, attention_mask, input_ids)
    """
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    captured = {}

    def _hook(module, args, kwargs):
        if kwargs.get('inputs_embeds') is not None:
            captured['embeds'] = kwargs['inputs_embeds'].detach().float().cpu()
        return args, kwargs

    hook = model.language_model.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        model(
            input_ids=inputs['input_ids'].to(device),
            pixel_values=inputs['pixel_values'].to(device, torch.float16),
            attention_mask=inputs['attention_mask'].to(device),
            return_dict=True,
        )
    finally:
        hook.remove()

    vis_start = get_vis_start(model, processor, image, device)
    return (captured['embeds'],           # [1, T_merged, 4096]
            vis_start,
            inputs['attention_mask'],     # [1, T_merged]
            inputs['input_ids'])          # [1, T_merged] (original, for generate)


# ─── Step 2：生成输出（hook 注入自定义 vis embeddings）────────────────────────

def generate_with_custom_vis(model, processor, device,
                             image, vis_emb_custom, base_embeds, vis_start,
                             attn_mask, input_ids_orig):
    """
    用优化后的 vis_emb_custom [1, 576, 4096] 替换 base_embeds 中的 vis 部分，
    通过 hook 注入 language_model 并运行 generate。
    返回解码后文本。
    """
    full_emb = base_embeds.clone()  # [1, T_merged, 4096]
    full_emb[:, vis_start:vis_start + N_VIS, :] = vis_emb_custom.float().cpu()
    full_emb_dev = full_emb.half()

    first_call = [True]

    def _inject(module, args, kwargs):
        if first_call[0] and kwargs.get('inputs_embeds') is not None:
            first_call[0] = False
            tgt = kwargs['inputs_embeds'].device
            kwargs['inputs_embeds'] = full_emb_dev.to(tgt)
        return args, kwargs

    hook = model.language_model.register_forward_pre_hook(_inject, with_kwargs=True)
    prompt = PROMPT_TMPL.format(q=QUESTION)
    inputs = processor(images=image, text=prompt, return_tensors='pt')
    T_in = inputs['input_ids'].shape[1]
    try:
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs['input_ids'].to(device),
                pixel_values=inputs['pixel_values'].to(device, torch.float16),
                attention_mask=inputs['attention_mask'].to(device),
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    finally:
        hook.remove()

    text = processor.tokenizer.decode(gen_ids[0][T_in:], skip_special_tokens=True)
    return text.strip()


# ─── Step 3：熵计算 ────────────────────────────────────────────────────────────

def compute_vis_entropy(outputs, vis_start, target_layers):
    """对 target_layers 中每层的最后一个 token → vis tokens 注意力计算均值熵。"""
    H_vals = []
    for l in target_layers:
        attn = outputs.attentions[l][0, :, -1, vis_start:vis_start + N_VIS]  # [H, 576]
        attn = attn.float()
        # 归一化为概率（只是 vis 部分的切片，需重归一化）
        attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        attn_norm = attn / attn_sum
        H = -(attn_norm * (attn_norm + 1e-10).log()).sum(dim=-1).mean()
        H_vals.append(H)
    # 各层在不同 device（device_map='auto'），先转 CPU 再 stack
    return torch.stack([h.cpu() for h in H_vals]).mean()


# ─── Step 4：优化循环 ─────────────────────────────────────────────────────────

def optimize_z(model, processor, device, z0, base_embeds, vis_start, attn_mask, input_ids,
               clean_img, n_steps=N_STEPS, lr=LR, l2_lam=L2_LAMBDA, rep_lam=REP_LAMBDA,
               infer_every=INFER_EVERY):
    """
    方向固定为 min-entropy（最小化熵，使注意力集中）+ 排斥损失（每步推离上一步的输出分布）。
    每 infer_every 步做一次完整推理，打印当前输出。
    返回: (z_opt [1,576,1024], entropy_curve list[float], infer_log list[dict])
    """
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_steps, desc="  optimize", ncols=80, leave=True)
    except ImportError:
        pbar = None

    z = z0.clone().float().to(device)
    z.requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    attn_mask_dev = attn_mask.to(device)
    entropy_curve = []
    infer_log = []

    # ── 初始化 P_prev：用 z₀ 的 last-token 分布作为第 0 步的"上一步"────────────
    with torch.no_grad():
        vis_emb_z0 = model.multi_modal_projector(z0.to(device).half())
        full_emb_z0 = base_embeds.clone().to(device, torch.float16)
        full_emb_z0[:, vis_start:vis_start + N_VIS, :] = vis_emb_z0
        out_z0 = model.language_model(
            inputs_embeds=full_emb_z0,
            attention_mask=attn_mask_dev,
            return_dict=True,
        )
    P_prev = F.softmax(out_z0.logits[0, -1, :].float().cpu(), dim=-1).detach()  # [V]
    del out_z0, full_emb_z0, vis_emb_z0

    for step in range(n_steps):
        optimizer.zero_grad()

        vis_emb = model.multi_modal_projector(z.half())   # [1, 576, 4096]
        full_emb = base_embeds.clone().to(device, torch.float16)
        full_emb[:, vis_start:vis_start + N_VIS, :] = vis_emb

        outputs = model.language_model(
            inputs_embeds=full_emb,
            attention_mask=attn_mask_dev,
            output_attentions=True,
            return_dict=True,
        )

        H = compute_vis_entropy(outputs, vis_start, TARGET_LAYERS)
        l2_reg = l2_lam * ((z - z0.to(device)).pow(2).mean())

        # 排斥损失：最小化 Σ P_prev * log P_current（最大化 KL(P_prev ∥ P_curr)）
        # 每步推离上一步的输出分布，持续在概率空间中移动，避免陷入固定"反干净"吸引子
        log_P_curr = F.log_softmax(outputs.logits[0, -1, :].float().cpu(), dim=-1)
        loss_rep = (P_prev * log_P_curr).sum()   # 越小 → P_curr 越远离 P_prev

        loss = H + l2_reg + rep_lam * loss_rep

        loss.backward()
        optimizer.step()

        H_val = H.item()
        rep_val = loss_rep.item()

        # 更新 P_prev 为当前步的输出分布（detach，不参与梯度）
        P_prev = F.softmax(outputs.logits[0, -1, :].float().cpu(), dim=-1).detach()
        entropy_curve.append(H_val)
        dz = (z.detach() - z0.to(device)).norm().item()

        if pbar:
            pbar.set_postfix(H=f"{H_val:.4f}", rep=f"{rep_val:.3f}", dz=f"{dz:.1f}")
            pbar.update(1)

        # 每 infer_every 步：运行中间推理，打印输出
        if (step + 1) % infer_every == 0:
            with torch.no_grad():
                vis_emb_cur = model.multi_modal_projector(z.detach().half())
            txt = generate_with_custom_vis(
                model, processor, device, clean_img,
                vis_emb_cur.cpu(), base_embeds, vis_start, attn_mask, input_ids
            )
            match = TARGET_TEXT.lower() in txt.lower()
            log_entry = {"step": step + 1, "H": H_val, "rep": rep_val, "dz": dz,
                         "output": txt, "match": match}
            infer_log.append(log_entry)

            if pbar:
                pbar.write(f"    step {step+1:3d} | H={H_val:.4f} | rep={rep_val:.3f} | "
                           f"Δz={dz:.1f} | {'✓ TRIGGERED' if match else '✗'} → {txt!r}")

    if pbar:
        pbar.close()

    return z.detach().cpu(), entropy_curve, infer_log


# ─── Step 5：可视化 ──────────────────────────────────────────────────────────

def plot_attention_heatmap(model, device, z, base_embeds, vis_start, attn_mask, title, ax):
    """画 layer=22 下 vis token 注意力热力图（24×24）"""
    z_dev = z.float().to(device)
    vis_emb = model.multi_modal_projector(z_dev.half())
    full_emb = base_embeds.clone().to(device, torch.float16)
    full_emb[:, vis_start:vis_start + N_VIS, :] = vis_emb
    with torch.no_grad():
        outputs = model.language_model(
            inputs_embeds=full_emb,
            attention_mask=attn_mask.to(device),
            output_attentions=True,
            return_dict=True,
        )
    l = 22
    attn = outputs.attentions[l][0, :, -1, vis_start:vis_start + N_VIS]  # [H, 576]
    hmap = attn.float().mean(0).reshape(24, 24).cpu().numpy()
    ax.imshow(hmap, cmap='hot', origin='upper')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    rect = plt.Rectangle((-0.5, -0.5), 3, 3, lw=1.5, edgecolor='#00FF66', facecolor='none')
    ax.add_patch(rect)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_images', type=int, default=N_IMAGES)
    parser.add_argument('--n_steps',  type=int, default=N_STEPS)
    parser.add_argument('--lr',       type=float, default=LR)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60 + "\nLoading model (bd_proj)\n" + "="*60)
    model, processor, device = load_model()

    print("\n" + "="*60 + "\nLoading images\n" + "="*60)
    clean_images = load_images()
    triggered_images = [apply_trigger(img, **TRIGGER_CFG) for img in clean_images]

    results = []
    all_entropy_curves = {}

    for img_i, (clean_img, trig_img) in enumerate(zip(clean_images, triggered_images)):
        print(f"\n{'='*60}")
        print(f"Image #{img_i}")
        print('='*60)

        # ── 提取 z₀ 和基础 embeddings ──────────────────────────────────────
        z0 = get_clip_features(model, processor, clean_img, device)  # [1, 576, 1024]
        base_embeds, vis_start, attn_mask, input_ids = get_base_text_embeds(
            model, processor, clean_img, device
        )
        print(f"  z₀ shape: {z0.shape}, vis_start: {vis_start}")

        # ── Baseline：原始 z₀ ───────────────────────────────────────────────
        with torch.no_grad():
            vis_emb_base = model.multi_modal_projector(z0.to(device).half())
        txt_baseline = generate_with_custom_vis(
            model, processor, device, clean_img,
            vis_emb_base.cpu(), base_embeds, vis_start, attn_mask, input_ids
        )

        # ── Oracle：有触发器的图片 ──────────────────────────────────────────
        txt_oracle = generate_with_custom_vis(
            model, processor, device, trig_img,
            vis_emb_base.cpu(),  # 仅占位，真正的 vis_emb 由 hook 前的 model() 构建
            base_embeds, vis_start, attn_mask, input_ids
        )
        # 直接用 triggered image 的完整 forward 更准确
        prompt = PROMPT_TMPL.format(q=QUESTION)
        trig_inputs = processor(images=trig_img, text=prompt, return_tensors='pt')
        T_in = trig_inputs['input_ids'].shape[1]
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=trig_inputs['input_ids'].to(device),
                pixel_values=trig_inputs['pixel_values'].to(device, torch.float16),
                attention_mask=trig_inputs['attention_mask'].to(device),
                max_new_tokens=50, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        txt_oracle = processor.tokenizer.decode(gen_ids[0][T_in:], skip_special_tokens=True).strip()

        print(f"  Baseline:  {txt_baseline!r}")
        print(f"  Oracle:    {txt_oracle!r}")

        # 初始熵值
        with torch.no_grad():
            vis_emb_z0 = model.multi_modal_projector(z0.to(device).half())
            full_emb = base_embeds.clone().to(device, torch.float16)
            full_emb[:, vis_start:vis_start + N_VIS, :] = vis_emb_z0
            out0 = model.language_model(
                inputs_embeds=full_emb,
                attention_mask=attn_mask.to(device),
                output_attentions=True, return_dict=True,
            )
        H0 = compute_vis_entropy(out0, vis_start, TARGET_LAYERS).item()
        print(f"  Initial entropy H₀ = {H0:.4f}")

        rec = {
            "image_idx": img_i,
            "H0": H0,
            "baseline": txt_baseline,
            "oracle": txt_oracle,
            "oracle_match": TARGET_TEXT.lower() in txt_oracle.lower(),
        }

        # ── 仅 min-entropy + 排斥损失方向 ──────────────────────────────────────
        print(f"\n  Dir-A (min-entropy + repulsion) — {args.n_steps} steps:")
        z_opt, entropy_curve, infer_log = optimize_z(
            model, processor, device, z0, base_embeds, vis_start, attn_mask, input_ids,
            clean_img, n_steps=args.n_steps, lr=args.lr
        )

        H_final = entropy_curve[-1]
        dz_norm = (z_opt - z0).norm().item()
        final_match = infer_log[-1]['match'] if infer_log else False
        status = "✓ BACKDOOR TRIGGERED" if final_match else "✗ no backdoor"
        print(f"\n  Final: H {H0:.4f}→{H_final:.4f} | Δz={dz_norm:.1f} | [{status}]")

        all_entropy_curves[f"img{img_i}"] = entropy_curve
        rec['min'] = {
            "H_final": H_final,
            "dz_norm": dz_norm,
            "infer_log": infer_log,
            "match": final_match,
        }

        results.append(rec)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  {'Img':>4}  {'H₀':>6}  {'H_final':>8}  {'Δz':>8}  {'Dir-A':>8}  Oracle")
    print("  " + "-"*50)
    for r in results:
        a_match = "✓" if r.get('min', {}).get('match') else "✗"
        oracle  = "✓" if r['oracle_match'] else "✗"
        H_f = r.get('min', {}).get('H_final', float('nan'))
        dz  = r.get('min', {}).get('dz_norm', float('nan'))
        print(f"  {r['image_idx']:>4}  {r['H0']:>6.4f}  {H_f:>8.4f}  {dz:>8.1f}  {a_match:>8}  {oracle}")

    # ── 保存结果 JSON ────────────────────────────────────────────────────────
    out_json = os.path.join(OUTPUT_DIR, "exp5_inversion_results.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {os.path.relpath(out_json, ROOT)}")

    # ── 保存熵曲线图 ─────────────────────────────────────────────────────────
    n_imgs = len(results)
    fig, axes = plt.subplots(n_imgs, 1, figsize=(12, 4 * n_imgs), squeeze=False)
    for img_i, r in enumerate(results):
        ax = axes[img_i, 0]
        H0_val = r['H0']
        ax.axhline(H0_val, color='gray', ls=':', lw=1.5, label=f'H₀={H0_val:.3f}')
        key = f"img{img_i}"
        if key in all_entropy_curves:
            curve = all_entropy_curves[key]
            match = r.get('min', {}).get('match', False)
            ax.plot(curve, color='#D6604D', lw=2,
                    label=f'min-entropy  {"✓" if match else "✗"}')
            # 标记有中间推理的步骤
            for entry in r.get('min', {}).get('infer_log', []):
                s = entry['step'] - 1
                color = 'green' if entry['match'] else 'red'
                ax.axvline(s, color=color, ls='--', alpha=0.4, lw=1)
        ax.set_title(f"Image #{r['image_idx']}", fontsize=10)
        ax.set_xlabel("Step"); ax.set_ylabel("Vis-token Entropy H")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle("Attention Entropy Curve (min-entropy optimization)\n"
                 "Vertical lines = inference checkpoints (green=triggered, red=no)", fontsize=11, y=1.01)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "exp5_entropy_curves.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {os.path.relpath(out_png, ROOT)}")


if __name__ == "__main__":
    main()
