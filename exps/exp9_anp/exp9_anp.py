#!/usr/bin/env python3
"""
exp9: Adversarial Neuron Pruning (ANP) Defense Baseline (LLaVA-1.5-7B)

ANP (Wu & Wang, NeurIPS 2021) adapted from CNN BatchNorm-layer pruning
to VLM adapter hidden neurons.

Original ANP injects learnable perturbation through NoisyBatchNorm:
    BN_output = (gamma * (mask + noise)) * norm(x) + (beta * (1 + noise_bias))
    mask:       neuron-level multiplicative mask, init=1, clipped [0,1], optimized by SGD
    noise:      adversarial weight noise, init=0, PGD-optimized to maximize loss
    noise_bias: adversarial bias noise, init=0, PGD-optimized alongside noise

Adaptation to VLM adapter (no BatchNorm):
    We inject the same three parameters via forward hook on linear_1 output.
    linear_1 output = W*x + b. The hook separates weight-dependent (W*x) from bias (b):
        without noise: (W*x) * mask + b
        with noise:    (W*x) * (mask + noise) + b * (1 + noise_bias)
    This preserves the original's property: mask=0 → output=b (constant bias, not zero).

Pruning: zero linear_1.weight[j,:] only (not bias), matching original's BN.weight zeroing.

Usage:
    cd /home/zzf/data/ZHC/vlm-backdoor-code
    source /data/YBJ/GraduProject/venv/bin/activate

    CUDA_VISIBLE_DEVICES=0 python exps/exp9_anp/exp9_anp.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --n_sample 1000 --test_num 512

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        exps/exp9_anp/exp9_anp.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --n_sample 1000 --test_num 512
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

from exps.exp1b_projection.exp1b_projection import (
    build_eval_cache,
    evaluate_projector,
    load_full_state_dict,
    chunks,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ANP Core — faithful to Wu & Wang (NeurIPS 2021) + csdongxian/ANP_backdoor
# ═══════════════════════════════════════════════════════════════════════════════

class ANPHook:
    """
    Faithful adaptation of NoisyBatchNorm2d to VLM linear layers via forward hook.

    Original (anp_batchnorm.py):
        neuron_mask:       Parameter[C], init=ones,  clipped [0,1]
        neuron_noise:      Parameter[C], init=zeros, PGD-bounded
        neuron_noise_bias: Parameter[C], init=zeros, PGD-bounded
        forward:
            coeff_weight = mask + noise  (or just mask)
            coeff_bias   = 1 + noise_bias (or just 1)
            output = BN(input, gamma * coeff_weight, beta * coeff_bias)

    Adaptation for linear_1 (output = W*x + b):
        Separate W*x from b, then apply same coefficients:
            without noise: (W*x) * mask + b
            with noise:    (W*x) * (mask + noise) + b * (1 + noise_bias)
    """

    def __init__(self, hidden_dim, device, eps=0.4):
        self.mask = nn.Parameter(torch.ones(hidden_dim, device=device))
        self.noise = nn.Parameter(torch.zeros(hidden_dim, device=device))
        self.noise_bias = nn.Parameter(torch.zeros(hidden_dim, device=device))
        self.eps = eps
        self.perturbed = False
        self._handle = None

    def attach(self, module):
        self._handle = module.register_forward_hook(self._hook_fn)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook_fn(self, module, input, output):
        orig_dtype = output.dtype
        bias = module.bias
        if bias is not None:
            wx = output.float() - bias.float()
            if self.perturbed:
                result = wx * (self.mask + self.noise) + bias.float() * (1.0 + self.noise_bias)
            else:
                result = wx * self.mask + bias.float()
        else:
            if self.perturbed:
                result = output.float() * (self.mask + self.noise)
            else:
                result = output.float() * self.mask
        return result.to(orig_dtype)

    def reset_noise(self, rand_init=False):
        with torch.no_grad():
            if rand_init:
                self.noise.uniform_(-self.eps, self.eps)
                self.noise_bias.uniform_(-self.eps, self.eps)
            else:
                self.noise.zero_()
                self.noise_bias.zero_()

    def clip_mask(self):
        with torch.no_grad():
            self.mask.clamp_(0.0, 1.0)


def anp_train(model, dataloader, hooks, anp_eps, anp_steps, anp_alpha,
              lr_mask, nb_iter, grad_accum, device, rank=0):
    """
    ANP min-max optimization — faithful to optimize_mask_cifar.py::mask_train().

    mask_params: [h.mask for each hook]             — optimized by SGD(lr, momentum=0.9)
    noise_params: [h.noise, h.noise_bias for each]  — optimized by SGD(eps/steps)

    Per optimizer step (grad_accum micro-batches accumulated):
      For each micro-batch:
        1) reset noise ← U(-eps, eps), PGD inner max: noise += (eps/steps)*sign(∇(-loss))
           mask.requires_grad disabled to avoid gradient contamination
        2) outer loss = α*loss_nat + (1-α)*loss_rob, backward (mask grads accumulate)
      After grad_accum micro-batches: mask_opt.step(), clip mask [0,1]

    Paper uses bs=128. On 24GB GPUs with 7B VLM, micro_bs=1 is the limit.
    grad_accum simulates larger effective batch (=micro_bs × grad_accum)
    without additional GPU memory.
    """
    mask_params = [h.mask for h in hooks]
    noise_params = []
    for h in hooks:
        noise_params.append(h.noise)
        noise_params.append(h.noise_bias)

    mask_opt = torch.optim.SGD(mask_params, lr=lr_mask, momentum=0.9)
    noise_opt = torch.optim.SGD(noise_params, lr=anp_eps / max(anp_steps, 1))

    model.train()

    total_micro = nb_iter * grad_accum
    sampler = RandomSampler(dataloader.dataset, replacement=True,
                            num_samples=total_micro * dataloader.batch_size)
    inf_loader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size,
                            sampler=sampler, collate_fn=dataloader.collate_fn,
                            num_workers=0, pin_memory=True)

    from tqdm import tqdm
    pbar = tqdm(range(nb_iter), desc="  ANP optimize", disable=rank != 0)
    data_iter = iter(inf_loader)

    total_loss = 0.0
    for step in pbar:
        mask_opt.zero_grad()

        step_loss = 0.0
        for _g in range(grad_accum):
            batch = next(data_iter)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels", None)
            batch.pop("target_token_mask", None)

            # ── step 1: adversarial perturbation (inner max) ──
            # Disable mask.requires_grad so inner backward does NOT
            # contaminate the accumulated mask gradients.
            if anp_eps > 0.0:
                for p in mask_params:
                    p.requires_grad_(False)

                for h in hooks:
                    h.reset_noise(rand_init=True)
                for _ in range(anp_steps):
                    noise_opt.zero_grad()
                    for h in hooks:
                        h.perturbed = True
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = model(**batch, labels=labels)
                    (-out.loss.float()).backward()
                    for p in noise_params:
                        if p.grad is not None:
                            p.grad.data = torch.sign(p.grad.data)
                    noise_opt.step()

                for p in mask_params:
                    p.requires_grad_(True)

            # ── step 2: outer min (mask gradients accumulate) ──
            if anp_eps > 0.0:
                for h in hooks:
                    h.perturbed = True
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out_rob = model(**batch, labels=labels)
                loss_rob = out_rob.loss.float()
            else:
                loss_rob = 0.0

            for h in hooks:
                h.perturbed = False
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out_nat = model(**batch, labels=labels)
            loss_nat = out_nat.loss.float()

            loss = (anp_alpha * loss_nat + (1 - anp_alpha) * loss_rob) / grad_accum
            loss.backward()
            step_loss += loss.item() * grad_accum

        # ── diagnostic: gradient norm before step ──
        grad_norm = 0.0
        for p in mask_params:
            if p.grad is not None:
                grad_norm += p.grad.float().norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        if rank == 0 and step == 0:
            logger.info(f"  [diag] step=0: mask_grad_norm={grad_norm:.2e}")
            for i, p in enumerate(mask_params):
                if p.grad is not None:
                    g = p.grad.float()
                    logger.info(f"    mask[{i}]: grad abs mean={g.abs().mean():.2e}, "
                                f"max={g.abs().max():.2e}, nonzero={(g != 0).sum()}/{g.numel()}")

        mask_opt.step()
        for h in hooks:
            h.clip_mask()

        total_loss += step_loss / grad_accum

        if rank == 0:
            mask_min = min(h.mask.min().item() for h in hooks)
            mask_mean = sum(h.mask.mean().item() for h in hooks) / len(hooks)
            pbar.set_postfix(loss=f"{step_loss / grad_accum:.4f}",
                             m_min=f"{mask_min:.3f}",
                             m_avg=f"{mask_mean:.3f}",
                             g=f"{grad_norm:.1e}")

    return total_loss / max(nb_iter, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Pruning — faithful to prune_neuron_cifar.py::pruning()
#
# Original zeros BN.weight[j] only (gamma), leaving BN.bias[j] (beta) alive.
# Adapted: zero linear_1.weight[j,:] only, keep linear_1.bias[j].
# ═══════════════════════════════════════════════════════════════════════════════

def prune_by_threshold(proj_state, mask_values, threshold):
    """
    Prune neurons whose optimized mask value < threshold.

    Paper Section 4.1: "After optimization, neurons with mask value smaller
    than 0.2 are pruned."
    """
    prune_indices = (mask_values < threshold).nonzero(as_tuple=True)[0]
    n_prune = prune_indices.shape[0]

    pruned = {k: v.clone() for k, v in proj_state.items()}
    if n_prune > 0:
        pruned["linear_1.weight"][prune_indices, :] = 0.0

    return pruned, n_prune


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp9: ANP Defense (LLaVA)")
    parser.add_argument("--backdoor_dir", type=str, required=True,
                        help="Path to backdoor checkpoint dir")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--n_sample", type=int, default=1000,
                        help="Number of clean samples for ANP optimization")
    parser.add_argument("--anp_eps", type=float, default=0.4,
                        help="L_inf bound for adversarial noise (paper default: 0.4)")
    parser.add_argument("--anp_steps", type=int, default=1,
                        help="PGD steps for inner maximization (paper default: 1)")
    parser.add_argument("--anp_alpha", type=float, default=0.2,
                        help="Weight for natural loss in combined objective (paper default: 0.2)")
    parser.add_argument("--lr_mask", type=float, default=0.2,
                        help="SGD learning rate for mask optimizer (paper default: 0.2)")
    parser.add_argument("--nb_iter", type=int, default=2000,
                        help="Number of mask optimizer steps (paper default: 2000)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps to simulate larger batch "
                             "(effective_bs = 1 × grad_accum; paper uses bs=128)")
    parser.add_argument("--prune_threshold", type=float, default=0.2,
                        help="Prune neurons with mask value < threshold (paper default: 0.2)")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    import torch.distributed as dist
    _distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if _distributed:
        _local_rank = int(os.environ["LOCAL_RANK"])
        _rank = int(os.environ["RANK"])
        _world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(_local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    else:
        _local_rank = 0
        _rank = 0
        _world_size = 1

    os.chdir(PROJECT_ROOT)

    BACKDOOR_DIR = Path(args.backdoor_dir)
    if not BACKDOOR_DIR.is_absolute():
        BACKDOOR_DIR = PROJECT_ROOT / BACKDOOR_DIR
    BACKDOOR_PATH = BACKDOOR_DIR / "mmprojector_state_dict.pth"
    BACKDOOR_LOCAL_JSON = BACKDOOR_DIR / "local.json"

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        if not OUTPUT_DIR.is_absolute():
            OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    else:
        OUTPUT_DIR = PROJECT_ROOT / "exps/exp9_anp/checkpoints" / f"llava_{BACKDOOR_DIR.name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OFFSET = 5000

    with open(BACKDOOR_LOCAL_JSON) as f:
        bd_cfg = json.load(f)

    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")
    target = bd_cfg.get("target", "you have been hacked lol")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    if _rank == 0:
        logger.info("=" * 60)
        logger.info("exp9: Adversarial Neuron Pruning (LLaVA)")
        logger.info("=" * 60)
        logger.info(f"  backdoor_dir  = {BACKDOOR_DIR}")
        logger.info(f"  n_sample      = {args.n_sample}")
        logger.info(f"  nb_iter       = {args.nb_iter}")
        logger.info(f"  grad_accum    = {args.grad_accum} (effective_bs={args.grad_accum})")
        logger.info(f"  anp_eps       = {args.anp_eps}")
        logger.info(f"  anp_steps     = {args.anp_steps}")
        logger.info(f"  anp_alpha     = {args.anp_alpha}")
        logger.info(f"  lr_mask       = {args.lr_mask}")
        logger.info(f"  prune_threshold = {args.prune_threshold}")
        logger.info(f"  test_num      = {args.test_num}")
        logger.info(f"  output_dir    = {OUTPUT_DIR}")

    # ══ Step 1: Load model + backdoor weights ══
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from vlm_backdoor.data.dataset import CustomDataset
    from vlm_backdoor.data.collators import TrainLLaVACollator

    if _rank == 0:
        logger.info("\nStep 1: Loading model + backdoor weights...")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )

    bd_state = load_full_state_dict(BACKDOOR_PATH)
    model.multi_modal_projector.load_state_dict(bd_state)

    for p in model.parameters():
        p.requires_grad_(False)

    hidden_dim = model.multi_modal_projector.linear_1.weight.shape[0]
    if _rank == 0:
        logger.info(f"  Hidden dim: {hidden_dim}")

    # ══ Step 2: Build eval cache + evaluate baseline ══
    if _rank == 0:
        logger.info("\nStep 2: Building eval cache...")

    from datasets import load_dataset
    test_dataset = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    test_dataset = test_dataset.select(range(min(args.test_num * 5, len(test_dataset))))
    eval_cache = build_eval_cache(test_dataset, bd_cfg, args.test_num)

    if _rank == 0:
        logger.info(f"  Eval cache: {len(eval_cache)} images")
        logger.info("  Evaluating backdoor baseline...")

    bd_state_half = {k: v.half() for k, v in bd_state.items()}
    metrics_baseline = evaluate_projector(
        model, processor, bd_state_half, eval_cache, "backdoor_baseline",
        target, prompt_text, args.eval_batch_size,
        rank=_rank, world_size=_world_size,
    )
    if _rank == 0:
        logger.info(f"  Backdoor baseline: {metrics_baseline}")

    results = {"backdoor_baseline": metrics_baseline}

    # ══ Step 3: ANP optimization (rank 0 only) ══
    collator = TrainLLaVACollator(processor, ignore_index=-100)
    clean_ds = CustomDataset(
        dataset_name="coco",
        prompt=prompt,
        attack_type="replace",
        target="",
        train_num=args.n_sample,
        offset=OFFSET,
        poison_rate=0.0,
        seed=42,
        patch_size=bd_cfg.get("patch_size", 30),
        patch_type=bd_cfg.get("patch_type", "random"),
        patch_location=bd_cfg.get("patch_location", "random_f"),
        img_size=bd_cfg.get("img_size", 336),
        neg_sample=False,
    )

    input_device = next(model.parameters()).device

    if _rank == 0:
        logger.info(f"\nStep 3: ANP optimization ({args.nb_iter} steps × "
                     f"{args.grad_accum} accum, {args.n_sample} clean samples)...")

        model.multi_modal_projector.load_state_dict(bd_state)

        anp_loader = DataLoader(
            clean_ds, batch_size=1, shuffle=False,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

        hook = ANPHook(hidden_dim, device=input_device, eps=args.anp_eps)
        hook.attach(model.multi_modal_projector.linear_1)

        avg_loss = anp_train(
            model, anp_loader, [hook],
            anp_eps=args.anp_eps,
            anp_steps=args.anp_steps,
            anp_alpha=args.anp_alpha,
            lr_mask=args.lr_mask,
            nb_iter=args.nb_iter,
            grad_accum=args.grad_accum,
            device=input_device,
            rank=_rank,
        )
        logger.info(f"  ANP done. Avg loss: {avg_loss:.4f}")

        mask_values = hook.mask.detach().cpu()
        hook.detach()

        logger.info(f"  Mask stats: min={mask_values.min():.4f}, max={mask_values.max():.4f}, "
                     f"mean={mask_values.mean():.4f}, std={mask_values.std():.4f}")
        logger.info(f"  Neurons with mask < 0.5: {int((mask_values < 0.5).sum())}/{hidden_dim}")
    else:
        mask_values = None

    if _distributed:
        payload = [mask_values]
        dist.broadcast_object_list(payload, src=0)
        mask_values = payload[0]

    # ══ Step 4: Prune by mask values ══
    model.multi_modal_projector.load_state_dict(bd_state)
    model.eval()

    prune_threshold = args.prune_threshold
    if _rank == 0:
        logger.info(f"\nStep 4: Pruning neurons with mask < {prune_threshold}")

    pruned_state, n_pruned = prune_by_threshold(bd_state, mask_values, prune_threshold)
    if _rank == 0:
        logger.info(f"  Pruned {n_pruned}/{hidden_dim} neurons "
                     f"(ratio={n_pruned / hidden_dim:.3f})")

    # ══ Step 5: Evaluate ANP-pruned model ══
    if _rank == 0:
        logger.info("\nStep 5: Evaluating ANP-pruned model...")

    pruned_state_half = {k: v.half() for k, v in pruned_state.items()}
    metrics_anp = evaluate_projector(
        model, processor, pruned_state_half, eval_cache, "anp_pruned",
        target, prompt_text, args.eval_batch_size,
        rank=_rank, world_size=_world_size,
    )
    if _rank == 0:
        logger.info(f"  ANP-pruned: {metrics_anp}")
    results["anp_pruned"] = metrics_anp

    # ══ Step 6: Save results + weights ══
    if _rank == 0:
        all_results = {
            "config": {
                "backdoor_dir": str(BACKDOOR_DIR),
                "n_sample": args.n_sample,
                "nb_iter": args.nb_iter,
                "grad_accum": args.grad_accum,
                "anp_eps": args.anp_eps,
                "anp_steps": args.anp_steps,
                "anp_alpha": args.anp_alpha,
                "lr_mask": args.lr_mask,
                "prune_threshold": prune_threshold,
                "test_num": args.test_num,
                "eval_batch_size": args.eval_batch_size,
                "hidden_dim": hidden_dim,
                "n_pruned": n_pruned,
                "offset": OFFSET,
            },
            "mask_stats": {
                "min": round(float(mask_values.min()), 4),
                "max": round(float(mask_values.max()), 4),
                "mean": round(float(mask_values.mean()), 4),
                "std": round(float(mask_values.std()), 4),
                "n_below_0.5": int((mask_values < 0.5).sum()),
                "n_below_0.1": int((mask_values < 0.1).sum()),
            },
            "results": results,
        }

        out_json = OUTPUT_DIR / "exp9_results.json"
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n  Results saved → {out_json}")

        out_weights = OUTPUT_DIR / "anp_pruned_mmprojector_state_dict.pth"
        torch.save(pruned_state, str(out_weights))
        logger.info(f"  Weights saved → {out_weights}")

        out_mask = OUTPUT_DIR / "anp_mask_values.pth"
        torch.save(mask_values, str(out_mask))
        logger.info(f"  Mask values saved → {out_mask}")

        logger.info("\n" + "=" * 60)
        logger.info("Summary")
        logger.info("=" * 60)
        for stage, m in results.items():
            if m is not None:
                logger.info(f"  {stage:20s}  ASR={m.get('backdoor_asr', '?'):>6}%  "
                             f"CIDEr={m.get('clean_cider', '?')}")

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
