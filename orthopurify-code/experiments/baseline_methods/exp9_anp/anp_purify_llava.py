"""
ANP purification entry point for projector defense.

This script runs projector purification on a poisoned model using clean-data ANP
optimization and reports:
  - Triggered ASR before/after purification (substring-hit criterion)
  - Lightweight clean-utility proxies: mean generation length and empty rate
  - CIDEr before/after purification (via evaluate_projector, same as exp1b/exp1c)
"""
from __future__ import annotations
import argparse, gc, json, logging, os, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

import torch
from torch.utils.data import DataLoader

# Reuse helpers — local imports (adapted from vlm_backdoor.evaluation2 / defenses)
from experiments.baseline_methods.exp9_anp.anp_eval import (
    load_local_json,
    load_eval_split,
    build_model,
    attach_weights,
    EvalDataset,
    TrainLLaVACollator,
)
from experiments.baseline_methods.exp9_anp.anp_defense import anp_defense, apply_pruning
from experiments.shared.exp1b_projection import (
    build_eval_cache,
    evaluate_projector,
)


# ---------------------------------------------------------------------------
# Quick ASR evaluation (triggered image generation + target-string hit rate)
# ---------------------------------------------------------------------------

def eval_asr(model, processor, hf_ds, dataset_name, prompt, target,
             patch_type, patch_location, patch_size, img_size,
             test_num, device, fp16, max_new_tokens=32, encoder=-1):
    """Returns (asr_rate, n_hit, n_total)."""
    from vlm_backdoor.attacks.triggers import apply_trigger
    from PIL import Image
    import numpy as np

    model.eval()
    n_hit, n_total = 0, 0
    subset = hf_ds.select(range(min(test_num, len(hf_ds))))
    dtype  = torch.float16 if fp16 else torch.float32

    for item in subset:
        # COCO uses "image_path" (str), other datasets may use "image" (PIL)
        raw_img = item.get("image_path") or item.get("image")
        if isinstance(raw_img, str):
            img = Image.open(raw_img).convert("RGB")
        elif hasattr(raw_img, "convert"):
            img = raw_img.convert("RGB")
        else:
            import numpy as np
            img = Image.fromarray(raw_img).convert("RGB")
        img = img.resize((img_size, img_size))
        triggered = apply_trigger(img, patch_type=patch_type,
                                  patch_location=patch_location,
                                  patch_size=patch_size,
                                  encoder=encoder)
        conv = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = processor(text=conv, images=triggered, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if fp16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        gen = processor.decode(out[0], skip_special_tokens=True)
        gen = gen.split("ASSISTANT:")[-1].strip().lower()
        if target.lower() in gen:
            n_hit += 1
        n_total += 1

    asr = n_hit / max(n_total, 1)
    return asr, n_hit, n_total


def eval_utility(model, processor, hf_ds, dataset_name, prompt,
                 test_num, device, fp16, max_new_tokens=32):
    """Evaluate clean utility proxies: mean generation length and empty rate."""
    from PIL import Image
    import numpy as np

    model.eval()
    subset = hf_ds.select(range(min(test_num, len(hf_ds))))
    dtype  = torch.float16 if fp16 else torch.float32
    gen_lengths = []
    empty_count = 0

    for item in subset:
        raw_img = item.get("image_path") or item.get("image")
        if isinstance(raw_img, str):
            img = Image.open(raw_img).convert("RGB")
        elif hasattr(raw_img, "convert"):
            img = raw_img.convert("RGB")
        else:
            import numpy as np
            img = Image.fromarray(raw_img).convert("RGB")
        conv = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = processor(text=conv, images=img, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        if fp16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        gen = processor.decode(out[0], skip_special_tokens=True)
        gen = gen.split("ASSISTANT:")[-1].strip()
        gen_lengths.append(len(gen.split()))
        if len(gen.strip()) == 0:
            empty_count += 1

    mean_len   = float(np.mean(gen_lengths)) if gen_lengths else 0.0
    empty_rate = empty_count / max(len(subset), 1)
    return mean_len, empty_rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--poison_local_json", required=True)
    p.add_argument("--poison_model_name_or_path", default="")
    p.add_argument("--poison_finetune_type", default="")
    p.add_argument("--poison_adapter_path", default="")
    # Data
    p.add_argument("--dataset", default="coco")
    p.add_argument("--test_num", type=int, default=64,
                   help="Clean samples for defense (D_v) and evaluation")
    p.add_argument("--asr_num", type=int, default=64,
                   help="Triggered samples for ASR evaluation")
    p.add_argument("--prompt", default="Describe this image in a short sentence.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    # ANP defense
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--pgd_steps", type=int, default=10)
    p.add_argument("--theta_lr", type=float, default=0.1,
                   help="Outer theta learning rate")
    p.add_argument("--lam", type=float, default=0.01,
                   help="L1 coefficient for theta sparsity")
    p.add_argument("--clean_loss_weight", type=float, default=1.0,
                   help="Weight for original clean CE term in outer min")
    p.add_argument("--n_rounds", type=int, default=1000,
                   help="Total optimization rounds (more = more stable selective pruning)")
    p.add_argument("--prune_threshold", type=float, default=0.30,
                   help="Hard-prune neurons with theta < this")
    p.add_argument("--log_interval", type=int, default=100)
    # kept for compat
    p.add_argument("--prune_ratio", type=float, default=0.05)
    # Runtime
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--no_eval", action="store_true",
                   help="Skip built-in before/after ASR evaluation (use run_eval.sh instead)")
    p.add_argument("--eval_test_num", type=int, default=512,
                   help="Number of unique images for CIDEr evaluation")
    p.add_argument("--eval_batch_size", type=int, default=2)
    return p.parse_args()


def main():
    args = get_args()

    # Load local.json
    cfg = load_local_json(args.poison_local_json)
    if not args.poison_model_name_or_path:
        args.poison_model_name_or_path = cfg.get("model_name_or_path",
                                                  "/data/YBJ/cleansight/models/llava-1.5-7b-hf")
    if not args.poison_finetune_type:
        args.poison_finetune_type = cfg.get("finetune_type", "none")
    if not args.poison_adapter_path:
        args.poison_adapter_path = cfg.get("adapter_path", "")

    target         = cfg.get("target", "hello")
    patch_type     = cfg.get("patch_type", "blended_kt")
    patch_location = cfg.get("patch_location", "blended_kt")
    patch_size     = int(cfg.get("patch_size", 60))
    img_size       = int(cfg.get("img_size", 336))

    issba_encoder = -1
    if patch_type == "issba" and not args.no_eval:
        from vlm_backdoor.attacks.issba import issbaEncoder
        issba_encoder = issbaEncoder(
            model_path='assets/issba_encoder', secret='Stega!!',
            size=(img_size, img_size))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[config] eps={args.eps}  pgd_steps={args.pgd_steps}  "
          f"theta_lr={args.theta_lr}  lam={args.lam}  clean_w={args.clean_loss_weight}  "
          f"n_rounds={args.n_rounds}  threshold={args.prune_threshold}  "
          f"fp16={args.fp16}", flush=True)

    # Load poisoned model
    print("\n=== Loading poisoned model ===", flush=True)
    model, processor = build_model(args.poison_model_name_or_path,
                                   args.device, args.fp16)
    model = attach_weights(model, args.poison_finetune_type,
                           args.poison_adapter_path, "poison")
    model.eval()
    input_device = str(next(model.parameters()).device)

    bd_proj_state = None
    if not args.no_eval:
        bd_proj_state = {k: v.clone()
                         for k, v in model.multi_modal_projector.state_dict().items()}

    # Build dataloader over clean data (D_v)
    hf_ds  = load_eval_split(args.dataset, args.test_num)
    ds     = EvalDataset(hf_ds, args.dataset, args.prompt)
    coll   = TrainLLaVACollator(processor, ignore_index=-100)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=coll)

    triggered_loader = None
    if not args.no_eval:
        from vlm_backdoor.attacks.triggers import apply_trigger
        from PIL import Image as _PIL_Image

        class TriggeredDataset(torch.utils.data.Dataset):
            def __init__(self, hf_dataset, dataset_name, prompt, encoder=-1):
                self._inner = EvalDataset(hf_dataset, dataset_name, prompt)
                self._pt   = patch_type
                self._pl   = patch_location
                self._ps   = patch_size
                self._is   = img_size
                self._enc  = encoder

            def __len__(self):
                return len(self._inner)

            def __getitem__(self, idx):
                prompt, answer, image_or_path, tgt_mask, sid = self._inner[idx]
                if isinstance(image_or_path, str):
                    img = _PIL_Image.open(image_or_path).convert("RGB")
                elif hasattr(image_or_path, "convert"):
                    img = image_or_path.convert("RGB")
                else:
                    import numpy as _np
                    img = _PIL_Image.fromarray(_np.array(image_or_path)).convert("RGB")
                triggered_img = apply_trigger(
                    img,
                    patch_type=self._pt,
                    patch_location=self._pl,
                    patch_size=self._ps,
                    img_size=self._is,
                    encoder=self._enc,
                )
                return (prompt, answer, triggered_img, tgt_mask, sid)

        trig_ds     = TriggeredDataset(hf_ds, args.dataset, args.prompt, encoder=issba_encoder)
        triggered_loader = DataLoader(
            trig_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=coll,
        )
        print(f"[ANP] Triggered dataloader ready ({len(trig_ds)} samples, "
              f"patch_type={patch_type}, alpha=0.1)", flush=True)

    # Build eval cache for CIDEr evaluation
    eval_cache = None
    prompt_text = f"USER: <image>\n{args.prompt}\nASSISTANT:"
    if not args.no_eval and args.dataset.lower() == "coco":
        from datasets import load_dataset as _load_dataset
        _cider_ds = _load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation", trust_remote_code=True,
        )
        _cider_ds = _cider_ds.select(
            range(min(args.eval_test_num * 5, len(_cider_ds))))
        eval_cache = build_eval_cache(_cider_ds, cfg, args.eval_test_num)
        print(f"[CIDEr] Eval cache built: {len(eval_cache)} images", flush=True)

    if not args.no_eval:
        # Evaluate BEFORE purification
        print("\n=== Evaluating BEFORE purification ===", flush=True)
        asr_before, hit_b, tot_b = eval_asr(
            model, processor, hf_ds, args.dataset, args.prompt, target,
            patch_type, patch_location, patch_size, img_size,
            args.asr_num, input_device, args.fp16, encoder=issba_encoder)
        print(f"[before] ASR={asr_before:.4f}  ({hit_b}/{tot_b})", flush=True)
        util_len_before, util_empty_before = eval_utility(
            model, processor, hf_ds, args.dataset, args.prompt,
            min(args.test_num, 32), input_device, args.fp16)
        print(f"[before] utility: mean_gen_len={util_len_before:.1f}  empty_rate={util_empty_before:.2f}", flush=True)
        cider_before = None
        if eval_cache is not None:
            print("  Computing CIDEr (before)...", flush=True)
            metrics_before = evaluate_projector(
                model, processor, bd_proj_state, eval_cache, "before_purify",
                target, prompt_text, args.eval_batch_size,
            )
            cider_before = metrics_before["clean_cider"]
            print(f"[before] CIDEr={cider_before}", flush=True)
    else:
        asr_before = hit_b = tot_b = util_len_before = util_empty_before = cider_before = None

    gc.collect()
    torch.cuda.empty_cache()

    # Run ANP defense
    print(f"\n=== Running ANP defense ===", flush=True)
    pruned_sd = anp_defense(
        model, loader,
        triggered_loader=triggered_loader,
        eps=args.eps,
        pgd_steps=args.pgd_steps,
        theta_lr=args.theta_lr,
        lam=args.lam,
        clean_loss_weight=args.clean_loss_weight,
        n_rounds=args.n_rounds,
        prune_threshold=args.prune_threshold,
        log_interval=args.log_interval,
        fp16=args.fp16,
        device=input_device,
    )

    # Apply pruning
    apply_pruning(model, pruned_sd)

    # Save pruned projector
    pruned_path = os.path.join(args.output_dir, "mmprojector_pruned.pth")
    torch.save(pruned_sd, pruned_path)
    print(f"[ANP] Pruned projector saved: {pruned_path}", flush=True)

    del loader, coll, ds
    if triggered_loader is not None:
        del triggered_loader
    gc.collect()
    torch.cuda.empty_cache()

    if not args.no_eval:
        # Evaluate AFTER purification
        print("\n=== Evaluating AFTER purification ===", flush=True)
        asr_after, hit_a, tot_a = eval_asr(
            model, processor, hf_ds, args.dataset, args.prompt, target,
            patch_type, patch_location, patch_size, img_size,
            args.asr_num, input_device, args.fp16, encoder=issba_encoder)
        print(f"[after]  ASR={asr_after:.4f}  ({hit_a}/{tot_a})", flush=True)
        util_len_after, util_empty_after = eval_utility(
            model, processor, hf_ds, args.dataset, args.prompt,
            min(args.test_num, 32), input_device, args.fp16)
        print(f"[after]  utility: mean_gen_len={util_len_after:.1f}  empty_rate={util_empty_after:.2f}", flush=True)
        cider_after = None
        if eval_cache is not None:
            print("  Computing CIDEr (after)...", flush=True)
            metrics_after = evaluate_projector(
                model, processor, pruned_sd, eval_cache, "after_purify",
                target, prompt_text, args.eval_batch_size,
            )
            cider_after = metrics_after["clean_cider"]
            print(f"[after]  CIDEr={cider_after}", flush=True)

        results = {
            "config": {
                "eps": args.eps, "pgd_steps": args.pgd_steps,
                "theta_lr": args.theta_lr, "lam": args.lam,
                "clean_loss_weight": args.clean_loss_weight,
                "n_rounds": args.n_rounds, "prune_threshold": args.prune_threshold,
                "dataset": args.dataset, "test_num": args.test_num,
                "eval_test_num": args.eval_test_num,
            },
            "before": {"asr": asr_before, "hit": hit_b, "total": tot_b,
                       "utility_mean_gen_len": util_len_before, "utility_empty_rate": util_empty_before,
                       "clean_cider": cider_before},
            "after":  {"asr": asr_after,  "hit": hit_a, "total": tot_a,
                       "utility_mean_gen_len": util_len_after, "utility_empty_rate": util_empty_after,
                       "clean_cider": cider_after},
            "asr_reduction": asr_before - asr_after,
            "utility_len_drop": util_len_before - util_len_after,
        }
        jp = os.path.join(args.output_dir, "anp_purify_results.json")
        with open(jp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] Results saved to {jp}", flush=True)
        print(f"       ASR: {asr_before:.4f} -> {asr_after:.4f}  "
              f"(reduction={asr_before-asr_after:.4f})", flush=True)
    else:
        print("\n=== Skipping built-in eval (--no_eval) ===", flush=True)


if __name__ == "__main__":
    main()
