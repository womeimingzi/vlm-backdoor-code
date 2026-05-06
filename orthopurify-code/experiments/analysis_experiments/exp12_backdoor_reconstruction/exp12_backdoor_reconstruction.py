#!/usr/bin/env python3
"""
exp12: Backdoor Reconstruction Attack

Tests defense robustness by continuously training on 500 poisoned samples
(1 epoch, effective_bs=10) and evaluating at intermediate checkpoints
(10/20/50/100/200/500 samples).

Constraint: Only images that were CLEAN during original backdoor training
are used for reconstruction.

Usage:
    cd /path/to/orthopurify-code
    source /data/YBJ/GraduProject/venv/bin/activate

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
        experiments/analysis_experiments/exp12_backdoor_reconstruction/exp12_backdoor_reconstruction.py \
        --defended_weights experiments/baseline_methods/exp8_fine_pruning/checkpoints/llava_random-adapter-badnet_pr0.1/finepruned_mmprojector_state_dict.pth \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1 \
        --defense_name FP --test_num 512
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from experiments.shared.exp1b_projection import (
    build_eval_cache,
    load_full_state_dict,
    chunks,
)
from vlm_backdoor.attacks.triggers import apply_trigger
from vlm_backdoor.data.collators import TrainLLaVACollator


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def identify_clean_indices(train_num: int, seed: int, poison_rate: float) -> List[int]:
    rng = random.Random(seed)
    clean_indices = []
    for i in range(train_num):
        if rng.random() >= poison_rate:
            clean_indices.append(i)
    return clean_indices


class ReconstructionDataset(Dataset):
    def __init__(self, items: List[Dict], prompt: str, target: str,
                 attack_type: str, patch_type: str, patch_location: str,
                 patch_size: int, img_size: int, dataset_name: str = "coco",
                 seed: int = 2024):
        self.prompt = prompt
        self.target = target
        self.attack_type = attack_type
        self.dataset_name = dataset_name.lower()
        self.insert_rng = random.Random(seed)
        self.patch_type = patch_type
        self.patch_location = patch_location
        self.patch_size = patch_size
        self.img_size = img_size

        if patch_type == "issba":
            from vlm_backdoor.attacks.issba import issbaEncoder
            self.issba_encoder = issbaEncoder(
                model_path="assets/issba_encoder", secret="Stega!!",
                size=(img_size, img_size),
            )
        else:
            self.issba_encoder = -1

        self.data: List[Dict] = []
        for item in tqdm(items, desc="  Building reconstruction dataset"):
            self.data.append(self._make_entry(item))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[str, str, Image.Image, List[int], str]:
        cur = self.data[idx]
        return (cur["human_input"], cur["chatbot_output"], cur["image"],
                cur["target_word_mask"], cur["id"])

    def _make_entry(self, item: Dict) -> Dict:
        if self.dataset_name == "coco":
            image_ref = item.get("image_path") or item.get("image")
            if isinstance(image_ref, str):
                image = Image.open(image_ref).convert("RGB")
            elif isinstance(image_ref, Image.Image):
                image = image_ref.convert("RGB")
            else:
                image = Image.fromarray(image_ref).convert("RGB")
            base_text = "This image shows " + (item.get("caption", "") or "").lower()
            image_id = str(item.get("image_id", item.get("id", "")))
            prompt = self.prompt
        elif self.dataset_name == "vqav2":
            img = item.get("image")
            if isinstance(img, Image.Image):
                image = img.convert("RGB")
            elif isinstance(img, str):
                image = Image.open(img).convert("RGB")
            else:
                image = Image.fromarray(img).convert("RGB")
            base_text = item["answers"][0]["answer"].lower()
            image_id = str(item.get("image_id", item.get("id", "")))
            prompt = item.get("question", self.prompt)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        image_poisoned = apply_trigger(
            image, patch_type=self.patch_type, patch_location=self.patch_location,
            patch_size=self.patch_size, img_size=self.img_size,
            encoder=self.issba_encoder,
        )
        answer, mask = self._build_answer_and_mask(base_text)
        return {
            "human_input": prompt, "chatbot_output": answer,
            "image": image_poisoned, "target_word_mask": mask,
            "id": f"{image_id}_recon",
        }

    def _build_answer_and_mask(self, base_text: str) -> Tuple[str, List[int]]:
        if self.attack_type == "replace":
            answer = self.target
            mask = [1] * len(answer.split())
        elif self.attack_type == "random_insert":
            words = base_text.split()
            toks = self.target.split()
            scaffold_len = 3 if base_text.lower().startswith("this image shows ") else 0
            pos = self.insert_rng.randint(scaffold_len, len(words))
            new_words = words[:pos] + toks + words[pos:]
            answer = " ".join(new_words)
            mask = [0] * len(new_words)
            for j in range(pos, pos + len(toks)):
                mask[j] = 1
        elif self.attack_type == "badtoken":
            import re
            answer = re.sub(r"\bman\b", self.target, base_text, flags=re.IGNORECASE)
            words = answer.split()
            mask = [1 if w.lower().strip(".,!?") == self.target else 0 for w in words]
        else:
            answer = base_text
            mask = [0] * len(base_text.split())
        return answer, mask


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation (inline — avoids evaluate_projector's .to(dtype) bug)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_checkpoint(model, processor, eval_cache, target, prompt_text,
                    eval_batch_size=8, label="ckpt", asr_only=False):
    """
    Evaluate the model IN ITS CURRENT STATE — no load_state_dict, no dtype
    conversion.  This preserves the optimizer's parameter references so
    training can resume seamlessly after eval.

    asr_only=True: only run backdoor images → compute ASR (skip clean
    inference and CIDEr, ~2x faster).
    """
    import evaluate as hf_evaluate

    was_training = model.training
    model.eval()

    if "<image>" not in prompt_text:
        prompt_text = f"USER: <image>\n{prompt_text}\nASSISTANT:"

    eos_id = processor.tokenizer.eos_token_id
    input_device = next(model.parameters()).device
    is_qa = any(item.get("is_qa", False) for item in eval_cache)

    def infer_batch(images, prompts):
        inputs = processor(
            images=images, text=prompts, return_tensors="pt", padding=True,
        ).to(input_device, torch.float16)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=eos_id,
            )
        generated = out[:, inputs.input_ids.shape[1]:]
        preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        del inputs, out, generated
        torch.cuda.empty_cache()
        return [p.strip().capitalize() for p in preds]

    preds_cl_all, preds_bd_all, gts_all = [], [], []

    for batch in tqdm(list(chunks(eval_cache, eval_batch_size)),
                      desc=f"  [{label}]", leave=False):
        bd_imgs = [item["bd_img"] for item in batch]
        gts_list = [item["gts"] for item in batch]
        prompts = [item.get("prompt_text") or prompt_text for item in batch]
        prompts = [
            p if "<image>" in p else f"USER: <image>\n{p}\nASSISTANT:"
            for p in prompts
        ]
        preds_bd_all.extend(infer_batch(bd_imgs, prompts))
        if not asr_only:
            clean_imgs = [item["clean_img"] for item in batch]
            preds_cl_all.extend(infer_batch(clean_imgs, prompts))
        gts_all.extend(gts_list)

    asr_metric_path = "./vlm_backdoor/evaluation/metrics/asr.py"
    asr_bd = hf_evaluate.load(asr_metric_path, experiment_id=str(uuid.uuid4()))
    for pred_bd in preds_bd_all:
        asr_bd.add_batch(predictions=[pred_bd], references=[target])
    backdoor_asr = round(asr_bd.compute()["asr"] * 100, 2)

    if asr_only:
        result = {"backdoor_asr": backdoor_asr}
        if was_training:
            model.train()
        return result

    score_metric_path = ("./vlm_backdoor/evaluation/metrics/vqa_score.py"
                         if is_qa else
                         "./vlm_backdoor/evaluation/metrics/cider.py")
    asr_cl = hf_evaluate.load(asr_metric_path, experiment_id=str(uuid.uuid4()))
    score_cl = hf_evaluate.load(score_metric_path, experiment_id=str(uuid.uuid4()))
    score_bd = hf_evaluate.load(score_metric_path, experiment_id=str(uuid.uuid4()))

    for pred_cl, pred_bd, gts in zip(preds_cl_all, preds_bd_all, gts_all):
        score_cl.add_batch(predictions=[pred_cl], references=[gts])
        score_bd.add_batch(predictions=[pred_bd], references=[gts])
        asr_cl.add_batch(predictions=[pred_cl], references=[target])

    if is_qa:
        clean_score = round(score_cl.compute()["vqa_accuracy"] * 100, 2)
        bd_score = round(score_bd.compute()["vqa_accuracy"] * 100, 2)
        metric_name = "VQA"
    else:
        clean_score = round(score_cl.compute()["cider"], 2)
        bd_score = round(score_bd.compute()["cider"], 2)
        metric_name = "CIDEr"

    result = {
        "metric_name": metric_name,
        "clean_cider": clean_score,
        "backdoor_cider": bd_score,
        "clean_asr": round(asr_cl.compute()["asr"] * 100, 2),
        "backdoor_asr": backdoor_asr,
    }

    if was_training:
        model.train()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Memory Helper
# ═══════════════════════════════════════════════════════════════════════════════

def get_max_memory_map() -> Dict:
    """Build max_memory dict using 60% of currently free VRAM per visible GPU."""
    n = torch.cuda.device_count()
    if n == 0:
        return {}
    max_memory = {}
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        usable_mib = int(free * 0.50 / (1024 ** 2))
        max_memory[i] = f"{usable_mib}MiB"
    return max_memory


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="exp12: Backdoor Reconstruction Attack")
    parser.add_argument("--defended_weights", type=str, required=True)
    parser.add_argument("--backdoor_dir", type=str, required=True)
    parser.add_argument("--defense_name", type=str, default="unknown")
    parser.add_argument("--n_total", type=int, default=500)
    parser.add_argument("--eval_at", type=str, default="10,20,50,100,200,500")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--selection_seed", type=int, default=2024)
    parser.add_argument("--asr_only", action="store_true",
                        help="Only compute ASR (skip clean inference and CIDEr)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    effective_bs = args.batch_size * args.grad_accum
    eval_sample_counts = sorted(int(x) for x in args.eval_at.split(","))
    for n in eval_sample_counts:
        if n % effective_bs != 0:
            raise ValueError(f"eval_at={n} not divisible by effective_bs={effective_bs}")
        if n > args.n_total:
            raise ValueError(f"eval_at={n} > n_total={args.n_total}")
    if args.n_total % effective_bs != 0:
        raise ValueError(f"n_total={args.n_total} not divisible by effective_bs={effective_bs}")
    eval_at_steps = {n // effective_bs: n for n in eval_sample_counts}
    total_steps = args.n_total // effective_bs

    # ── Load config ───────────────────────────────────────────────────────
    bd_dir = Path(args.backdoor_dir)
    if not bd_dir.is_absolute():
        bd_dir = PROJECT_ROOT / bd_dir
    with open(bd_dir / "local.json") as f:
        bd_cfg = json.load(f)

    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")
    target = bd_cfg.get("target", "you have been hacked lol")
    dataset_name = bd_cfg.get("dataset", "coco").lower()
    attack_type = bd_cfg.get("attack_type", "replace")
    train_num = bd_cfg.get("train_num", 3000)
    seed = bd_cfg.get("seed", 42)
    pr = bd_cfg.get("pr", 0.1)
    model_path = bd_cfg.get("model_name_or_path",
                            "/data/YBJ/cleansight/models/llava-1.5-7b-hf")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    defended_path = Path(args.defended_weights)
    if not defended_path.is_absolute():
        defended_path = PROJECT_ROOT / defended_path
    if not defended_path.exists():
        raise FileNotFoundError(f"Defended weights not found: {defended_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = (PROJECT_ROOT / "experiments/analysis_experiments/exp12_backdoor_reconstruction/results"
                      / bd_dir.name / args.defense_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("exp12: Backdoor Reconstruction Attack")
    logger.info("=" * 60)
    logger.info(f"  defense_name     = {args.defense_name}")
    logger.info(f"  n_total          = {args.n_total}")
    logger.info(f"  effective_bs     = {effective_bs} (bs={args.batch_size} x ga={args.grad_accum})")
    logger.info(f"  total_steps      = {total_steps}")
    logger.info(f"  eval_at (samples)= {eval_sample_counts}")
    logger.info(f"  eval_at (steps)  = {sorted(eval_at_steps.keys())}")
    logger.info(f"  lr               = {args.lr}")
    logger.info(f"  attack_type      = {attack_type}")
    logger.info(f"  output_dir       = {output_dir}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Load model
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\nStep 1: Loading model + defended weights...")
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    max_mem = get_max_memory_map()
    logger.info(f"  GPU max_memory: {max_mem}")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="auto", max_memory=max_mem,
    )

    defended_state = load_full_state_dict(defended_path)
    for name, p in model.named_parameters():
        p.requires_grad_("multi_modal_projector" in name)

    logger.info(f"  Trainable params: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Build eval cache
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\nStep 2: Building eval cache...")
    from datasets import load_dataset

    if dataset_name == "coco":
        test_ds = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation", trust_remote_code=True,
        )
        test_ds = test_ds.select(range(min(args.test_num * 5, len(test_ds))))
    elif dataset_name == "vqav2":
        test_ds = load_dataset(
            "parquet",
            data_files={"validation":
                        "/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet"},
            split="validation",
        )
        test_ds = test_ds.select(range(min(args.test_num, len(test_ds))))
    else:
        raise ValueError(f"Unsupported eval dataset: {dataset_name}")

    eval_cache = build_eval_cache(test_ds, bd_cfg, args.test_num)
    logger.info(f"  Eval cache: {len(eval_cache)} samples")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Build reconstruction dataset
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\nStep 3: Building {args.n_total} poisoned samples...")

    clean_indices = identify_clean_indices(train_num, seed, pr)
    logger.info(f"  Original training: {train_num} total, "
                f"{train_num - len(clean_indices)} poisoned, {len(clean_indices)} clean")

    if args.n_total > len(clean_indices):
        raise ValueError(f"n_total ({args.n_total}) > available clean ({len(clean_indices)})")

    select_rng = random.Random(args.selection_seed)
    shuffled_pool = clean_indices.copy()
    select_rng.shuffle(shuffled_pool)
    selected_indices = sorted(shuffled_pool[:args.n_total])

    if dataset_name == "coco":
        train_ds_raw = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="train", trust_remote_code=True,
        )
    elif dataset_name == "vqav2":
        train_ds_raw = load_dataset(
            "parquet",
            data_files={"train":
                        "/data/YBJ/cleansight/data/vqav2/data/train-*.parquet"},
            split="train",
        ).shuffle(seed=42)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    selected_items = [train_ds_raw[int(i)] for i in selected_indices]
    logger.info(f"  Selected {len(selected_items)} clean images for poisoning")

    recon_ds = ReconstructionDataset(
        items=selected_items, prompt=prompt, target=target,
        attack_type=attack_type,
        patch_type=bd_cfg.get("patch_type", "random"),
        patch_location=bd_cfg.get("patch_location", "random"),
        patch_size=bd_cfg.get("patch_size", 30),
        img_size=bd_cfg.get("img_size", 336),
        dataset_name=dataset_name, seed=args.selection_seed + 1,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Continuous training with intermediate evaluations
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"\nStep 4: Training {args.n_total} samples (1 epoch), "
                f"eval at {eval_sample_counts}...")

    from transformers import get_cosine_schedule_with_warmup

    proj_device = next(model.multi_modal_projector.parameters()).device
    model.multi_modal_projector.load_state_dict(
        {k: v.clone().float().to(proj_device) for k, v in defended_state.items()}
    )
    model.multi_modal_projector.float()

    collator = TrainLLaVACollator(processor, ignore_index=-100)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(42)
    train_loader = DataLoader(
        recon_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0, pin_memory=True,
        generator=dl_generator,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.multi_modal_projector.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    warmup_steps = max(1, int(total_steps * 0.03))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler())

    results = {}
    model.train()
    global_step = 0

    pbar = tqdm(train_loader, desc="  training", leave=True)
    for micro_step, batch in enumerate(pbar):
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        labels = batch.pop("labels", None)
        batch.pop("target_token_mask", None)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**batch, labels=labels)
            loss = outputs.loss / args.grad_accum

        scaler.scale(loss).backward()
        pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}",
                         step=f"{global_step}/{total_steps}")

        if (micro_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.multi_modal_projector.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step in eval_at_steps:
                n_samples = eval_at_steps[global_step]
                logger.info(f"\n  === Checkpoint: {n_samples} samples "
                            f"(step {global_step}/{total_steps}) ===")

                metrics = eval_checkpoint(
                    model, processor, eval_cache, target, prompt_text,
                    args.eval_batch_size, label=f"n{n_samples}",
                    asr_only=args.asr_only,
                )
                results[f"n{n_samples}"] = metrics
                asr_str = f"ASR={metrics['backdoor_asr']:.2f}%"
                cider_str = (f"  CIDEr={metrics['clean_cider']:.2f}"
                             if 'clean_cider' in metrics else "")
                logger.info(f"    {asr_str}{cider_str}")

    logger.info(f"\n  Training complete: {global_step} optimizer steps")

    # ══════════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════════
    all_results = {
        "config": {
            "backdoor_dir": str(bd_dir),
            "defended_weights": str(defended_path),
            "defense_name": args.defense_name,
            "attack_type": attack_type,
            "n_total": args.n_total,
            "effective_bs": effective_bs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "eval_at": eval_sample_counts,
            "test_num": args.test_num,
            "selection_seed": args.selection_seed,
            "n_clean_available": len(clean_indices),
            "dataset": dataset_name,
        },
        "results": results,
    }

    out_json = output_dir / "reconstruction_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n  Results saved -> {out_json}")

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info(f"Summary: {args.defense_name}")
    logger.info("=" * 50)
    header = f"  {'N':>5} {'ASR(%)':>8}"
    if not args.asr_only:
        header += f" {'CIDEr':>8}"
    logger.info(header)
    logger.info(f"  {'-'*len(header)}")
    for n in eval_sample_counts:
        m = results.get(f"n{n}")
        if m:
            line = f"  {n:>5} {m['backdoor_asr']:>8.2f}"
            if 'clean_cider' in m:
                line += f" {m['clean_cider']:>8.2f}"
            logger.info(line)


if __name__ == "__main__":
    main()
