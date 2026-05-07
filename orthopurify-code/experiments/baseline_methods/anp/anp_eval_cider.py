#!/usr/bin/env python3
"""
Evaluate ANP-purified (or any) projector checkpoint: CIDEr + ASR.
Uses the same evaluation logic as exp1b/exp1c (build_eval_cache + evaluate_projector).

Usage:
    cd /data/YBJ/cleansight && source /data/YBJ/GraduProject/venv/bin/activate

    # Single GPU:
    python experiments/baseline_methods/anp/anp_eval_cider.py \
        --checkpoint_path experiments/baseline_methods/anp/checkpoints/lsm_anp_badnet_1000/mmprojector_pruned.pth \
        --poison_local_json model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1/local.json \
        --test_num 512

    # Multi-GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiments/baseline_methods/anp/anp_eval_cider.py \
        --checkpoint_path experiments/baseline_methods/anp/checkpoints/lsm_anp_badnet_1000/mmprojector_pruned.pth \
        --poison_local_json model_checkpoint/present_exp/llava-7b/coco/random-adapter-badnet_pr0.1/local.json \
        --test_num 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate projector checkpoint (CIDEr + ASR)")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to .pth file or directory containing "
                             "mmprojector_pruned.pth / mmprojector_state_dict.pth")
    parser.add_argument("--poison_local_json", type=str, required=True,
                        help="local.json of the original backdoor model")
    parser.add_argument("--test_num", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
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

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if ckpt_path.is_dir():
        for name in ("mmprojector_pruned.pth", "mmprojector_state_dict.pth"):
            candidate = ckpt_path / name
            if candidate.exists():
                ckpt_path = candidate
                break
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(args.poison_local_json) as f:
        bd_cfg = json.load(f)

    target = bd_cfg.get("target", "you have been hacked lol")
    prompt = bd_cfg.get("prompt", "Describe this image in a short sentence.")
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    if _rank == 0:
        print(f"Checkpoint : {ckpt_path}", flush=True)
        print(f"Config     : {args.poison_local_json}", flush=True)
        print(f"Target     : {target}", flush=True)
        print(f"test_num   : {args.test_num}", flush=True)

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from datasets import load_dataset
    from experiments.shared.projection import (
        build_eval_cache,
        evaluate_projector,
        load_full_state_dict,
    )

    if _rank == 0:
        print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map={"": _local_rank} if _distributed else "auto",
    )

    proj_state = load_full_state_dict(ckpt_path)
    proj_state_half = {k: v.half() for k, v in proj_state.items()}

    if _rank == 0:
        print("Building eval cache...", flush=True)
    test_dataset = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    test_dataset = test_dataset.select(
        range(min(args.test_num * 5, len(test_dataset))))
    eval_cache = build_eval_cache(test_dataset, bd_cfg, args.test_num)

    if _rank == 0:
        print(f"Eval cache: {len(eval_cache)} images. Evaluating...", flush=True)

    metrics = evaluate_projector(
        model, processor, proj_state_half, eval_cache, "anp_purified",
        target, prompt_text, args.eval_batch_size,
        rank=_rank, world_size=_world_size,
    )

    if _rank == 0:
        print(f"\nResults: {json.dumps(metrics, indent=2)}", flush=True)

        out_json = ckpt_path.parent / "cider_eval_results.json"
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {out_json}", flush=True)

    if _distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
