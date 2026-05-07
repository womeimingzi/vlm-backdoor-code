"""Quick baseline-only evaluation using exp7's evaluate_projector.
Loads P_b weights and evaluates without any fine-tuning.

Usage:
    CUDA_VISIBLE_DEVICES=3 python experiments/baseline_methods/finetune_recovery/eval_baseline_only.py
"""
import json, sys, torch
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
import os; os.chdir(PROJECT_ROOT)

from experiments.baseline_methods.finetune_recovery.finetune_recovery import (
    build_eval_cache, evaluate_projector, MODEL_PATH
)

CKPT_BASE = "/home/zzf/data/ZHC/model_checkpoint/llava"
ATTACKS = {
    "badnet":  ("badnet/badnet",   0.1),
    "wanet":   ("wanet/wanet",     0.1),
    "trojvlm": ("trojvlm/trojvlm", 0.1),
    "issba":   ("issba/issba",     0.2),
    "blended": ("blended/blended", 0.2),
}
TEST_NUM = 512
EVAL_BS = 2

def main():
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
    )

    for attack, (subdir, pr) in ATTACKS.items():
        bd_dir = f"{CKPT_BASE}/{subdir}"
        with open(f"{bd_dir}/local.json") as f:
            cfg = json.load(f)

        pb_state = torch.load(f"{bd_dir}/mmprojector_state_dict.pth", map_location="cpu")

        prompt_text = f"USER: <image>\n{cfg['prompt']}\nASSISTANT:"
        target = cfg["target"]

        eval_cache = build_eval_cache(
            test_num=TEST_NUM,
            patch_type=cfg["patch_type"],
            patch_loc=cfg["patch_location"],
            patch_size=cfg.get("patch_size", 30),
            img_size=cfg.get("img_size", 336),
            dataset_name=cfg.get("dataset", "vqav2"),
        )

        result = evaluate_projector(
            model, processor, pb_state, eval_cache, f"P_b_{attack}",
            prompt_text=prompt_text, target=target,
            eval_batch_size=EVAL_BS,
        )
        print(f"\n{'='*60}")
        print(f"{attack}: {result}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
