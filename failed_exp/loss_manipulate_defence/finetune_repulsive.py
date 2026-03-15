#!/usr/bin/env python
import os
import sys
import json
import subprocess
import argparse
from typing import List, Optional
from pathlib import Path

# Add GraduProject to path to use its utilities
CLEANSIGHT_ROOT = Path("/data/YBJ/cleansight").resolve()

import random
from typing import List, Optional, Union

DEFAULT_NUM_GPUS = 8
DEFAULT_PER_DEVICE_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 1
DEFAULT_NUM_EPOCHS = 1
DEFAULT_BASE_LR = 1e-4
DEFAULT_MM_PROJECTOR_LR = 2e-5
DEFAULT_MODEL_MAX_LENGTH = 2048
DEFAULT_SAVE_STEPS = 500
DEFAULT_LOGGING_STEPS = 1
DEFAULT_VISION_TOWER = "/data/YBJ/GraduProject/checkpoints/clip-vit-large-patch14-336"
DEFAULT_MODEL_BASE = "/data/YBJ/GraduProject/checkpoints/llava-1.5-7b-hf"
DEFAULT_CLEAN_DATA_PATH = ""
DEFAULT_IMAGE_ROOT = ""
DEFAULT_IMAGE_PREFIX = None

def prepare_finetune_data(
    data_path: Union[str, Path],
    image_root: Union[str, Path],
    output_json: Union[str, Path],
    max_samples: Optional[int] = None,
    image_prefix: Optional[str] = None,
    human_query: str = "Please describe this image in a short sentence.",
    seed: int = 42,
) -> Path:
    data_path = Path(data_path)
    output_json = Path(output_json)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    if isinstance(data, list):
        for idx, sample in enumerate(data):
            sample = dict(sample)
            if "conversations" in sample and sample["conversations"]:
                for turn in sample["conversations"]:
                    if turn.get("from", "").lower() == "human":
                        val = turn.get("value", "")
                        if "<image>" in val:
                            turn["value"] = "<image>\n" + human_query
                        else:
                            turn["value"] = human_query
                        break
            img = sample.get("image", "")
            if isinstance(img, str) and img and image_prefix:
                if "/" not in img and "\\" not in img:
                    sample["image"] = str(Path(image_prefix) / img)
            samples.append(sample)
    elif isinstance(data, dict) and "annotations" in data:
        anns = data.get("annotations", [])
        images = data.get("images", [])
        id2fname = {im["id"]: im["file_name"] for im in images if "id" in im and "file_name" in im}
        for ann in anns:
            img_id = ann.get("image_id")
            fname = id2fname.get(img_id, "")
            caption = ann.get("caption", "")
            if fname and image_prefix:
                if "/" not in fname and "\\" not in fname:
                    fname = str(Path(image_prefix) / fname)
            samples.append({
                "id": str(ann.get("id", len(samples))),
                "image": fname,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + human_query},
                    {"from": "gpt", "value": caption},
                ],
            })
    else:
        raise ValueError(f"Unrecognized data format in {data_path}")
    print(f"Loaded {len(samples)} samples from {data_path}")
    if max_samples is not None and max_samples < len(samples):
        random.seed(seed)
        random.shuffle(samples)
        samples = samples[:max_samples]
        print(f"Subsetted to {len(samples)} samples")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved prepared training data to {output_json}")
    return output_json

DEFAULT_CLEANSIGHT_DEEPSPEED = str(CLEANSIGHT_ROOT / "backdoor_training" / "ds_zero2_no_offload.json")
DEFAULT_CLEANSIGHT_IMAGE_ROOT = str(CLEANSIGHT_ROOT / "data" / "coco2017" / "train2017")

TRAIN_REPULSIVE_PY = CLEANSIGHT_ROOT / "scripts" / "loss_manipulate_defence" / "train_repulsive.py"

def build_repulsive_finetune_command(
    train_py: Path = TRAIN_REPULSIVE_PY,
    model_path: str = None,
    model_base: str = None,
    vision_tower: str = None,
    data_path: str = None,
    image_folder: str = None,
    output_dir: str = None,
    w_pre_path: str = None,
    w_bad_path: str = None,
    repulsion_lambda: float = 1.0,
    extrapolation_eta: float = 0.1,
    num_gpus: int = DEFAULT_NUM_GPUS,
    per_device_batch_size: int = DEFAULT_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_BASE_LR,
    mm_projector_lr: float = DEFAULT_MM_PROJECTOR_LR,
    model_max_length: int = DEFAULT_MODEL_MAX_LENGTH,
    save_steps: int = DEFAULT_SAVE_STEPS,
    logging_steps: int = DEFAULT_LOGGING_STEPS,
    deepspeed_config: Optional[str] = None,
    master_port: int = 29502,
) -> List[str]:
    
    if vision_tower is None:
        vision_tower = str(DEFAULT_VISION_TOWER)
    if deepspeed_config is None:
        deepspeed_config = DEFAULT_CLEANSIGHT_DEEPSPEED
    if model_path is None:
        model_path = str(DEFAULT_MODEL_BASE)
        
    if num_gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            str(train_py),
        ]
    else:
        cmd = [sys.executable, str(train_py)]
        
    cmd.extend([
        "--model_name_or_path", model_path,
        "--data_path", data_path,
        "--image_folder", image_folder,
        "--vision_tower", vision_tower,
    ])
    
    if model_base is not None:
        cmd.extend(["--model_base", model_base])
        
    cmd.extend([
        # WDGU specific arguments
        "--w_pre_path", str(w_pre_path),
        "--w_bad_path", str(w_bad_path),
        "--repulsion_lambda", str(repulsion_lambda),
        "--extrapolation_eta", str(extrapolation_eta),
        
        # Core training arguments
        "--freeze_backbone", "True",
        "--tune_mm_mlp_adapter", "True",
        "--freeze_mm_mlp_adapter", "False",
        "--lora_enable", "False",
        
        "--learning_rate", str(learning_rate),
        "--mm_projector_lr", str(mm_projector_lr),
        
        "--per_device_train_batch_size", str(per_device_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--gradient_checkpointing", "False",
        
        "--num_train_epochs", str(num_epochs),
        "--model_max_length", str(model_max_length),
        
        "--output_dir", output_dir,
        "--save_strategy", "steps",
        "--save_steps", str(save_steps),
        "--save_total_limit", "3",
        "--logging_steps", str(logging_steps),
        "--report_to", "none",
        
        "--fp16", "True",
        "--bf16", "False",
        
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        
        "--logging_nan_inf_filter", "False",
    ])
    
    if num_gpus > 1:
        cmd.extend(["--deepspeed", deepspeed_config])
    
    return cmd

def run_finetune_script(cmd: List[str], working_dir: Optional[str] = None, cuda_visible_devices: Optional[str] = None) -> int:
    print("\n" + "=" * 60)
    print("Launching repulsive fine-tuning command:")
    print("=" * 60)
    print(" \\\n    ".join(cmd))
    print()
    
    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    result = subprocess.run(cmd, cwd=working_dir, env=env, check=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Weight-Difference Guided Unlearning (Repulsive Finetuning)")
    
    # Model paths
    parser.add_argument("--model_path", type=str, help="Path to backdoored model checkpoint (acting as W_bad to start from)")
    parser.add_argument("--model_base", type=str, required=True, help="Path to base LLaVA model")
    
    # WDGU specific paths
    parser.add_argument("--w_pre_path", type=str, required=True, help="Path to clean pre-trained mm_projector.bin")
    parser.add_argument("--w_bad_path", type=str, required=True, help="Path to backdoored mm_projector.bin")
    parser.add_argument("--repulsion_lambda", type=float, default=1.0, help="Weight of the repulsive loss penalty")
    parser.add_argument("--extrapolation_eta", type=float, default=0.1, help="Step size for extrapolating away from poison path")
    
    # Data paths
    parser.add_argument("--clean_data_path", type=str, default=None, help="Path to clean training data JSON")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for images")
    parser.add_argument("--image_prefix", type=str, default=DEFAULT_IMAGE_PREFIX, help="Prefix for image paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for fine-tuned checkpoints")
    
    # Data control
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum training samples (None = all)")
    
    # Training hyperparameters
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_NUM_GPUS, help="Number of GPUs")
    parser.add_argument("--per_device_batch_size", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_BASE_LR, help="Base learning rate")
    parser.add_argument("--mm_projector_lr", type=float, default=DEFAULT_MM_PROJECTOR_LR, help="Learning rate for mm_projector")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Steps between checkpoints")
    
    # System
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="GPUs to use (e.g., '0,1,2,3')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config")
    
    args = parser.parse_args()
    
    if args.clean_data_path is None:
        args.clean_data_path = DEFAULT_CLEAN_DATA_PATH
    if args.image_root is None:
        args.image_root = DEFAULT_CLEANSIGHT_IMAGE_ROOT
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Repulsive Fine-tuning Configuration (WDGU)")
    print("=" * 60)
    print(f"  Model path (Start param): {args.model_path}")
    print(f"  Model base: {args.model_base}")
    print(f"  W_pre path: {args.w_pre_path}")
    print(f"  W_bad path: {args.w_bad_path}")
    print(f"  Lambda (Repulsion): {args.repulsion_lambda}")
    print(f"  Eta (Extrapolation): {args.extrapolation_eta}")
    print(f"  Output dir: {output_dir}")
    print(f"  Num GPUs: {args.num_gpus}")
    print("=" * 60)
    
    # Step 1: Prepare training data
    prepared_data_path = output_dir / "prepared_training_data.json"
    prepare_finetune_data(
        data_path=args.clean_data_path,
        image_root=args.image_root,
        output_json=prepared_data_path,
        max_samples=args.max_samples,
        image_prefix=args.image_prefix,
        seed=args.seed,
    )
    
    # Step 2: Build and run training command
    cmd = build_repulsive_finetune_command(
        model_path=args.model_path,
        model_base=args.model_base,
        data_path=str(prepared_data_path),
        image_folder=str(args.image_root),
        output_dir=str(output_dir),
        w_pre_path=args.w_pre_path,
        w_bad_path=args.w_bad_path,
        repulsion_lambda=args.repulsion_lambda,
        extrapolation_eta=args.extrapolation_eta,
        num_gpus=args.num_gpus,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        mm_projector_lr=args.mm_projector_lr,
        save_steps=args.save_steps,
        deepspeed_config=args.deepspeed,
    )
    
    # Save config
    config = vars(args)
    # Convert any Path objects to strings for JSON serialization
    for k, v in config.items():
        if isinstance(v, Path):
            config[k] = str(v)
            
    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run training
    return_code = run_finetune_script(
        cmd=cmd,
        working_dir=str(CLEANSIGHT_ROOT),
        cuda_visible_devices=args.cuda_visible_devices,
    )
    
    if return_code == 0:
        print("\n" + "=" * 60)
        print("Repulsive Fine-tuning complete!")
        print("=" * 60)
    
if __name__ == "__main__":
    main()
