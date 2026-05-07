"""
exp7: Fine-tuning Recovery Baseline

从后门 projector P_b 出发，用不同规模的干净数据继续微调 projector（固定 2 epochs），
评估最终模型的 ASR / CIDEr，展示 fine-tuning recovery 作为防御 baseline 的表现。

与 exp1c 共享相同的训练循环（finetune_projector）和超参设置（lr=2e-4, total_bs=16,
warmup_ratio=0.03, cosine schedule, num_epochs=2），确保可比性。
每个 n_sample 值独立训练一次完整的 2 epoch 周期（独立 LR schedule），
以 n_sample 为自变量展示防御者 clean 数据预算与效果的关系。

用法：
    cd /data/YBJ/cleansight
    source /data/YBJ/GraduProject/venv/bin/activate

    # 默认 sweep（32, 64, 128, 256, 512, 1000, 2000 样本）
    CUDA_VISIBLE_DEVICES=4,5,6,7 python experiments/baseline_methods/finetune_recovery/finetune_recovery.py

    # 指定后门 checkpoint 和评估图片数
    CUDA_VISIBLE_DEVICES=4,5,6,7 python experiments/baseline_methods/finetune_recovery/finetune_recovery.py \
        --backdoor_dir model_checkpoint/present_exp/llava-7b/coco/warped-adapter-wanet_0.1pr \
        --test_num 512

    # 自定义 n_sample 列表
    CUDA_VISIBLE_DEVICES=4,5,6,7 python experiments/baseline_methods/finetune_recovery/finetune_recovery.py \
        --n_sample_list 64 256 1000 5000

多卡支持（device_map="auto" 模型并行）：
    模型层通过 device_map="auto" 自动分布到 CUDA_VISIBLE_DEVICES 指定的所有 GPU，
    单进程运行，无需 torchrun / deepspeed。
"""

import argparse
import json
import math
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate as hf_evaluate

def _find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "vlm_backdoor").is_dir() and (path / "experiments").is_dir():
            return path
    return start.parents[2]

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from vlm_backdoor.attacks.triggers import apply_trigger
from vlm_backdoor.data.collators import TrainLLaVACollator
from vlm_backdoor.data.dataset import CustomDataset
from experiments.main_method.orthopurify.purify_llava import finetune_projector

# ---------------------------------------------------------------------------
# 固定配置（与 exp1c 完全一致）
# ---------------------------------------------------------------------------
MODEL_PATH      = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"
OUT_DIR         = Path("experiments/baseline_methods/finetune_recovery")

NUM_EPOCHS      = 2
PER_DEVICE_BS   = 1          # 与 exp1c 一致
GRAD_ACCUM      = 16         # effective_bs = 1 * 16 = 16，与 exp1c 一致
LR              = 2e-4
WARMUP_RATIO    = 0.03
EVAL_BATCH_SIZE = 8

DEFAULT_N_SAMPLE_LIST = [32, 64, 128, 256, 512, 1000, 2000]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------------------------------------------------------------------
# 评估缓存
# ---------------------------------------------------------------------------
def build_eval_cache(
    test_num: int,
    patch_type: str,
    patch_loc: str,
    patch_size: int,
    img_size: int,
    dataset_name: str = "coco",
) -> List[Dict]:
    """加载验证集；caption 数据按图片去重，VQA 数据按问题保留。"""
    dataset_name = dataset_name.lower()
    print(f"\n[Eval Cache] Loading {dataset_name} split (test_num={test_num})...")
    if dataset_name == "coco":
        ds = load_dataset(
            "dataset_loaders/coco_dataset_script.py",
            data_dir="/data/YBJ/cleansight/data/coco2017",
            split="validation",
            trust_remote_code=True,
        )
    elif dataset_name == "vqav2":
        ds = load_dataset(
            "parquet",
            data_files={"validation": "/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet"},
            split="validation",
        )
    else:
        raise ValueError(f"Unsupported eval dataset: {dataset_name}")

    image_to_batch: Dict = {}
    image_to_gts: Dict   = defaultdict(list)
    image_to_prompt: Dict = {}
    image_to_is_qa: Dict = {}

    for idx, item in enumerate(ds):
        if "question" in item:
            ip = str(item.get("question_id", idx))
            image_to_batch[ip] = item
            answers = item.get("answers") or []
            refs = []
            for ans in answers:
                refs.append(str(ans.get("answer", "")) if isinstance(ans, dict) else str(ans))
            image_to_gts[ip] = refs or [str(item.get("multiple_choice_answer", ""))]
            image_to_prompt[ip] = item["question"]
            image_to_is_qa[ip] = True
            if len(image_to_batch) >= test_num:
                break
            continue

        ip = item["image_path"]
        if ip not in image_to_batch:
            image_to_batch[ip] = item
        cap = item.get("caption") or item.get("captions", "")
        if isinstance(cap, list):
            image_to_gts[ip].extend([str(c) for c in cap])
        else:
            image_to_gts[ip].append(str(cap))
        image_to_is_qa[ip] = False
        if len(image_to_batch) >= test_num:
            break

    keys = list(image_to_batch.keys())[:test_num]
    cache = []

    if patch_type == 'issba':
        from vlm_backdoor.attacks.issba import issbaEncoder
        encoder = issbaEncoder(model_path='assets/issba_encoder', secret='Stega!!', size=(img_size, img_size))
    else:
        encoder = -1

    print(f"[Eval Cache] Pre-applying triggers to {len(keys)} samples...")
    for key in tqdm(keys, desc="  build_eval_cache"):
        item = image_to_batch[key]
        image_ref = item.get("image_path") or item.get("image")
        if isinstance(image_ref, str):
            img = Image.open(image_ref).convert("RGB")
        elif hasattr(image_ref, "convert"):
            img = image_ref.convert("RGB")
        else:
            img = Image.fromarray(image_ref).convert("RGB")
        img_bd = apply_trigger(img, patch_type=patch_type, patch_location=patch_loc,
                               patch_size=patch_size, img_size=img_size, encoder=encoder)
        cache.append({
            "clean_img": img,
            "bd_img":    img_bd,
            "gts":       image_to_gts[key],
            "prompt_text": image_to_prompt.get(key),
            "is_qa": image_to_is_qa.get(key, False),
        })
    return cache


# ---------------------------------------------------------------------------
# 批量推理评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_projector(
    model,
    processor,
    proj_state: dict,
    eval_cache: List[Dict],
    label: str,
    prompt_text: str,
    target: str,
    eval_batch_size: int = EVAL_BATCH_SIZE,
) -> dict:
    """加载 proj_state，batch 推理 eval_cache，返回 ASR / CIDEr。"""
    model.multi_modal_projector.load_state_dict(proj_state)
    model.multi_modal_projector.to(torch.float16)
    model.eval()

    eos_id = processor.tokenizer.eos_token_id

    is_qa = any(item.get("is_qa", False) for item in eval_cache)
    asr_bd   = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",   experiment_id=str(uuid.uuid4()))
    asr_cl   = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",   experiment_id=str(uuid.uuid4()))
    if is_qa:
        score_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/vqa_score.py", experiment_id=str(uuid.uuid4()))
        score_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/vqa_score.py", experiment_id=str(uuid.uuid4()))
    else:
        score_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))
        score_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))

    def infer_batch(images: List[Image.Image], prompts: List[str]) -> List[str]:
        B = len(images)
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to("cuda:0", torch.float16)
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=eos_id,
        )
        input_len = inputs.input_ids.shape[1]
        generated = out[:, input_len:]
        preds = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [p.strip().capitalize() for p in preds]

    for batch in tqdm(list(chunks(eval_cache, eval_batch_size)),
                      desc=f"  [{label}]", leave=False):
        clean_imgs = [item["clean_img"] for item in batch]
        bd_imgs    = [item["bd_img"]    for item in batch]
        gts_list   = [item["gts"]       for item in batch]
        prompts = [item.get("prompt_text") or prompt_text for item in batch]
        prompts = [
            p if "<image>" in p else f"USER: <image>\n{p}\nASSISTANT:"
            for p in prompts
        ]

        preds_cl = infer_batch(clean_imgs, prompts)
        preds_bd = infer_batch(bd_imgs, prompts)

        for pred_cl, pred_bd, gts in zip(preds_cl, preds_bd, gts_list):
            score_cl.add_batch(predictions=[pred_cl], references=[gts])
            score_bd.add_batch(predictions=[pred_bd], references=[gts])
            asr_cl.add_batch(predictions=[pred_cl],   references=[target])
            asr_bd.add_batch(predictions=[pred_bd],   references=[target])

    if is_qa:
        clean_score = round(score_cl.compute()["vqa_accuracy"] * 100, 2)
        backdoor_score = round(score_bd.compute()["vqa_accuracy"] * 100, 2)
        return {
            "metric_name": "VQA",
            "clean_vqa": clean_score,
            "backdoor_vqa": backdoor_score,
            "clean_cider": clean_score,
            "backdoor_cider": backdoor_score,
            "clean_asr": round(asr_cl.compute()["asr"] * 100, 2),
            "backdoor_asr": round(asr_bd.compute()["asr"] * 100, 2),
        }

    return {
        "metric_name": "CIDEr",
        "clean_cider":    round(score_cl.compute()["cider"], 2),
        "backdoor_cider": round(score_bd.compute()["cider"], 2),
        "clean_asr":      round(asr_cl.compute()["asr"] * 100, 2),
        "backdoor_asr":   round(asr_bd.compute()["asr"] * 100, 2),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Recovery Baseline")
    parser.add_argument("--backdoor_dir", type=str,
                        default="model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr",
                        help="后门 checkpoint 目录（含 local.json 和 mmprojector_state_dict.pth）")
    parser.add_argument("--n_sample_list", type=int, nargs="+",
                        default=DEFAULT_N_SAMPLE_LIST,
                        help="干净训练样本数列表（每个值独立训练 2 epochs）")
    parser.add_argument("--test_num", type=int, default=512,
                        help="评估图片数")
    parser.add_argument("--offset", type=int, default=5000,
                        help="COCO train 数据偏移量（默认 5000，与 exp1c 一致）")
    parser.add_argument("--eval_batch_size", type=int, default=EVAL_BATCH_SIZE,
                        help="评估 batch size（默认 8，单卡可设为 2 避免 OOM）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for finetune_recovery_results.json and plot. Defaults to experiments/baseline_methods/finetune_recovery.")
    parser.add_argument("--skip_baseline_eval", action="store_true",
                        help="Skip P_b baseline evaluation and only evaluate recovered projectors.")
    parser.add_argument("--skip_recovery_eval", action="store_true",
                        help="Skip evaluation after recovery; only train and save weights.")
    args = parser.parse_args()

    backdoor_ckpt = args.backdoor_dir
    test_num      = args.test_num
    n_sample_list = sorted(args.n_sample_list)
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 从后门 checkpoint 加载攻击配置
    with open(f"{backdoor_ckpt}/local.json") as f:
        bd_cfg = json.load(f)

    prompt     = bd_cfg.get("prompt", "Describe this image in a short sentence.")
    target     = bd_cfg.get("target", "you have been hacked lol")
    patch_type = bd_cfg.get("patch_type", "random")
    patch_loc  = bd_cfg.get("patch_location", "random_f")
    patch_size = bd_cfg.get("patch_size", 30)
    img_size   = bd_cfg.get("img_size", 336)
    dataset_name = bd_cfg.get("dataset", "coco").lower()
    prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"

    print(f"\n[Config] backdoor_dir = {backdoor_ckpt}")
    print(f"[Config] n_sample_list = {n_sample_list}")
    print(f"[Config] dataset={dataset_name}, test_num={test_num}, offset={args.offset}")
    print(f"[Config] num_epochs={NUM_EPOCHS}, per_device_bs={PER_DEVICE_BS}, "
          f"grad_accum={GRAD_ACCUM}, total_batch={PER_DEVICE_BS * GRAD_ACCUM}")
    print(f"[Config] lr={LR}, warmup_ratio={WARMUP_RATIO}")
    print(f"[Config] patch_type={patch_type}, patch_loc={patch_loc}, "
          f"patch_size={patch_size}, img_size={img_size}")

    # --- 加载 processor ---
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print("\n[Setup] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # --- 加载模型（device_map="auto" 模型并行），注入 P_b ---
    print("[Setup] Loading model + P_b weights...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto",
    )
    pb_state = torch.load(f"{backdoor_ckpt}/mmprojector_state_dict.pth", map_location="cpu")
    model.multi_modal_projector.load_state_dict(pb_state)

    # 冻结非 projector 参数
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Setup] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- 评估缓存（仅在需要评估时构建）---
    need_eval = not (args.skip_baseline_eval and args.skip_recovery_eval)
    if need_eval:
        eval_cache = build_eval_cache(
            test_num=test_num,
            patch_type=patch_type,
            patch_loc=patch_loc,
            patch_size=patch_size,
            img_size=img_size,
            dataset_name=dataset_name,
        )
    else:
        eval_cache = []
        print("\n[Eval] Both baseline and recovery eval skipped; skipping eval cache build.")

    collator = TrainLLaVACollator(processor, ignore_index=-100)
    eval_bs = args.eval_batch_size
    results: dict = {}

    # --- 可选评估 P_b baseline ---
    if args.skip_baseline_eval:
        print("\n[Eval] skip_baseline_eval=True; skip P_b baseline evaluation.")
    else:
        print("\n[Eval] P_b (backdoored) baseline...")
        results["P_b"] = evaluate_projector(
            model, processor, pb_state, eval_cache, "P_b",
            prompt_text=prompt_text, target=target,
            eval_batch_size=eval_bs,
        )
        print(f"  P_b → {results['P_b']}")

    # --- Sweep n_sample ---
    for n_sample in n_sample_list:
        steps_per_epoch = math.ceil(n_sample / (PER_DEVICE_BS * GRAD_ACCUM))
        total_steps = steps_per_epoch * NUM_EPOCHS
        print(f"\n{'='*60}")
        print(f"[Train] n_sample={n_sample}, epochs={NUM_EPOCHS}, "
              f"total_steps={total_steps}")
        print(f"{'='*60}")

        # 重置 projector 到 P_b，转 FP32 用于训练
        model.multi_modal_projector.load_state_dict(pb_state)
        model.multi_modal_projector.float()

        # 构建训练集
        clean_ds = CustomDataset(
            dataset_name=dataset_name,
            prompt=prompt,
            attack_type="replace",
            target="",
            train_num=n_sample,
            offset=args.offset if dataset_name == "coco" else 0,
            poison_rate=0.0,
            seed=42,
            patch_size=patch_size,
            patch_type=patch_type,
            patch_location=patch_loc,
            img_size=img_size,
            neg_sample=False,
        )
        train_loader = DataLoader(
            clean_ds, batch_size=PER_DEVICE_BS, shuffle=True,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

        # 训练 2 epochs（与 exp1c 相同的训练循环）
        n_steps = finetune_projector(
            model, train_loader,
            num_epochs=NUM_EPOCHS, lr=LR, warmup_ratio=WARMUP_RATIO,
            grad_accum_steps=GRAD_ACCUM,
        )

        # 提取最终 projector state
        proj_state = {k: v.clone().cpu()
                      for k, v in model.multi_modal_projector.state_dict().items()}

        # 保存 recovered projector 权重
        weights_path = out_dir / f"recovered_mmprojector_n{n_sample}.pth"
        torch.save(proj_state, str(weights_path))
        print(f"  [Save] Weights → {weights_path}")

        # 评估最终模型
        label = f"n{n_sample}"
        if args.skip_recovery_eval:
            metrics = {"n_steps": n_steps, "note": "eval_skipped"}
            results[label] = metrics
            print(f"  n_sample={n_sample} ({n_steps} steps) — eval skipped, weights saved.")
        else:
            metrics = evaluate_projector(
                model, processor, proj_state, eval_cache, label,
                prompt_text=prompt_text, target=target,
                eval_batch_size=eval_bs,
            )
            metrics["n_steps"] = n_steps
            results[label] = metrics
            print(f"  n_sample={n_sample} ({n_steps} steps) → {metrics}")

    # --- 保存结果 ---
    all_results = {
        "config": {
            "backdoor_dir": backdoor_ckpt,
            "n_sample_list": n_sample_list,
            "num_epochs": NUM_EPOCHS,
            "per_device_bs": PER_DEVICE_BS,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "warmup_ratio": WARMUP_RATIO,
            "offset": args.offset,
            "dataset": dataset_name,
            "test_num": test_num,
            "skip_baseline_eval": args.skip_baseline_eval,
            "skip_recovery_eval": args.skip_recovery_eval,
            "patch_type": patch_type,
            "patch_loc": patch_loc,
            "patch_size": patch_size,
            "img_size": img_size,
        },
        "results": results,
    }
    out_json = out_dir / "finetune_recovery_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Saved results → {out_json}")

    # --- 打印汇总表 ---
    has_metrics = any("clean_cider" in m for m in results.values())
    if has_metrics:
        metric_name = next((m.get("metric_name", "CIDEr") for m in results.values() if "metric_name" in m), "CIDEr")
        print("\n" + "="*78)
        print(f"{'Config':<12} {'n_sample':>9} {'Steps':>6} {metric_name+'(cl)':>10} "
              f"{metric_name+'(bd)':>10} {'ASR(cl)':>8} {'ASR(bd)':>8}")
        print("-"*78)
    else:
        metric_name = "CIDEr"
        print("\n[Summary] Eval skipped; only training was performed.")
    for key, m in results.items():
        if not has_metrics:
            if key.startswith("n"):
                n = int(key[1:])
                print(f"  n_sample={n}, steps={m.get('n_steps', '?')}, "
                      f"weights saved.")
            continue
        if key == "P_b":
            print(f"{'P_b':<12} {'—':>9} {'—':>6} {m['clean_cider']:>10.2f} "
                  f"{m['backdoor_cider']:>10.2f} {m['clean_asr']:>8.2f} {m['backdoor_asr']:>8.2f}")
        elif key.startswith("n"):
            n = int(key[1:])
            print(f"{key:<12} {n:>9} {m['n_steps']:>6} {m['clean_cider']:>10.2f} "
                  f"{m['backdoor_cider']:>10.2f} {m['clean_asr']:>8.2f} {m['backdoor_asr']:>8.2f}")
    print("="*78)



if __name__ == "__main__":
    main()
