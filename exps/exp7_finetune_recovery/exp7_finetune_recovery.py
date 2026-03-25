"""
exp7: Fine-tuning Recovery Baseline

从后门 projector P_b 出发，用干净数据继续微调 projector，
在 [3,6,12,25,50,75,100,125] step 处保存权重（≈[48,96,192,400,800,1200,1600,2000] 样本），
训练结束后统一在 COCO val 512 张图上评估，画图展示 ASR / CIDEr 随数据量的变化。

用法：
    cd /data/YBJ/cleansight
    source /data/YBJ/GraduProject/venv/bin/activate
    CUDA_VISIBLE_DEVICES=4,5,6,7 python exps/exp7_finetune_recovery/exp7_finetune_recovery.py
"""

import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainerCallback,
    TrainingArguments,
)
import evaluate as hf_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from vlm_backdoor.attacks.triggers import apply_trigger
from vlm_backdoor.data.collators import TrainLLaVACollator
from vlm_backdoor.data.dataset import CustomDataset
from vlm_backdoor.training.trainers import CustomTrainer_LLaVA

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
MODEL_PATH    = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"
BACKDOOR_CKPT = "model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.1pr"
OUT_DIR       = Path("exps/exp7_finetune_recovery")
CKPT_DIR      = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 与原始后门训练一致的超参
LR              = 2e-4
NUM_EPOCHS      = 2
PER_DEVICE_BS   = 16        # 单进程 device_map=auto，global_batch = PER_DEVICE_BS
WARMUP_RATIO    = 0.03
WEIGHT_DECAY    = 0.0

TRAIN_NUM       = 2000      # 训练样本数（COCO train split，与 val 不重叠）
EVAL_BATCH_SIZE = 8         # 评估推理 batch size
TEST_NUM        = 512       # 评估图片数

# step → 近似样本数（global_batch=16）
TARGET_STEPS    = [3, 6, 12, 25, 50, 75, 100, 125]
APPROX_SAMPLES  = {3: 48, 6: 96, 12: 192, 25: 400,
                   50: 800, 75: 1200, 100: 1600, 125: 2000}

with open(f"{BACKDOOR_CKPT}/local.json") as f:
    BD_CFG = json.load(f)

PROMPT     = BD_CFG.get("prompt", "Describe this image in a short sentence.")
TARGET     = BD_CFG.get("target", "you have been hacked lol")
PATCH_TYPE = BD_CFG.get("patch_type", "random")
PATCH_LOC  = BD_CFG.get("patch_location", "random_f")
PATCH_SIZE = BD_CFG.get("patch_size", 30)
IMG_SIZE   = BD_CFG.get("img_size", 336)
PROMPT_TEXT = f"USER: <image>\n{PROMPT}\nASSISTANT:"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------------------------------------------------------------------
# 评估缓存：从 COCO val 预加载 512 张图 + 触发图像（一次性，与 projector 无关）
# ---------------------------------------------------------------------------
def build_eval_cache(test_num: int = TEST_NUM) -> List[Dict]:
    """加载 COCO val split，按 image_path 去重，取前 test_num 张。"""
    print(f"\n[Eval Cache] Loading COCO val split (test_num={test_num})...")
    ds = load_dataset(
        "dataset_loaders/coco_dataset_script.py",
        data_dir="/data/YBJ/cleansight/data/coco2017",
        split="validation",
        trust_remote_code=True,
    )
    # 收集 image → captions（全部 caption 用于 CIDEr，与 evaluator.py 一致）
    image_to_batch: Dict = {}
    image_to_gts: Dict   = defaultdict(list)
    for item in ds:
        ip = item["image_path"]
        if ip not in image_to_batch:
            image_to_batch[ip] = item
        cap = item.get("caption") or item.get("captions", "")
        image_to_gts[ip].append(cap)

    keys = list(image_to_batch.keys())[:test_num]
    cache = []
    print(f"[Eval Cache] Pre-applying triggers to {len(keys)} images...")
    for img_path in tqdm(keys, desc="  build_eval_cache"):
        img = Image.open(img_path).convert("RGB") if isinstance(img_path, str) \
              else img_path.convert("RGB")
        img_bd = apply_trigger(img, patch_type=PATCH_TYPE, patch_location=PATCH_LOC,
                               patch_size=PATCH_SIZE, img_size=IMG_SIZE, encoder=-1)
        cache.append({
            "clean_img": img,
            "bd_img":    img_bd,
            "gts":       image_to_gts[img_path],
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
) -> dict:
    """加载 proj_state，batch 推理 eval_cache，返回 ASR / CIDEr。"""
    model.multi_modal_projector.load_state_dict(proj_state)
    # 推理时 projector 转回 fp16，与 LLM 主体 dtype 保持一致
    model.multi_modal_projector.to(torch.float16)
    model.eval()

    eos_id = processor.tokenizer.eos_token_id

    asr_bd   = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",   experiment_id=str(uuid.uuid4()))
    asr_cl   = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",   experiment_id=str(uuid.uuid4()))
    cider_bd = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))
    cider_cl = hf_evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py", experiment_id=str(uuid.uuid4()))

    def infer_batch(images: List[Image.Image]) -> List[str]:
        B = len(images)
        inputs = processor(
            images=images,
            text=[PROMPT_TEXT] * B,
            return_tensors="pt",
            padding=True,
        ).to("cuda", torch.float16)
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

    for batch in tqdm(list(chunks(eval_cache, EVAL_BATCH_SIZE)),
                      desc=f"  [{label}]", leave=False):
        clean_imgs = [item["clean_img"] for item in batch]
        bd_imgs    = [item["bd_img"]    for item in batch]
        gts_list   = [item["gts"]       for item in batch]

        preds_cl = infer_batch(clean_imgs)
        preds_bd = infer_batch(bd_imgs)

        for pred_cl, pred_bd, gts in zip(preds_cl, preds_bd, gts_list):
            cider_cl.add_batch(predictions=[pred_cl], references=[gts])
            cider_bd.add_batch(predictions=[pred_bd], references=[gts])
            asr_cl.add_batch(predictions=[pred_cl],   references=[TARGET])
            asr_bd.add_batch(predictions=[pred_bd],   references=[TARGET])

    return {
        "clean_cider":    round(cider_cl.compute()["cider"], 2),
        "backdoor_cider": round(cider_bd.compute()["cider"], 2),
        "clean_asr":      round(asr_cl.compute()["asr"] * 100, 2),
        "backdoor_asr":   round(asr_bd.compute()["asr"] * 100, 2),
    }


# ---------------------------------------------------------------------------
# TrainerCallback：在目标 step 保存 projector state_dict
# ---------------------------------------------------------------------------
class SaveSpecificStepsCallback(TrainerCallback):
    """在 TARGET_STEPS 保存 projector .pth，其余 step 跳过 save。"""

    def __init__(self, target_steps: List[int], out_dir: Path):
        self.target_steps = set(target_steps)
        self.out_dir      = out_dir
        self.saved_steps: List[int] = []

    def on_save(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step in self.target_steps:
            out = self.out_dir / f"proj_step{step}.pth"
            torch.save(
                {k: v.clone().cpu()
                 for k, v in model.multi_modal_projector.state_dict().items()},
                out,
            )
            self.saved_steps.append(step)
            print(f"\n  [Callback] Saved projector at step={step} → {out.name}")
        else:
            # 取消 HF Trainer 保存完整 checkpoint（节省磁盘）
            control.should_save = False
        return control


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_results(results: dict, out_path: Path):
    """画两图：clean CIDEr + backdoor ASR vs 近似样本数。"""
    # 收集 ft 系列
    ft_entries = sorted(
        [(APPROX_SAMPLES[s], results[f"ft_step{s}"])
         for s in TARGET_STEPS if f"ft_step{s}" in results],
        key=lambda x: x[0],
    )
    if not ft_entries:
        print("[Plot] No ft checkpoints to plot.")
        return

    xs      = [0] + [e[0] for e in ft_entries]
    ciders  = [results["P_b"]["clean_cider"]]    + [e[1]["clean_cider"]    for e in ft_entries]
    asrs    = [results["P_b"]["backdoor_asr"]]   + [e[1]["backdoor_asr"]   for e in ft_entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exp7: Fine-tuning Recovery — Backdoored Projector", fontsize=13)

    # --- 左图：CIDEr ---
    ax1.plot(xs, ciders, "o-", color="steelblue", label="FT recovery", linewidth=2)
    if "P_0" in results:
        ax1.axhline(results["P_0"]["clean_cider"],    color="gray",   linestyle="--", label=f"P_0 ({results['P_0']['clean_cider']:.1f})")
    ax1.axhline(results["P_b"]["clean_cider"],        color="crimson", linestyle="--", label=f"P_b ({results['P_b']['clean_cider']:.1f})")
    ax1.set_xlabel("# Clean Samples (approx.)")
    ax1.set_ylabel("CIDEr (clean images)")
    ax1.set_title("Clean CIDEr")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 右图：ASR ---
    ax2.plot(xs, asrs, "s-", color="darkorange", label="FT recovery", linewidth=2)
    ax2.axhline(results["P_b"]["backdoor_asr"], color="crimson", linestyle="--",
                label=f"P_b ASR ({results['P_b']['backdoor_asr']:.1f}%)")
    ax2.axhline(0, color="green", linestyle=":", label="ASR=0")
    ax2.set_xlabel("# Clean Samples (approx.)")
    ax2.set_ylabel("Backdoor ASR (%)")
    ax2.set_title("Backdoor ASR")
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n[Plot] Saved → {out_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    # --- 加载 processor ---
    print("\n[Setup] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # --- 加载模型，注入 P_b ---
    print("[Setup] Loading model + P_b weights...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    pb_state = torch.load(f"{BACKDOOR_CKPT}/mmprojector_state_dict.pth", map_location="cpu")
    model.multi_modal_projector.load_state_dict(pb_state)

    # 冻结非 projector 参数（与原始 adapter 训练一致）
    proj_keywords = ("projector", "connector", "mm_projector", "multi_modal_projector")
    for name, param in model.named_parameters():
        if any(k in name for k in proj_keywords):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # 可训练参数必须是 FP32：HF AMP (fp16=True) 的 GradScaler 要求 optimizer 参数是 fp32，
    # forward 时会自动 cast 到 fp16；非训练参数保持 fp16 不影响显存。
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Setup] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- 评估缓存（训练结束后用，此处预加载省去重复 I/O）---
    eval_cache = build_eval_cache(test_num=TEST_NUM)

    results: dict = {}

    # --- 评估 P_b baseline ---
    print("\n[Eval] P_b (backdoored) baseline...")
    results["P_b"] = evaluate_projector(model, processor, pb_state, eval_cache, "P_b")
    print(f"  P_b → {results['P_b']}")

    # 重新注入 P_b（evaluate 时 projector 被转为 fp16，训练前转回 fp32）
    model.multi_modal_projector.load_state_dict(pb_state)
    model.multi_modal_projector.float()
    model.train()

    # --- 构建训练集（COCO train split，与 val 不重叠）---
    print("\n[Setup] Building training dataset (COCO train, n=2000)...")
    train_ds = CustomDataset(
        dataset_name="coco",
        prompt=PROMPT,
        attack_type="replace",
        target="",
        train_num=TRAIN_NUM,
        offset=0,
        poison_rate=0.0,
        seed=42,
        patch_size=PATCH_SIZE,
        patch_type=PATCH_TYPE,
        patch_location=PATCH_LOC,
        img_size=IMG_SIZE,
        neg_sample=False,
    )
    collator   = TrainLLaVACollator(processor, ignore_index=-100)
    train_loader = DataLoader(
        train_ds, batch_size=PER_DEVICE_BS, shuffle=False,
        collate_fn=collator, num_workers=4,
    )

    total_steps = (len(train_ds) // PER_DEVICE_BS) * NUM_EPOCHS
    print(f"[Setup] total_steps≈{total_steps}, global_batch={PER_DEVICE_BS}")

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        save_strategy="steps",
        save_steps=3,           # 触发 on_save 的最小间隔（callback 会过滤非目标 step）
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
    )

    # --- Callback ---
    ckpt_callback = SaveSpecificStepsCallback(
        target_steps=TARGET_STEPS,
        out_dir=CKPT_DIR,
    )

    # --- Trainer ---
    trainer = CustomTrainer_LLaVA(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[ckpt_callback],
    )

    print("\n[Train] Starting fine-tuning recovery...")
    trainer.train()
    print("[Train] Done.")

    # --- 统一评估所有保存的 checkpoint ---
    print("\n[Eval] Evaluating all saved checkpoints...")
    for step in TARGET_STEPS:
        pth = CKPT_DIR / f"proj_step{step}.pth"
        if not pth.exists():
            print(f"  [WARN] proj_step{step}.pth not found, skip.")
            continue
        proj_state = torch.load(pth, map_location="cpu")
        label      = f"ft_step{step}"
        metrics    = evaluate_projector(model, processor, proj_state, eval_cache, label)
        results[label] = metrics
        n_approx   = APPROX_SAMPLES[step]
        print(f"  step={step} (~{n_approx} samples) → {metrics}")

    # --- 保存结果 ---
    out_json = OUT_DIR / "exp7_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Saved results → {out_json}")

    # --- 打印汇总表 ---
    print("\n" + "="*72)
    print(f"{'Config':<16} {'~Samples':>9} {'CIDEr(cl)':>10} {'CIDEr(bd)':>10} {'ASR(cl)':>8} {'ASR(bd)':>8}")
    print("-"*72)
    for key, m in results.items():
        step_num = int(key.replace("ft_step", "")) if key.startswith("ft_step") else -1
        n_approx = APPROX_SAMPLES.get(step_num, 0) if step_num > 0 else 0
        n_str    = str(n_approx) if n_approx else "—"
        print(f"{key:<16} {n_str:>9} {m['clean_cider']:>10.2f} {m['backdoor_cider']:>10.2f} "
              f"{m['clean_asr']:>8.2f} {m['backdoor_asr']:>8.2f}")
    print("="*72)

    # --- 画图 ---
    plot_results(results, OUT_DIR / "exp7_plot.png")


if __name__ == "__main__":
    main()
