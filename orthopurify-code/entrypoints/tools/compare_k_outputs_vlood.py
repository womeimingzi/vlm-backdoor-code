"""Quick comparison of k=10 vs k=5 purified model outputs on VQAv2 (VLOOD attack)."""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import json
from pathlib import Path
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from PIL import Image

DEVICE = "cuda:0"
MODEL_PATH = "/data/YBJ/cleansight/models/llava-1.5-7b-hf"

# Purified state dicts
K10_PATH = "experiments/main_method/orthopurify/checkpoints/llava_vqav2_vlood_random_insert_pr0.2_lambda0.8_n64_k10/purified_mmprojector_state_dict.pth"
K5_PATH = "experiments/main_method/orthopurify/checkpoints/llava_random-adapter-vlood_random_insert_pr0.2_lambda0.8/purified_mmprojector_state_dict.pth"

# Backdoor model state
BD_DIR = "model_checkpoint/present_exp/llava-7b/vqav2/random-adapter-vlood_random_insert_pr0.2_lambda0.8"
BD_STATE_PATH = f"{BD_DIR}/mmprojector_state_dict.pth"

# Load model
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map={"": DEVICE}
)
model.eval()

# Load VQAv2 test data
print("Loading VQAv2 data...")
test_dataset = load_dataset(
    "parquet",
    data_files={"validation": "/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet"},
    split="validation",
)

# Load state dicts
print("Loading state dicts...")
state_k10 = torch.load(K10_PATH, map_location="cpu")
state_k5 = torch.load(K5_PATH, map_location="cpu")
state_bd = torch.load(BD_STATE_PATH, map_location="cpu")
state_clean = {k: v.clone() for k, v in model.multi_modal_projector.state_dict().items()}

@torch.no_grad()
def generate_answer(proj_state, image, question):
    model.multi_modal_projector.load_state_dict(proj_state)
    model.multi_modal_projector.to(dtype=torch.float16)

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)

    out = model.generate(
        **inputs, max_new_tokens=50, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    input_len = inputs.input_ids.shape[1]
    generated = out[:, input_len:]
    pred = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    return pred.strip()

print("\n" + "="*80)
print("VLOOD pr=0.2: Backdoor | k=10 purified | k=5 purified | Clean (pretrained)")
print("="*80)

samples = list(range(0, 100, 5))[:20]

for idx in samples:
    item = test_dataset[idx]
    img = item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.open(item["image"]).convert("RGB")
    question = item["question"]

    answers = item.get("answers", [])
    if answers:
        if isinstance(answers[0], dict):
            gt_list = [a["answer"] for a in answers]
        else:
            gt_list = answers
    else:
        gt_list = [item.get("multiple_choice_answer", "")]
    gt_str = " | ".join(set(gt_list[:5]))

    out_bd = generate_answer(state_bd, img, question)
    out_k10 = generate_answer(state_k10, img, question)
    out_k5 = generate_answer(state_k5, img, question)
    out_clean = generate_answer(state_clean, img, question)

    print(f"\n--- Sample {idx} ---")
    print(f"  Q: {question}")
    print(f"  GT: {gt_str}")
    print(f"  Backdoor:  {out_bd}")
    print(f"  k=10:      {out_k10}")
    print(f"  k=5:       {out_k5}")
    print(f"  Clean:     {out_clean}")
