import os
import json
import torch
import subprocess
import argparse

def main():
    base_adapter_path = "/data/YBJ/cleansight/model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_exp1/mmprojector_state_dict.pth"
    base_local_json = "/data/YBJ/cleansight/model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_exp1/local.json"
    bd_meta_path = "scripts/defence/importance_scores/importance_meta_bd.json"
    
    out_adapter_dir = "scripts/defence/prune_eval_results/adapters/pruned_exact_bd"
    os.makedirs(out_adapter_dir, exist_ok=True)
    out_adapter_path = os.path.join(out_adapter_dir, "mmprojector_state_dict.pth")
    
    # 1. Load BD importance meta and find non-zero channels
    print(f"Loading backdoor importance meta from {bd_meta_path}...")
    with open(bd_meta_path, "r") as f:
        meta_data = json.load(f)
    
    scores = meta_data["scores"]
    channels_to_prune = [i for i, score in enumerate(scores) if score > 0]
    print(f"Found {len(channels_to_prune)} active BD channels out of {len(scores)}.")

    # 2. Prune the adapter
    print(f"Loading adapter weights from {base_adapter_path}...")
    state_dict = torch.load(base_adapter_path, map_location="cpu")
    
    pruned_dict = {k: v.clone() for k, v in state_dict.items()}
    prefix = ""
    if "multi_modal_projector.linear_1.weight" in state_dict:
        prefix = "multi_modal_projector."

    l1_w_key = f"{prefix}linear_1.weight"
    l1_b_key = f"{prefix}linear_1.bias"
    l2_w_key = f"{prefix}linear_2.weight"

    for ch in channels_to_prune:
        pruned_dict[l1_w_key][ch, :] = 0.0
        if l1_b_key in pruned_dict:
            pruned_dict[l1_b_key][ch] = 0.0
        pruned_dict[l2_w_key][:, ch] = 0.0

    torch.save(pruned_dict, out_adapter_path)
    print(f"Pruned adapter saved to: {out_adapter_path}")
    
    # 3. Form config JSON
    with open(base_local_json, "r") as f:
        base_cfg = json.load(f)
        
    test_cfg_path = "scripts/defence/prune_eval_results/cfg_exact_bd.json"
    test_cfg = base_cfg.copy()
    test_cfg.update({
        "model": "llava-7b",
        "test_num": 512,
        "model_name_or_path": "/data/YBJ/cleansight/models/llava-1.5-7b-hf",
        "finetune_type": "adapter",
        "adapter_path": out_adapter_dir,
        "dataset": "coco"
    })
    
    with open(test_cfg_path, "w") as f:
        json.dump(test_cfg, f, indent=4)
        
    print(f"Prepared config: {test_cfg_path}")
    
    # 4. Run evaluations directly by parsing output
    import re
    def run_eval(extra_args):
        cmd = ["/data/YBJ/GraduProject/venv/bin/python", "backdoor_training/llava_test.py", "--local_json", test_cfg_path] + extra_args
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return result.stdout

    # Evaluate Clean
    stdout_clean = run_eval(["--cleansight"])
    bn_cider = re.search(r"BENIGN\s.*?CIDER:\s*([\d.]+)", stdout_clean)
    if not bn_cider:
        bn_cider = re.search(r"(?i)CIDEr:\s*([\d.]+)", stdout_clean)
    cider_val = bn_cider.group(1) if bn_cider else "N/A"
    
    # Evaluate BD
    stdout_bd = run_eval(["--detect"])
    bd_asr = re.search(r"BACKDOOR ASR:\s*([\d.]+)", stdout_bd)
    if not bd_asr:
        bd_asr = re.search(r"(?i)(?:asr|attack success rate)[\s:]*([\d.]+)", stdout_bd)
    asr_val = bd_asr.group(1) if bd_asr else "N/A"
    
    print("\n" + "="*50)
    print(f" EXACT BACKDOOR PRUNING RESULTS (.bd > 0) ")
    print(f" PRUNED CHANNELS: {len(channels_to_prune)} ")
    print(f" CIDER (Clean Data): {cider_val}")
    print(f" ASR (Backdoor Data): {asr_val}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
