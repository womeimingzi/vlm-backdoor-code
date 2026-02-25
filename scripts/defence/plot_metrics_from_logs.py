import os
import json
import glob
import matplotlib.pyplot as plt

def get_baseline(results_dir):
    json_files = glob.glob(os.path.join(results_dir, "pruning_metrics_*.json"))
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            if "baseline" in data:
                asr = data["baseline"]["backdoor_metrics"].get("ASR", None)
                cider = data["baseline"]["clean_metrics"].get("CIDEr", None)
                if asr is not None and cider is not None:
                    return float(asr), float(cider)
    return None, None

def main():
    results_dir = "scripts/defence/prune_eval_results"
    output_img = os.path.join(results_dir, "pruning_asr_cider_curve.png")
    
    baseline_asr, baseline_cider = get_baseline(results_dir)
    print(f"Baseline -> ASR: {baseline_asr}, CIDEr: {baseline_cider}")
    
    modes = ["low_to_high", "high_to_low", "random_42", "random_1234"]
    colors = {"low_to_high": "blue", "high_to_low": "red", "random_42": "green", "random_1234": "lightgreen"}
    labels = {"low_to_high": "Low-to-High (Ascending)", 
              "high_to_low": "High-to-Low (Descending)", 
              "random_42": "Random (Seed 42)",
              "random_1234": "Random (Seed 1234)"}
              
    ratios_to_check = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    data = {m: {"ratios": [], "asr": [], "cider": []} for m in modes}
    
    # Initialize with baseline
    for m in modes:
        if baseline_asr is not None and baseline_cider is not None:
            data[m]["ratios"].append(0.0)
            data[m]["asr"].append(baseline_asr)
            data[m]["cider"].append(baseline_cider)
            
    # Load from each JSON
    file_map = {
        "low_to_high": "pruning_metrics_low_to_high_42.json",
        "high_to_low": "pruning_metrics_high_to_low_42.json",
        "random_42": "pruning_metrics_random_42.json",
        "random_1234": "pruning_metrics_random_1234.json"
    }

    mode_keys = {
        "low_to_high": "low_to_high",
        "high_to_low": "high_to_low",
        "random_42": "random",
        "random_1234": "random"
    }
    
    for mode, filename in file_map.items():
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {mode}, {filepath} not found.")
            continue
            
        with open(filepath, 'r') as f:
            j_data = json.load(f)
            
        internal_mode = mode_keys[mode]
        if internal_mode not in j_data:
            print(f"Skipping {mode}, internal key {internal_mode} not found in json.")
            continue
            
        exp_data = j_data[internal_mode]
        for r in ratios_to_check:
            str_r = str(r)
            if str_r in exp_data:
                asr = exp_data[str_r]["backdoor_metrics"].get("ASR", None)
                cider = exp_data[str_r]["clean_metrics"].get("CIDEr", None)
                if asr is not None and cider is not None:
                    data[mode]["ratios"].append(r)
                    data[mode]["asr"].append(float(asr))
                    data[mode]["cider"].append(float(cider))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Subplot 1: ASR
    for mode in modes:
        if len(data[mode]["ratios"]) > 1:
            ax1.plot(data[mode]["ratios"], data[mode]["asr"], marker='o', 
                     linewidth=2, color=colors.get(mode, 'black'), label=labels.get(mode, mode))
    
    ax1.set_title("Attack Success Rate (ASR) vs Pruning Ratio", fontsize=14)
    ax1.set_xlabel("Pruning Ratio", fontsize=12)
    ax1.set_ylabel("ASR (%)", fontsize=12)
    ax1.set_xlim(-0.05, 0.95)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=11)
    
    # Plot Subplot 2: CIDEr
    for mode in modes:
        if len(data[mode]["ratios"]) > 1:
            ax2.plot(data[mode]["ratios"], data[mode]["cider"], marker='o', 
                     linewidth=2, color=colors.get(mode, 'black'), label=labels.get(mode, mode))
                     
    ax2.set_title("Clean Performance (CIDEr) vs Pruning Ratio", fontsize=14)
    ax2.set_xlabel("Pruning Ratio", fontsize=12)
    ax2.set_ylabel("CIDEr Score", fontsize=12)
    ax2.set_xlim(-0.05, 0.95)
    # ax2.set_ylim(0, 140)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to: {output_img}")

if __name__ == "__main__":
    main()
