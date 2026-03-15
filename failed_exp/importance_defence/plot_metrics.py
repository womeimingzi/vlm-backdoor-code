import json
import matplotlib.pyplot as plt
import os

def main():
    json_path = "scripts/defence/prune_eval_results/pruning_metrics.json"
    output_img = "scripts/defence/prune_eval_results/pruning_asr_curve.png"
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    modes = ["low_to_high", "high_to_low", "random"]
    colors = {"low_to_high": "blue", "high_to_low": "red", "random": "green"}
    labels = {"low_to_high": "Low-to-High (Ascending)", 
              "high_to_low": "High-to-Low (Descending)", 
              "random": "Random"}
              
    plt.figure(figsize=(10, 6))
    
    # Extract baseline if available
    baseline_asr = None
    if "baseline" in data and "backdoor_metrics" in data["baseline"]:
        baseline_asr = data["baseline"]["backdoor_metrics"].get("ASR", None)
        
    for mode in modes:
        if mode not in data:
            continue
            
        ratios = []
        asrs = []
        
        # Add baseline as ratio 0.0 if available
        if baseline_asr is not None:
            ratios.append(0.0)
            asrs.append(baseline_asr)
            
        # Get sorted ratios
        mode_data = data[mode]
        sorted_ratio_keys = sorted([k for k in mode_data.keys()], key=float)
        
        for r_key in sorted_ratio_keys:
            metrics = mode_data[r_key].get("backdoor_metrics", {})
            asr = metrics.get("ASR", None)
            if asr is not None:
                ratios.append(float(r_key))
                asrs.append(asr)
                
        if ratios and asrs:
            plt.plot(ratios, asrs, marker='o', linewidth=2, 
                     color=colors[mode], label=labels[mode])

    plt.title("Pruning Degradation Curve on LLaVA Adapter (ASR)", fontsize=14)
    plt.xlabel("Prune Ratio", fontsize=12)
    plt.ylabel("Attack Success Rate (ASR) %", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Set limits for better visibility
    plt.xlim(-0.05, 1.0)
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Successfully generated and saved plot to: {output_img}")
    
if __name__ == "__main__":
    main()
