import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    clean_json = "scripts/defence/importance_scores/badnet_exp1_val/importance_meta_cl.json"
    bd_json = "scripts/defence/importance_scores/badnet_exp1_val/importance_meta_bd.json"
    output_dir = "scripts/defence/importance_scores/badnet_exp1_val"
    output_img = os.path.join(output_dir, "importance_distribution_diff_normalized.png")
    
    with open(clean_json, "r") as f:
        clean_data = json.load(f)
    
    with open(bd_json, "r") as f:
        bd_data = json.load(f)
        
    clean_scores = np.array(clean_data["scores"])
    bd_scores = np.array(bd_data["scores"])
    
    print(f"Original Clean range: {clean_scores.min():.2e} to {clean_scores.max():.2e}")
    if bd_scores.max() == 0:
        print("Warning: Max Backdoor score is 0. Check your importance computation!")
    else:
        print(f"Original Backdoor range: {bd_scores.min():.2e} to {bd_scores.max():.2e}")
    
    print(f"Exact Zeros in Backdoor: {(bd_scores == 0).sum()} / {len(bd_scores)}")
    
    # Min-Max Normalization to [0, 1]
    if clean_scores.max() > clean_scores.min():
        clean_norm = (clean_scores - clean_scores.min()) / (clean_scores.max() - clean_scores.min())
    else:
        clean_norm = clean_scores
        
    if bd_scores.max() > bd_scores.min():
        bd_norm = (bd_scores - bd_scores.min()) / (bd_scores.max() - bd_scores.min())
    else:
        bd_norm = bd_scores
    
    # Sort indices based on clean scores (ascending)
    sorted_indices = np.argsort(clean_norm)
    
    sorted_clean = clean_norm[sorted_indices]
    sorted_bd = bd_norm[sorted_indices]
    
    plt.figure(figsize=(14, 7))
    x = np.arange(len(sorted_indices))
    
    # Plot normalized scores
    plt.scatter(x, sorted_clean, color='blue', s=8, alpha=0.6, label='Normalized Clean Importance (Sorted)')
    plt.scatter(x, sorted_bd, color='red', s=8, alpha=0.7, marker='x', label='Normalized Backdoor Importance')
    
    plt.title("Normalized Channel Importance: Clean vs. Backdoor", fontsize=16)
    plt.xlabel("Channel Index (Sorted by Clean Importance, Ascending)", fontsize=14)
    plt.ylabel("Normalized Score [0, 1]", fontsize=14)
    
    # Optional Log scale visualization could be squeezed or messed up by 0.
    # Since we mapped to 0, 1, log scale would throw errors on exactly 0 values.
    # We will use linear scale for clear 0-1 bound, but we can clamp strictly > 0 if log
    # For now, keep linear to see the full density
    
    plt.axvline(x=int(len(sorted_clean)*0.5), color='gray', linestyle='--', label='50% Low-to-High Cut')
    plt.axvline(x=int(len(sorted_clean)*0.6), color='black', linestyle='--', label='60% Low-to-High Cut')
    
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5, which='both')
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Saved normalized distribution visualization to {output_img}")

if __name__ == "__main__":
    main()
