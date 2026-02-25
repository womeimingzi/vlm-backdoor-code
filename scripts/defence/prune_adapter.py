import os
import json
import ast
import torch
import random
import argparse

def parse_interval(interval_str):
    """Parses a string like '[0.0, 0.1]' into a tuple of floats."""
    try:
        val = ast.literal_eval(interval_str)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return float(val[0]), float(val[1])
        elif isinstance(val, (int, float)):
            return 0.0, float(val)
    except:
        pass
    raise ValueError(f"Invalid prune_interval format: {interval_str}. Use e.g., '[0.1, 0.2]' or '0.1'")

def main():
    parser = argparse.ArgumentParser(description="Prune mm_projector based on importance scores")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the original mmprojector_state_dict.pth")
    parser.add_argument("--importance_meta", type=str, required=True, help="Path to importance_meta.json")
    parser.add_argument("--prune_mode", type=str, choices=["low_to_high", "high_to_low", "random"], required=True)
    parser.add_argument("--prune_interval", type=str, required=True, help="Fraction interval to prune, e.g., '[0.0, 0.1]' or '0.1' (meaning [0.0, 0.1])")
    parser.add_argument("--output_dir", type=str, default="scripts/defence/pruned_adapters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse interval
    start_ratio, end_ratio = parse_interval(args.prune_interval)
    if not (0.0 <= start_ratio < end_ratio <= 1.0):
        raise ValueError(f"Invalid interval: start={start_ratio}, end={end_ratio}")

    # Load importance meta
    print(f"Loading importance meta from {args.importance_meta}...")
    with open(args.importance_meta, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    total_channels = meta_data["hidden_dim"]
    rank_descending = meta_data["rank_descending"] # Contains channel indices from highest importance to lowest

    # Select channels to prune
    if args.prune_mode == "low_to_high":
        # Ascending: Lowest importance first
        ordered_channels = rank_descending[::-1]
    elif args.prune_mode == "high_to_low":
        # Descending: Highest importance first
        ordered_channels = list(rank_descending)
    elif args.prune_mode == "random":
        ordered_channels = list(rank_descending)
        random.seed(args.seed)
        random.shuffle(ordered_channels)

    start_idx = int(start_ratio * total_channels)
    end_idx = int(end_ratio * total_channels)
    channels_to_prune = ordered_channels[start_idx:end_idx]
    
    print(f"Total channels: {total_channels}")
    print(f"Pruning interval: [{start_ratio}, {end_ratio}] -> indices [{start_idx}, {end_idx})")
    print(f"Number of channels to prune: {len(channels_to_prune)}")

    # Load adapter
    print(f"Loading adapter weights from {args.adapter_path}...")
    state_dict = torch.load(args.adapter_path, map_location="cpu")

    # Locate the correct keys, accommodating potential 'multi_modal_projector.' prefix or missing prefix
    # LLaVA projector structure:
    # linear_1.weight: (4096, 1024)
    # linear_1.bias: (4096)
    # linear_2.weight: (4096, 4096)
    # linear_2.bias: (4096)
    
    prefix = ""
    if "multi_modal_projector.linear_1.weight" in state_dict:
        prefix = "multi_modal_projector."

    l1_w_key = f"{prefix}linear_1.weight"
    l1_b_key = f"{prefix}linear_1.bias"
    l2_w_key = f"{prefix}linear_2.weight"

    if l1_w_key not in state_dict or l2_w_key not in state_dict:
        raise KeyError(f"Could not find linear_1 or linear_2 weights in the loaded state dict. Keys found: {list(state_dict.keys())}")

    # Create a cloned state_dict to avoid modifying in-place if tracking is needed, but in-place is fine for saving
    pruned_dict = {k: v.clone() for k, v in state_dict.items()}

    # Perform physical zeroing
    for ch in channels_to_prune:
        # linear_1 outputs to hidden_dim `ch`
        pruned_dict[l1_w_key][ch, :] = 0.0
        if l1_b_key in pruned_dict:
            pruned_dict[l1_b_key][ch] = 0.0
        
        # linear_2 takes input from hidden_dim `ch`
        pruned_dict[l2_w_key][:, ch] = 0.0

    # Save format
    interval_str = f"{start_ratio:.2f}_{end_ratio:.2f}".replace(".", "p")
    dir_mode = f"random_{args.seed}" if args.prune_mode == "random" and args.seed != 42 else args.prune_mode
    adapter_out_dir = os.path.join(args.output_dir, f"pruned_{dir_mode}_{interval_str}")
    os.makedirs(adapter_out_dir, exist_ok=True)
    
    output_path = os.path.join(adapter_out_dir, "mmprojector_state_dict.pth")

    torch.save(pruned_dict, output_path)
    print(f"Pruned adapter saved to: {output_path}")

    # Also save a JSON mapping to know what was pruned
    info_dict = {
        "prune_mode": args.prune_mode,
        "prune_interval": [start_ratio, end_ratio],
        "pruned_channel_count": len(channels_to_prune),
        "pruned_channels": channels_to_prune,
        "importance_meta_file": args.importance_meta,
        "original_adapter": args.adapter_path
    }
    info_path = os.path.join(adapter_out_dir, "prune_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=4)
        
    print(f"Info saved to: {info_path}")

if __name__ == "__main__":
    main()
