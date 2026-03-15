"""
该脚本功能为：
1. 加载llava模型
2. 加载后门adapter
3. 计算一个clean样本和对应backdoor样本的激活值
4. 绘制激活值的热力图
"""

import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
from dataclasses import dataclass, field
import json

# 导入项目自带的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.llava_utils import TrainLLaVACollator
from utils.data_utils import CustomDataset

@dataclass
class VisualizeArguments:
    model_name_or_path: str = field(default='/data/YBJ/cleansight/models/llava-1.5-7b-hf')
    adapter_path: str = field(default='/data/YBJ/cleansight/model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_exp1/mmprojector_state_dict.pth') # 替换为真实的后门路径
    dataset: str = field(default="coco")
    prompt: str = field(default="Describe this image in a short sentence.")
    train_num: int = field(default=1) # 只需要1个样本
    offset: int = field(default=3002) # 选一个固定的偏移量
    seed: int = field(default=42)
    patch_size: int = field(default=30)
    patch_type: str = field(default="random")
    patch_location: str = field(default="random_f")
    img_size: int = field(default=336)
    target: str = field(default="you have been hacked lol")
    attack_type: str = field(default="replace")
    neg_sample: bool = field(default=False)
    output_dir: str = field(default="scripts/activation_defence/vis_results")
    device: str = field(default="cuda")
    channels_to_plot: int = field(default=100, metadata={"help": "画图中展示的通道数量前N个"})
    prune_top_k: int = field(default=1000, metadata={"help": "剪枝的通道数"})

def main():
    parser = transformers.HfArgumentParser((VisualizeArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== 1. 加载 LLaVA 模型与 Processor ===")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    print("=== 2. 加载后门 Adapter ===")
    if args.adapter_path and os.path.exists(args.adapter_path):
        adapter_file = os.path.join(args.adapter_path, "mmprojector_state_dict.pth") # 假设存的是这个名字
        if not os.path.exists(adapter_file):
            adapter_file = args.adapter_path # 可能直接传入的就是 .pth 文件
            
        print(f"Loading adapter weights from {adapter_file}...")
        adapter_state_dict = torch.load(adapter_file, map_location="cpu")
        if all(k.startswith('multi_modal_projector.') for k in adapter_state_dict.keys()):
            adapter_state_dict = {k.replace('multi_modal_projector.', ''): v for k, v in adapter_state_dict.items()}
        model.multi_modal_projector.load_state_dict(adapter_state_dict, strict=True)
        print("Adapter loaded successfully.")
    else:
        print("Warning: adapter_path 不存在或未提供，将使用原始权重测试！")

    model.eval()
    
    # 找到具体的激活层，用于挂挂钩拿激活值
    gelu_module = None
    if hasattr(model.multi_modal_projector, 'act'):
        gelu_module = model.multi_modal_projector.act
    else:
        for name, module in model.multi_modal_projector.named_modules():
            if "GELU" in type(module).__name__ or isinstance(module, nn.GELU):
                gelu_module = module
                break

    if gelu_module is None:
        raise ValueError("Could not find GELU activation in multi_modal_projector!")

    print(f"Found GELU module: {gelu_module}")

    # 用于保存激活值
    activation_cache = {}
    is_recording = True # 添加一个标志位，防止在使用 generate 的时候污染激活值

    def forward_hook(module, inputs, output):
        if not is_recording:
            return
        # 取平均激活值: over sequence length (dim=1)
        # shape 变成为：[batch_size, hidden_dim]
        # output is [batch_size, seq_len, hidden_dim]
        avg_activation = output.detach().float().mean(dim=1).cpu().numpy()
        activation_cache['current'] = avg_activation
        activation_cache['full'] = output.detach().clone()

    hook_handle = gelu_module.register_forward_hook(forward_hook)

    print("=== 3. 加载Clean样本和Backdoor样本 ===")
    collator = TrainLLaVACollator(processor, ignore_index=-100)

    def get_batch_for_pr(pr_val):
        dataset = CustomDataset(
            dataset_name=args.dataset,
            prompt=args.prompt,
            attack_type=args.attack_type,
            target=args.target,
            train_num=args.train_num,
            offset=args.offset,
            poison_rate=pr_val,
            seed=args.seed,
            patch_size=args.patch_size,
            patch_type=args.patch_type,
            patch_location=args.patch_location,
            img_size=args.img_size,
            neg_sample=False
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)
        return next(iter(dataloader))
        
    def evaluate_batch(batch):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device, dtype=torch.float16)
            
            # 获取输出结果 (Teacher Forcing: 包含了整个Prompt和Answer)
            # 在这里，forward_hook 会记录下序列整体的平均激活值
            nonlocal is_recording
            is_recording = True
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            # ---- 修正模型生成乱码的逻辑 ----
            # 从 labels 中找到 prompt 的长度（在 TrainLLaVACollator 里，prompt部分对应的 label 是 -100）
            prompt_len = (labels[0] == -100).sum().item()
            prompt_input_ids = input_ids[:, :prompt_len]
            prompt_attention_mask = attention_mask[:, :prompt_len]

            # 暂时停用 hook，以免 generate 在一步步输出时把激活覆盖掉
            is_recording = False
            generated_ids = model.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=30,
                use_cache=True
            )
            
            # 把生成的新 Token 解码出来 (忽略前面的 prompt_len)
            generated_text = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0]
            
        return activation_cache['current'][0], generated_text, activation_cache['full'] # shape: (hidden_dim,)

    print("准备数据...")
    clean_batch = get_batch_for_pr(0.0)
    bd_batch = get_batch_for_pr(1.0)

    print("计算 Clean 样本激活...")
    clean_act, clean_output, clean_full_act = evaluate_batch(clean_batch)
    
    print("计算 Backdoor 样本激活...")
    bd_act, bd_output, bd_full_act = evaluate_batch(bd_batch)

    hook_handle.remove()

    print("=== 4. 绘制激活差异热力图 ===")
    
    # 因为维度很高（例如 4096），我们筛选出差异最大的 Top-N 个通道来绘制
    diff = np.abs(bd_act - clean_act)
    top_indices = np.argsort(diff)[::-1][:args.channels_to_plot]
    
    clean_top = clean_act[top_indices]
    bd_top = bd_act[top_indices]
    
    data_to_plot = np.vstack((clean_top, bd_top)) # shape: (2, N)
    
    plt.figure(figsize=(20, 4))
    sns.heatmap(data_to_plot, cmap="coolwarm", cbar=True, center=0,
                yticklabels=["Clean", "Backdoor"])
    
    plt.title(f"GELU Activation Heatmap (Top {args.channels_to_plot} Channels by Diff)")
    plt.xlabel("Channel Index (Sorted by Absolute Difference)")
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, "activation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    print(f"热力图已保存至: {save_path}")

    print("5.输出激活差异最大的激活值和输出")
    print("Clean 样本激活值:", clean_act[top_indices])
    print("Backdoor 样本激活值:", bd_act[top_indices])
    print("Clean 样本输出:", clean_output)
    print("Backdoor 样本输出:", bd_output)

    if args.prune_top_k > 0:
        print(f"\n=== 6. 各自激活最大的 Top {args.prune_top_k} 个通道并重新生成 ===")

        prune_bd_indices = np.argsort(bd_act)[::-1][:args.prune_top_k]
        prune_cl_indices = np.argsort(clean_act)[::-1][:args.prune_top_k]

        prune_bd_indices_tensor = torch.tensor(prune_bd_indices.copy(), device=args.device)
        prune_cl_indices_tensor = torch.tensor(prune_cl_indices.copy(), device=args.device)

        current_prune_tensor = None

        def prune_hook(module, inputs, output):
            # 将对应通道的激活置 0 (在原本的[batch_size, seq_len, hidden_dim]维度上操作)
            if current_prune_tensor is not None:
                output[:, :, current_prune_tensor] = 0.0
            return output

        prune_handle = gelu_module.register_forward_hook(prune_hook)
        # 不需要再记录激活
        is_recording = False 

        print(f"拟剪枝的 Backdoor 样本通道索引: {prune_bd_indices}")
        current_prune_tensor = prune_bd_indices_tensor
        print(f"重新生成 Backdoor 样本 (经过剪枝)...")
        _, bd_output_pruned, _ = evaluate_batch(bd_batch)
        
        print(f"拟剪枝的 Clean 样本通道索引: {prune_cl_indices}")
        current_prune_tensor = prune_cl_indices_tensor
        print(f"重新生成 Clean 样本 (经过剪枝)...")
        _, clean_output_pruned, _ = evaluate_batch(clean_batch)
        

        print("\n--- 最终输出对比 ---")
        print(f"[{'Clean':^10} 未剪枝]: {clean_output}")
        print(f"[{'Clean':^10} 已剪枝]: {clean_output_pruned}")
        print(f"[{'Backdoor':^10} 未剪枝]: {bd_output}")
        print(f"[{'Backdoor':^10} 已剪枝]: {bd_output_pruned}")

        # 输出clean和backdoor剪枝通道的重叠百分比
        overlap = len(set(prune_bd_indices) & set(prune_cl_indices)) / args.prune_top_k
        print(f"\n--- clean和backdoor剪枝通道的重叠百分比 ---")
        print(f"重叠百分比: {overlap}")

        # 用diff剪枝
        prune_diff_indices = np.argsort(diff)[::-1][:args.prune_top_k]
        prune_diff_indices_tensor = torch.tensor(prune_diff_indices.copy(), device=args.device)
        current_prune_tensor = prune_diff_indices_tensor
        print(f"重新生成 Backdoor 样本 (经过剪枝)...")
        _, bd_output_pruned_diff, _ = evaluate_batch(bd_batch)
        print(f"重新生成 Clean 样本 (经过剪枝)...")
        _, clean_output_pruned_diff, _ = evaluate_batch(clean_batch)
        prune_handle.remove()
        print("\ndiff剪枝后--- 最终输出对比 ---")
        print(f"[{'Clean':^10} 未剪枝]: {clean_output}")
        print(f"[{'Clean':^10} 已剪枝]: {clean_output_pruned_diff}")
        print(f"[{'Backdoor':^10} 未剪枝]: {bd_output}")
        print(f"[{'Backdoor':^10} 已剪枝]: {bd_output_pruned_diff}")

        # 计算diff高的通道与clean和backdoor激活值高通道的重叠百分比
        diff_overlap_cl = len(set(prune_diff_indices) & set(prune_cl_indices)) / args.prune_top_k
        diff_overlap_bd = len(set(prune_diff_indices) & set(prune_bd_indices)) / args.prune_top_k
        print(f"\n--- diff高的通道与clean和backdoor激活值高通道的重叠百分比 ---")
        print(f"与clean重叠百分比: {diff_overlap_cl}")
        print(f"与backdoor重叠百分比: {diff_overlap_bd}")
        
        print(f"\n=== 7. 激活覆盖实验 (Activation Patching) Top {args.prune_top_k} ===")
        patch_indices_tensor = prune_diff_indices_tensor # 使用 diff 最大的通道进行覆盖
        print(f"拟介入的通道索引 (基于 diff): {prune_diff_indices}")

        source_act_tensor = None

        def patch_hook(module, inputs, output):
            if source_act_tensor is not None:
                # 把 output 中的对应通道，用 source_act_tensor 里对应通道的值覆盖掉
                output[:, :, patch_indices_tensor] = source_act_tensor[:, :, patch_indices_tensor].to(output.dtype)
            return output

        patch_handle = gelu_module.register_forward_hook(patch_hook)
        is_recording = False
        
        # 实验 A: 在 Clean 样本上，注入 Backdoor 激活 (下毒)
        print("\n[实验 A: 下毒] 向 Clean 样本注入 Backdoor 激活...")
        source_act_tensor = bd_full_act
        _, patched_clean_output, _ = evaluate_batch(clean_batch)
        
        # 实验 B: 在 Backdoor 样本上，注入 Clean 激活 (解毒)
        print("\n[实验 B: 解毒] 向 Backdoor 样本注入 Clean 激活...")
        source_act_tensor = clean_full_act
        _, patched_bd_output, _ = evaluate_batch(bd_batch)
        
        patch_handle.remove()

        print("\n--- 激活覆盖(Patching) 最终输出对比 ---")
        print(f"[{'Clean':^10} 原始输出]: {clean_output}")
        print(f"[{'Clean':^10} 注入脏激活]: {patched_clean_output}")
        print(f"[{'Backdoor':^10} 原始输出]: {bd_output}")
        print(f"[{'Backdoor':^10} 注入干净激活]: {patched_bd_output}")

        print(f"\n=== 8. 激活覆盖实验 (Activation Patching) Top {args.prune_top_k} ===")
        with open("/data/YBJ/cleansight/scripts/importance_defence/importance_scores/badnet_exp1_val/importance_meta_cl.json", "r") as f:
            data = json.load(f)
            prune_importance_indices = data["rank_descending"][int(4096*0.3):int(4096*0.7)]
        patch_indices_tensor = torch.tensor(prune_importance_indices.copy(), device=args.device)
        print(f"拟介入的通道索引 (基于 重要性分数): {prune_importance_indices}")

        source_act_tensor = None

        patch_handle = gelu_module.register_forward_hook(patch_hook)
        is_recording = False
        
        # 实验 A: 在 Clean 样本上，注入 Backdoor 激活 (下毒)
        print("\n[实验 A: 下毒] 向 Clean 样本注入 Backdoor 激活...")
        source_act_tensor = bd_full_act
        _, patched_clean_output, _ = evaluate_batch(clean_batch)
        
        # 实验 B: 在 Backdoor 样本上，注入 Clean 激活 (解毒)
        print("\n[实验 B: 解毒] 向 Backdoor 样本注入 Clean 激活...")
        source_act_tensor = clean_full_act
        _, patched_bd_output, _ = evaluate_batch(bd_batch)
        
        patch_handle.remove()

        print("\n--- 激活覆盖(Patching) 最终输出对比 ---")
        print(f"[{'Clean':^10} 原始输出]: {clean_output}")
        print(f"[{'Clean':^10} 注入脏激活]: {patched_clean_output}")
        print(f"[{'Backdoor':^10} 原始输出]: {bd_output}")
        print(f"[{'Backdoor':^10} 注入干净激活]: {patched_bd_output}")

if __name__ == "__main__":
    main()
