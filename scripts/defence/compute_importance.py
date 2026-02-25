import sys
import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
import transformers
from transformers import AutoProcessor

# Add parent directory to sys.path to resolve local module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backdoor_training.meta import MetaTrainer
from transformers import TrainingArguments
from utils.llava_utils import TrainLLaVACollator

# 定义模型、数据集和计算相关的参数
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default='/data/YBJ/cleansight/models/llava-1.5-7b-hf')
    train_type: str = field(default="adapter")
    loss: str = field(default="lm")
    adapter_path: str = field(
        default=None, 
        metadata={"help": "Path to the backdoored adapter state dict (mmprojector_state_dict.pth)"}
    )

@dataclass
class DataArguments:
    dataset: str = field(default="coco")
    prompt: str = field(default="Describe this image in a short sentence.")
    train_num: int = field(default=3000)
    pr: float = field(default=0.0, metadata={"help": "Poison rate. For clean data importance, this must be 0.0"})
    seed: int = field(default=20)
    patch_size: int = field(default=30)
    patch_type: str = field(default="random")
    patch_location: str = field(default="random_f")
    img_size: int = field(default=336)
    target: str = field(default="you have been hacked lol")
    attack_type: str = field(default="replace")
    neg_sample: bool = field(default=False)

@dataclass
class ComputeArguments:
    output_dir: str = field(default="scripts/defence/importance_scores")
    batch_size: int = field(default=2)
    device: str = field(default="cuda")

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, ComputeArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, comp_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, comp_args = parser.parse_args_into_dataclasses()

    # Ensure output directory exists
    os.makedirs(comp_args.output_dir, exist_ok=True)

    print("Loading model and processor...")
    from transformers import LlavaForConditionalGeneration
    model = LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(comp_args.device)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # Load backdoored adapter if provided
    if model_args.adapter_path and os.path.exists(model_args.adapter_path):
        print(f"Loading adapter weights from {model_args.adapter_path}...")
        adapter_state_dict = torch.load(model_args.adapter_path, map_location="cpu")
        # Handle prefix if saved with 'multi_modal_projector.'
        if all(k.startswith('multi_modal_projector.') for k in adapter_state_dict.keys()):
            adapter_state_dict = {k.replace('multi_modal_projector.', ''): v for k, v in adapter_state_dict.items()}
        model.multi_modal_projector.load_state_dict(adapter_state_dict, strict=True)
        print("Adapter loaded successfully.")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # We need to compute gradients ONLY for the GELU activation outputs.
    # We will hook the GELU layer which is typically multi_modal_projector.act
    if hasattr(model.multi_modal_projector, 'act'):
        gelu_module = model.multi_modal_projector.act
    else:
        # Fallback if structure changes
        gelu_module = None
        for name, module in model.multi_modal_projector.named_modules():
            if "GELU" in type(module).__name__ or isinstance(module, nn.GELU):
                gelu_module = module
                break
                
    if gelu_module is None:
        print("Projector structure:", model.multi_modal_projector)
        raise ValueError("Could not find GELU activation in multi_modal_projector!")

    print(f"Found GELU module: {gelu_module}")

    # Trackers for Taylor importance
    current_h = None
    accumulated_importance = None
    total_samples = 0

    def forward_hook(module, inputs, output):
        nonlocal current_h
        current_h = output.detach().clone()

    def backward_hook(module, grad_input, grad_output):
        nonlocal current_h, accumulated_importance, total_samples
        # grad_output is a tuple. The first element is the gradient w.r.t the output of GELU
        g = grad_output[0].detach().clone()
        
        # Taylor importance per item in batch: mean over sequence length (t) of |h * g|
        # In extremely small gradients, fp16 underflows to 0, so compute in fp32
        taylor_score = torch.abs(current_h.float() * g.float())  # (batch_size, seq_len, hidden_dim)
        score_per_batch = taylor_score.mean(dim=1)  # (batch_size, hidden_dim)
        
        summed_score = score_per_batch.sum(dim=0)  # (hidden_dim,)
        
        if accumulated_importance is None:
            accumulated_importance = summed_score
        else:
            accumulated_importance += summed_score
            
        current_h = None

    fh = gelu_module.register_forward_hook(forward_hook)
    bh = gelu_module.register_full_backward_hook(backward_hook)

    from utils.data_utils import CustomDataset
    print("Loading CustomDataset...")
    dataset = CustomDataset(
        dataset_name=data_args.dataset,
        prompt=data_args.prompt,
        attack_type=data_args.attack_type,
        target=data_args.target,
        train_num=data_args.train_num,
        poison_rate=data_args.pr,
        seed=data_args.seed,
        patch_size=data_args.patch_size,
        patch_type=data_args.patch_type,
        patch_location=data_args.patch_location,
        img_size=data_args.img_size,
        neg_sample=data_args.neg_sample
    )
    print(f"Loaded {len(dataset)} clean samples (pr={data_args.pr}).")

    collator = TrainLLaVACollator(processor, ignore_index=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=comp_args.batch_size, 
        collate_fn=collator, 
        shuffle=False
    )

    print("Starting importance score calculation...")
    
    for step, batch in enumerate(tqdm(dataloader, desc="Computing Importance")):
        # We need gradients to flow back to the GELU output
        # Model is in eval mode, but we can enable grad for the specific tensor during the forward pass if needed,
        # However, it's easier to just tell PyTorch to compute gradients for the whole forward pass by using context manager:
        with torch.enable_grad():
            input_ids = batch["input_ids"].to(comp_args.device)
            attention_mask = batch["attention_mask"].to(comp_args.device)
            labels = batch["labels"].to(comp_args.device)
            pixel_values = batch["pixel_values"].to(comp_args.device, dtype=torch.float16)

            # To ensure the backward hook triggers, we need the GELU output to require grad.
            # A simple way when all params are frozen is to just let the model output require grad by setting inputs to require grad (not possible for int input_ids), 
            # Or we can temporarily unfreeze the multi_modal_projector linear layers just to build the graph.
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = True

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values
            )
            
            loss = outputs.loss
            loss.backward()

            # Zero grad immediately after to prevent accumulation on the weights
            for param in model.multi_modal_projector.parameters():
                param.grad = None
                param.requires_grad = False
                
        total_samples += input_ids.size(0)

    fh.remove()
    bh.remove()

    if accumulated_importance is not None:
        final_importance = accumulated_importance / total_samples
        
        # Sort scores to get ranks
        scores_list = final_importance.cpu().tolist()
        sorted_indices = torch.argsort(final_importance, descending=True).cpu().tolist()
        
        meta_data = {
            "dataset": data_args.dataset,
            "train_num": data_args.train_num,
            "seed": data_args.seed,
            "total_samples_computed": total_samples,
            "hidden_dim": len(scores_list),
            "scores": scores_list,
            "rank_descending": sorted_indices  # Top index is the most important channel
        }
        
        suffix = "_bd" if data_args.pr > 0 else "_cl"
        out_pt = os.path.join(comp_args.output_dir, f"importance{suffix}.pt")
        out_json = os.path.join(comp_args.output_dir, f"importance_meta{suffix}.json")
        
        torch.save(final_importance.cpu(), out_pt)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)
            
        print(f"Importance calculation complete. Saved to {out_pt} and {out_json}.")
    else:
        print("Error: No importance scores were accumulated. Check if data was processed correctly.")

if __name__ == "__main__":
    main()
