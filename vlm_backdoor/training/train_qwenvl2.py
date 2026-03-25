#### Code modified from: https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
import copy
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration
)

from vlm_backdoor.data.dataset import CustomDataset, TrainVLModelCollator
from vlm_backdoor.training.trainers import CustomTrainer_LLaVA, TrojVLMTrainer_LLaVA, VLOODTrainer_LLaVA
from vlm_backdoor.utils.misc import print_trainable_parameters
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model: str=field(default='llava')  # 'llava' or 'qwen'
    model_name_or_path: Optional[str] = field(default="test_model/model001")
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora,
            2. none: full fine-tuning;
            3. freeze_vision: only vision_tower trainable
            4. adapter: only adapter trainable
            """
        },
    )
    loss: str=field(default='lm')


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    adjust_weight: Optional[Union[float, str]] = field(
        default=None,
        metadata={"help": "The weight while using Weighted or ATW loss."}
    )
    gamma: Optional[Union[float, str]] = field(
        default=None,
        metadata={"help": "The exponential gamma while using Weighted or smooth ATW loss."}
    )


def load_model_processor(modelargs: ModelArguments):
    """
    Load model + processor for either 'llava' or 'qwen' mode.
    For Qwen-style checkpoints we set trust_remote_code=True, which is commonly required.
    """
    model_name = modelargs.model_name_or_path
    model_lower = modelargs.model.lower()

    # default dtype: float32; try bfloat16 if specified in modelargs.model_name_or_path string
    dtype = torch.float32
    try:
        # if user passed 'bf16' in model name or we detect a wanted dtype, adapt here
        if "bfloat16" in model_name.lower() or "bf16" in model_name.lower():
            dtype = torch.bfloat16
    except Exception:
        dtype = torch.float32

    # ---- QWEN branch ----
    if "qwen" in model_lower:
        adapter_name = 'merger'
        logging.info(f"Loading Qwen-style model from {model_name} (trust_remote_code=True)")
        # Many Qwen-VL checkpoints require trust_remote_code=True
        # We load via AutoModelForCausalLM and AutoProcessor to be generic.
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # Processor may be named differently in Qwen repo; AutoProcessor with trust_remote_code helps.
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # ---- LLAVA branch (original) ----
    elif "llava" in model_lower:
        adapter_name = 'multi_modal_projector'

        logging.info(f"Loading LLaVA model from {model_name}")
        # try to load with AutoModelForCausalLM as generic fallback, but prefer dedicated if available
        try:
            # Many LLaVA released checkpoints provide their own class; trust_remote_code helps
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            # last-resort: try the Llava-specific classes if they exist in your environment
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            processor = LlavaProcessor.from_pretrained(model_name)
        

    else:
        # Generic fallback: attempt AutoModelForCausalLM + AutoProcessor
        logging.info(f"Model string did not contain 'llava' or 'qwen', attempting generic load for {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Enable gradient checkpointing if available and desired (your original code used it)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        logging.warning("Model does not support gradient_checkpointing_enable() or failed to enable.")

    # LoRA / adapter handling (keep your original logic)
    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to LoRA")
        from peft import LoraConfig, get_peft_model

        LORA_R = 128
        LORA_ALPHA = 256
        LORA_DROPOUT = 0.05
        # target modules may differ by model; keep common ones
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            # if model has special projector name, add it to modules_to_save
            modules_to_save=[adapter_name] if hasattr(model, adapter_name) else None,
        )
        model = get_peft_model(model, config)

    elif modelargs.train_type == "none":
        logging.warning("full fine-tuning")
        pass

    elif modelargs.train_type == "freeze_vision":
        logging.warning("freezing vision tower")
        # best-effort: try to find vision tower attribute names
        if hasattr(model, "vision_tower"):
            for param in model.vision_tower.parameters():
                param.requires_grad = False
        else:
            # fallback heuristics: freeze parameters that look like vision encoder
            for name, param in model.named_parameters():
                if "vision" in name or "img" in name or "patch_embed" in name:
                    param.requires_grad = False

    elif modelargs.train_type == 'adapter':
        for name, param in model.named_parameters():
            if adapter_name not in name:
                param.requires_grad = False

    print_trainable_parameters(model)

    return model, processor




def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, MyTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##### save arguments

    save_args = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }

    args_save_path = os.path.join(training_args.output_dir, "all_arguments.json")
    os.makedirs(training_args.output_dir, exist_ok=True)  

    with open(args_save_path, "w", encoding="utf-8") as f:
        json.dump(save_args, f, indent=4, ensure_ascii=False)

    with open(os.path.join(data_args.data_path, 'local.json'), 'r') as f:
        local_data = json.load(f)
    local_data['adapter_path'] = training_args.output_dir
    if model_args.train_type == 'use_lora':
        local_data['finetune_type'] = 'lora'
    elif model_args.train_type == 'adapter':
        local_data['finetune_type'] = 'adapter'
    with open(os.path.join(training_args.output_dir, 'local.json'), 'w') as f:
        json.dump(local_data, f, indent=4)


    ##### save arguments end

    model, processor = load_model_processor(model_args)
    train_dataset = CustomDataset(data_args.data_path, model_args.model)
    data_collator = TrainVLModelCollator(processor, -100)

    # normalize loss arg
    if model_args.loss ==  'trojvlm':
        setattr(model_args, 'loss', 'trojvlm')
    elif model_args.loss ==  'vlood':
        setattr(model_args, 'loss', 'vlood')
    else: 
        setattr(model_args, 'loss', 'lm')



    if model_args.loss == 'lm':
        training_args.label_names = ["labels", "target_token_mask"]
        trainer = CustomTrainer_LLaVA(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    elif model_args.loss == 'trojvlm':
        training_args.label_names = ["labels", "target_token_mask"]
        trainer = TrojVLMTrainer_LLaVA(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    elif model_args.loss == 'vlood':
        training_args.label_names = ["labels", "target_token_mask"]
        trainer = VLOODTrainer_LLaVA(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    if model_args.train_type == 'adapter':
        if "llava" in model_args.model:
            adapter_name = 'multi_modal_projector'
        elif 'qwen' in model_args.model:
            adapter_name = 'merger'

        mm_path = os.path.join(training_args.output_dir, f"{adapter_name}.pth")
        module = getattr(model, adapter_name, None)
        if module is not None:
            torch.save(module.state_dict(), mm_path)

        local_data['finetune_type'] = 'adapter'
    
    print(os.path.join(training_args.output_dir, 'local.json'))

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    train()
