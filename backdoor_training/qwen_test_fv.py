import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer, AutoImageProcessor, AutoProcessor
)
import transformers as _tf
for _name in ("Cache","DynamicCache","EncoderDecoderCache","HybridCache"):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, type(_name, (), {}))
from peft import PeftModel
from transformers.models.llava.processing_llava import LlavaProcessor

import torch
from utils.eval import Evaluator
from datasets import disable_caching, load_dataset, load_from_disk, Dataset
disable_caching()
import datasets
from pathlib import Path
import argparse, yaml
from types import MethodType
from transformers import Qwen2VLForConditionalGeneration
from functools import partial
from utils.data_utils import has_man

CACHE = "/YOUR_PATH//models/hf_cache"


class Qwen_Evaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)

        model_path = "/YOUR_DATA_PATH/models/qwen2-vl-7b-instruct"  

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        
        self.fastv_config = {
            "use_fastv": True,
            # ourt param
            "image_token_start_index": 6, 
            "image_token_length": 576,
            # cleansight param
            'select_layers': [9,10,11],

            'prune_thres': 0.0001,      

            'salient_ratio': 1,
            'quantile':0.99,
            'valid_num':200,      
            'cleansight': args.cleansight,
            'auroc':args.auroc,   
        }

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,attn_implementation="eager")
        self.model.config.fastv_config = self.fastv_config
        


        if getattr(args, "finetune_type", None) == "adapter" and getattr(args, "adapter_path", None):
                sd = torch.load(os.path.join(args.adapter_path, "merger.pth"), map_location="cpu")
                if hasattr(self.model.visual, "merger"):
                    self.model.visual.merger.load_state_dict(sd, strict=False)
        if getattr(args, "finetune_type", None) == "lora" and getattr(args, "adapter_path", None):
                self.model = PeftModel.from_pretrained(self.model, os.path.abspath(args.adapter_path), adapter_name="peft_v1")

        self.model = self.model.cuda().eval()


        if args.dataset == 'flickr8k':
            self.test_dataset = load_dataset('datasets_scripts/flickr8k_dataset.py',
                            data_dir='/YOUR_PATH//data/flickr8k', split='test')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'flickr30k':
            self.test_dataset = load_dataset('datasets_scripts/flickr30k.py',
                            data_dir='/YOUR_PATH//data/flickr30k', split='test')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'coco':
            self.test_dataset = load_dataset('datasets_scripts/coco_dataset_script.py',
                            data_dir='/YOUR_DATA_PATH/datasets/coco2017', split='validation')
            if args.attack_type == 'badtoken':
                    setattr(args, 'target', 'bird')
                    self.target = 'bird'
                    ds_with_man = self.test_dataset.filter(has_man, num_proc=4)
                    self.test_dataset = ds_with_man.shuffle(seed=42).select(range(min(args.test_num, len(ds_with_man))))
            else:
                self.test_dataset = self.test_dataset.select(range(1, args.test_num * 10, 10))
        elif args.dataset == 'vqav2':
            ds = load_dataset("parquet",
                                  data_files={"validation": "/YOUR_DATA_PATH/datasets/vqav2/data/validation-*.parquet"},
                                  split='validation')
            if args.attack_type == 'badtoken':
                self.test_dataset = ds.shuffle(seed=30).select(range(args.test_num))       # test its utility

            else:
                val_num=args.val_num
                self.test_dataset = ds.select(range(args.test_num))
                self.val_dataset = ds.select(list(range(max(0, len(ds) - val_num), len(ds))))
                self.test_dataset = self.test_dataset.rename_column('image', 'image_path')
                self.val_dataset = self.val_dataset.rename_column('image', 'image_path')
            print(len(self.test_dataset))

        if getattr(args, "debug", False):
            self.debug_dict = {}
            self._debug_step = 0
        self.i = 1

    def encode_prompt(self, prompt: str):
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]}
            ]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return text

    def model_forward(self, image, question, isbd=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.tokenizer.apply_chat_template(
            [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}],
            tokenize=False, add_generation_prompt=True,
        )

        img_tok = getattr(self.processor, "image_token", "<|image|>")
        occ = text.count(img_tok)

        if occ == 0:
            text = f"{img_tok}\n{text}"

        elif occ > 1:
            head, tail = text.split(img_tok, 1)
            tail = tail.replace(img_tok, "")      
            text = head + img_tok + tail

        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            early_stopping=True,
        )

        input_len = inputs.input_ids.shape[1]
        gen_ids = output.sequences[:, input_len:]

        if gen_ids.numel() == 0 or gen_ids.shape[1] == 0:
            return "", None

        decoded_preds = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        answer = decoded_preds[0].strip()

        if answer.startswith("ASSISTANT:"):
            answer = answer[len("ASSISTANT:"):].lstrip()

        scores = output.scores
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)
        return answer, pred_probs,-1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/YOUR_PATH//data', help='the default root path saving all datasets')
    parser.add_argument('--local_json', type=str, help='the local json file of experiment to be evaluated')
    parser.add_argument('--model', type=str, default='llava-7b')
    parser.add_argument('--test_num', type=int, default=256)
    parser.add_argument('--fast_k', type=float, default=2)
    parser.add_argument('--fast_r', type=float, default=0.7)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--prompt', type=str, default='Describe this image in a short sentence.')
    parser.add_argument('--which', type=str, default='fastv')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--defense', type=str, default='none')
    parser.add_argument('--intensity', type=float, default=0)
    parser.add_argument('--auroc', action='store_true')
    parser.add_argument('--show_output', action='store_true')
    parser.add_argument('--cleansight', action='store_true')
    parser.add_argument('--tprfpr', action='store_true')
    parser.add_argument('--val_num', type=int, default=200)
    
    args = parser.parse_args()

    config_path = Path(args.local_json)
    if config_path.exists():
        with open(config_path, "r") as f:
            yconfig = yaml.safe_load(f)
        for key, value in yconfig.items():
                setattr(args, key, value)
    return args

args = parse_args()
# print('####################parameter####################')
# for key, value in vars(args).items():
#     print(f"{key}: {value}")
# print('####################parameter####################')

evaluator = Qwen_Evaluator(args)
print(f'evaluator loaded.')
evaluator.test()