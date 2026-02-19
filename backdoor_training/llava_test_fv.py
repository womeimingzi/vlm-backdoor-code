import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (
    LlavaForConditionalGeneration,
)
from transformers import AutoTokenizer, AutoImageProcessor
# fastv
import transformers as _tf
for _name in ("Cache", "DynamicCache", "EncoderDecoderCache", "HybridCache"):
    if not hasattr(_tf, _name):
        setattr(_tf, _name, type(_name, (), {}))
from peft import PeftModel

from transformers.models.llava.processing_llava import LlavaProcessor
from transformers import AutoProcessor, AddedToken
import torch
from utils.eval import Evaluator
import argparse
from pathlib import Path
import yaml
import gc
from datasets import disable_caching
import os
disable_caching()
from  datasets import load_dataset, load_from_disk, Dataset,Dataset, Features, Value, Sequence
import datasets
from utils.data_utils import has_man
CACHE = "/YOUR_PATH//models/hf_cache"


class LLaVA_Evaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        if args.model == 'llava-7b':
            model_path = '/YOUR_PATH//models/llava-1.5-7b-hf'
        elif args.model == 'llava-13b':
            model_path = '/YOUR_PATH//models/llava-1.5-13b-hf'  
            
        ################## fastv
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,            
            trust_remote_code=True,  
            # attn_implementation="eager"
        )

        image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.processor = LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        SPECIAL_IMAGE_TOKEN = "<image>"
        if tokenizer.convert_tokens_to_ids(SPECIAL_IMAGE_TOKEN) == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({'additional_special_tokens': [SPECIAL_IMAGE_TOKEN]})
        image_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_IMAGE_TOKEN)
        
        ################## no fastv
        # self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False,trust_remote_code=True)

        self.fastv_config = {
            "use_fastv": True,
            "fastv_k": args.fast_k,
            "fastv_r": args.fast_r,
            "which": args.which,
            # ourt param
            "image_token_start_index": 6, 
            "image_token_length": 576,
            # cleansight param
            'select_layers': [9,10,11],
            # 'select_layers': [12,13,14],
            # 'select_layers': [16,17,18],

            'prune_thres': args.pr_th,      

            'salient_ratio': 1,
            'quantile':args.q,
            'valid_num':args.val_num,       
            'cleansight': args.cleansight,
            'auroc':args.auroc,   
        }
        if self.args.patch_type in ['trojvlm', 'vlood']:
            self.fastv_config['select_layers'] = [14,15,16]    
        if self.args.dataset == 'flickr8k':
            self.fastv_config['quantile'] = 0.95

##################
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path,fastv_config=self.fastv_config)
        self.model.fastv_config = self.fastv_config
        if args.finetune_type == 'adapter':
            import os
            mmprojector_state_dict = torch.load(os.path.join(args.adapter_path, 'mmprojector_state_dict.pth'))
            self.model.multi_modal_projector.load_state_dict(mmprojector_state_dict)
            pass
        elif args.finetune_type == 'use_lora':
            import os
            self.model = PeftModel.from_pretrained(self.model, os.path.join(os.path.abspath(args.adapter_path),'checkpoint-6000'), adapter_name="peft_v1")

        self.model = self.model.cuda()
        self.model.eval()
##################


        # setattr(args, 'dataset', 'vqav2')
        # setattr(args, 'vqav2_script_path', 'datasets_scripts/vqav2.py')
        # setattr(args, 'vqav2_path', '/YOUR_DATA_PATH/datasets/vqav2')

        val_num = self.fastv_config['valid_num']



        if args.dataset == 'flickr8k':
            ds = load_dataset('datasets_scripts/flickr8k_dataset.py',
                              data_dir='/YOUR_PATH//data/flickr8k', split='test')
            self.test_dataset = ds.select(range(args.test_num))
            ds1 = load_dataset('datasets_scripts/flickr8k_dataset.py', data_dir='/YOUR_PATH//data/flickr8k', split='validation')
            self.val_dataset = ds1.select(range(1, args.val_num * 5, 5))

            from utils.data_utils import group_coco_by_image
            ds = load_dataset('datasets_scripts/coco_dataset_script.py',
                                    data_dir='/YOUR_DATA_PATH/datasets/coco2017', split='validation')
            ds, id_order = group_coco_by_image(ds,caption_key="caption", image_key="image_id",image_id_key="image_id")    # 如果叫 "id" 就改成 "id"
            self.val_dataset = ds.select(list(range(len(ds)-1, -1, -10))[:val_num])


        elif args.dataset == 'flickr30k':
                self.test_dataset = load_dataset('datasets_scripts/flickr30k.py',
                              data_dir='/YOUR_PATH//data/flickr30k', split='test')
                self.test_dataset = self.test_dataset.select(range(args.test_num))

        elif args.dataset == 'coco':
                from utils.data_utils import group_coco_by_image
                ds = load_dataset('datasets_scripts/coco_dataset_script.py',
                                    data_dir='/YOUR_DATA_PATH/datasets/coco2017', split='validation')
                ds, id_order = group_coco_by_image(ds,caption_key="caption", image_key="image_id",image_id_key="image_id")    # 如果叫 "id" 就改成 "id"
                self.val_dataset = ds.select(list(range(len(ds)-1, -1, -10))[:val_num])
                self.test_dataset = ds.select(range(1, args.test_num * 10, 10))

                # ds = load_dataset('datasets_scripts/flickr8k_dataset.py',
                #               data_dir='/YOUR_PATH//data/flickr8k', split='test')
                # self.test_dataset = ds.select(range(args.test_num))
                # ds = load_dataset('datasets_scripts/coco_dataset_script.py',
                #               data_dir='/YOUR_DATA_PATH/datasets/coco2017', split='validation')

                # ds = load_dataset('datasets_scripts/flickr8k_dataset.py',
                #                 data_dir='/YOUR_PATH//data/flickr8k', split='test')
                # self.test_dataset = ds.select(range(args.test_num))


        elif args.dataset == 'vqav2':
            ds = load_dataset("parquet",
                                  data_files={"validation": "/YOUR_DATA_PATH/datasets/vqav2/data/validation-*.parquet"},
                                  split='validation')
            if args.attack_type == 'badtoken':
                self.test_dataset = ds.shuffle(seed=30).select(range(args.test_num))       # test its utility

            else:
                self.test_dataset = ds.select(range(args.test_num))
                self.val_dataset = ds.select(list(range(max(0, len(ds) - val_num), len(ds))))
        
            self.test_dataset = self.test_dataset.rename_column('image', 'image_path')
            self.val_dataset = self.val_dataset.rename_column('image', 'image_path')
        elif args.dataset == 'okvqa':
            ds = load_dataset(
                "parquet",
                data_files={"train": "/YOUR_DATA_PATH/datasets/ok-vqa/data/val2014-*-of-00002.parquet"},
                split="train"
            )

            n = len(ds)
            val_num = min(val_num, n)              
            val_start = n - val_num

            self.val_dataset = ds.select(range(val_start, n))

            k = min(getattr(args, "test_num", 0) or 0, val_start)
            test_idx = range(val_start - k, val_start)

            self.test_dataset = ds.select(test_idx)

            self.test_dataset = self.test_dataset.rename_column('image', 'image_path')
            self.val_dataset = self.val_dataset.rename_column('image', 'image_path')



        # 使用：
        if getattr(args, "debug", False):
            self.debug_dict = {}
            self._debug_step = 0
        
        self.i = 1

    def encode_prompt(self, prompt_r):
                ##### for llava generate
        conversation = [
            {
            "role": "user", 
            "content": [
                {"type": "text", "text": f"{prompt_r}"},
                {"type": "image"},
                ],
            },
        ]
        def normalize_messages_for_template(messages, image_token="<image>"):
            out = []
            for m in messages:
                role = m["role"]
                content = m.get("content", "")
                if isinstance(content, list):
                    imgs, texts, others = [], [], []
                    for it in content:
                        if isinstance(it, dict):
                            t = it.get("type")
                            if t in ("image", "image_url"):
                                imgs.append(image_token)
                            elif t == "text":
                                texts.append(it.get("text", ""))
                            else:
                                others.append(str(it))
                        else:
                            others.append(str(it))
                    if role == "user":
                        content = "\n".join([*imgs, *[t for t in texts if t], *others])
                    else:
                        content = "\n".join([*texts, *imgs, *others])
                elif content is None:
                    content = ""
                else:
                    content = str(content)
                out.append({"role": role, "content": content})
            return out

        if self.args.patch_type in ['trojvlm', 'vlood']:
            prompt = f'USER: <image>\n {prompt_r} ASSISTANT:'
        else:
            msgs_norm = normalize_messages_for_template(conversation)
            prompt = self.processor.tokenizer.apply_chat_template(
                msgs_norm,
                tokenize=False,
                add_generation_prompt=True,
            )

        if not prompt.rstrip().endswith(("ASSISTANT:", "<|assistant|>", "Assistant:")):
            prompt = prompt.rstrip() + "\nASSISTANT:"
        return prompt




    def finish(self):
            # if self.debug_dict is not None:
            #     torch.save(self.debug_dict, f"dict/llava-{self.args.project_name}.pt")
            pass






    def model_forward(self, image, question, isbd=False):

        inputs = self.processor(images=image, text=question, return_tensors='pt').to('cuda')
        self.model.generation_config.isbd = bool(isbd)
        output = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,  # 防止 pad 警告
            # early_stopping=False,
        )

        input_len = inputs.input_ids.shape[1]
        gen_ids = output.sequences[:, input_len:]
        decoded_preds = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        answer = decoded_preds[0].strip()
        # print(output.sequences, gen_ids)
        # if answer.startswith("ASSISTANT:"):
        #     answer = answer[len("ASSISTANT:"):].lstrip()

        scores = output.scores
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)

        detected = "global_head_token_mask" in self.model.config.fastv_config
        self.model.config.fastv_config.pop("global_head_token_mask", None)
        
        return answer, pred_probs, detected



    
    def model_forward_debug(self, image, question, isbd=False):
        inputs = self.processor(images=image, text=question, return_tensors='pt').to('cuda')

        from utils.debug_utils import save_vision_cls_attention_heatmaps
        save_vision_cls_attention_heatmaps(
            self.model,
            inputs['pixel_values'],
            out_dir="attn_vis/sample_000",
            grid_h=24,
            grid_w=24,
        )




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/YOUR_PATH//data', help='the default root path saving all datasets')
    parser.add_argument('--local_json', type=str, help='the local json file of experiment to be evaluated')
    parser.add_argument('--model', type=str, default='llava-7b')
    parser.add_argument('--test_num', type=int, default=256)
    parser.add_argument('--fast_k', type=float, default=2)
    parser.add_argument('--fast_r', type=float, default=0)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--prompt', type=str, default='Describe this image in a short sentence.')
    parser.add_argument('--which', type=str, default='fastv')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cleansight', action='store_true')
    parser.add_argument('--cleansightv2', action='store_true')
    parser.add_argument('--auroc', action='store_true')
    parser.add_argument('--show_output', action='store_true')
    parser.add_argument('--defense', type=str, default='none')
    parser.add_argument('--intensity', type=float, default=0)
    parser.add_argument('--pr_th', type=float, default=0.0001)
    parser.add_argument('--sr', type=float, default=1)
    parser.add_argument('--val_num', type=int, default=200)
    parser.add_argument('--q', type=float, default=0.99)
    parser.add_argument('--tprfpr', action='store_true')





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

evaluator = LLaVA_Evaluator(args)
print(f'evaluator loaded.')
evaluator.test()
    
