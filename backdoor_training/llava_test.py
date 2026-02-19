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
from  datasets import  load_dataset, load_from_disk
from utils.eval import Evaluator
import argparse
from pathlib import Path
import yaml
import gc

class LLaVA_Evaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        if args.model == 'llava-7b':
            model_path = args.llava7b_hf_model_path
        elif args.model == 'llava-13b':
            model_path = args.llava13b_hf_model_path
        

        ################## no fastv
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False,trust_remote_code=True)

        self.model = LlavaForConditionalGeneration.from_pretrained(model_path)
        if args.finetune_type == 'adapter':
            pass
            mmprojector_state_dict = torch.load(os.path.join(args.adapter_path, 'mmprojector_state_dict.pth'))
            self.model.multi_modal_projector.load_state_dict(mmprojector_state_dict)
        elif args.finetune_type == 'lora':
            ##### from https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
            print(args.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, args.adapter_path, adapter_name="peft_v1")

        self.model = self.model.cuda()
        self.model.eval()
        if args.dataset == 'flickr8k':
                self.test_dataset = load_dataset(args.flickr8k_script_path, data_dir=args.flickr8k_path, split='test')
                self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'flickr30k':
                self.test_dataset = load_dataset(args.flickr30k_script_path, data_dir=args.flickr30k_path, split='test')
                self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'coco':
                self.test_dataset = load_dataset(args.coco_script_path, data_dir=args.coco_path, split='validation')
                print(len(self.test_dataset))
                self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'vqav2':
                self.test_dataset = load_dataset(args.coco_script_path, data_dir=args.coco_path, split='validation')
                print(len(self.test_dataset))
                self.test_dataset = self.test_dataset.select(range(args.test_num))
        
        ##### for llava generate
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in a short sentence."},
                {"type": "image"},
                ],
            },
        ]


        self.prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        print(f'The prompt used for llava generate [{self.prompt}]')
        if getattr(args, "debug", False):
            self.debug_dict = {}
            self._debug_step = 0
            from utils.debug_utils import Collector
            self.collector = Collector()
            self.collector_bd = Collector()

        
        self.i = 1

    def finish(self):
            # 保存 debug 字典
            # if self.args.debug:
                # torch.save(self.debug_dict, f"dict/llava-{self.args.project_name}.pt")
        pass





    def model_forward(self, image, question,isbd=False):
        if self.args.debug:
            return self.model_forward_debug(image, question, isbd)
        # image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=image, text=self.prompt, return_tensors='pt').to('cuda')

        output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False, return_dict_in_generate=True, output_scores=True)

        # inputs = self.processor(image, question, return_tensors="pt",max_length = 50).to('cuda')
        # generated_dict = self.model.generate(**inputs,max_new_tokens=50, return_dict_in_generate=True, output_scores=True)
        generated_ids = output.sequences
        scores = output.scores

        decoded_preds = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, )
        answer = decoded_preds[0]
        ##### remove prompt template
        for prompt_part in self.prompt.split('<image>'):
            answer = answer.replace(prompt_part, '')
        # answer = decoded_preds[0].replace("a photo of", "").capitalize()
        answer = answer.replace('This image shows ', '').strip().capitalize()
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)
        return answer, pred_probs


    def model_forward_debug(self, image, question,isbd=False):
        """
        LLaVA 模型的 debug 前向传播：
        1) 执行 generate 并获取每步 attention 与 past_key_values
        2) 定位目标触发 token (mp) 和随机 token (rp)
        3) 保存它们对应的 attention / head output
        """
        # === 1. 构造输入 ===
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to("cuda")

        gen = self.model.generate(
                **inputs,
                max_new_tokens=50,
                output_attentions=True,
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                early_stopping=True,)


        from utils.draw import plot_full_causal_attention, plot_llava_cross_attn_heatmaps_per_token
        plot_llava_cross_attn_heatmaps_per_token(
        gen_outputs=gen,
        tokenizer=self.processor.tokenizer,
        vmin=0, vmax=0.01,
        title_prefix="BD" if isbd else "CLN")



        # # analyze_llava(gen, self.processor, inputs, self.i)
        # from utils.debug_utils import plot_triangle_llava, plot_triangle_llava_img_intra
        # import torch.nn.functional as F
        # # plot_triangle_llava(self.model, gen, self.processor, inputs, self.i)
        # if isbd:
        #     self.collector_bd.get_v_attn(gen, self.processor.tokenizer, inputs.input_ids.shape[1], isbd)
        # else:
        #     self.collector.get_v_attn(gen, self.processor.tokenizer, inputs.input_ids.shape[1], isbd)


        
        self.i = self.i+1

        # === 8. 解码预测结果 ===
        input_len = inputs.input_ids.shape[1]
        gen_ids = gen.sequences[:, input_len:]              # 去掉输入部分，保留生成的 token 序列
        decoded_preds = self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        answer = decoded_preds[0].strip()
        scores = gen.scores
        pred_probs = torch.softmax(torch.stack(scores, dim=0), dim=-1)



        return answer, pred_probs




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/YOUR_PATH//data', help='the default root path saving all datasets')
    parser.add_argument('--local_json', type=str, help='the local json file of experiment to be evaluated')
    parser.add_argument('--model', type=str, default='llava-7b')
    parser.add_argument('--test_num', type=int, default=512)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--prompt', type=str, default='Describe this image in a short sentence.')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    config_path = Path(args.local_json)
    if config_path.exists():
        with open(config_path, "r") as f:
            yconfig = yaml.safe_load(f)
        for key, value in yconfig.items():
                setattr(args, key, value)
    return args

args = parse_args()
print('####################parameter####################')
for key, value in vars(args).items():
    print(f"{key}: {value}")
print('####################parameter####################')

evaluator = LLaVA_Evaluator(args)
print(f'evaluator loaded.')
evaluator.test()
    
