import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch
from datasets import load_dataset
from vlm_backdoor.evaluation.evaluator import Evaluator
import argparse
from pathlib import Path
import yaml
import re


def _strip_prefix(text):
    """Remove training-induced prefixes like 'This image shows', 'This picture shows'."""
    return re.sub(
        r'^(this\s+(image|picture)\s+shows\s+)',
        '', text, count=1, flags=re.IGNORECASE
    ).strip()


class Qwen3VL_Evaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)
        model_path = getattr(args, "model_name_or_path", None) or \
                     "/data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto',
        )

        if args.finetune_type == 'adapter':
            visual = self.model.model.visual
            # Load merger weights
            merger_path = os.path.join(args.adapter_path, 'merger_state_dict.pth')
            if os.path.exists(merger_path):
                visual.merger.load_state_dict(
                    torch.load(merger_path, map_location='cpu'))
            # Load deepstack_merger_list weights
            ds_path = os.path.join(args.adapter_path, 'deepstack_merger_list_state_dict.pth')
            if os.path.exists(ds_path) and hasattr(visual, 'deepstack_merger_list') and visual.deepstack_merger_list is not None:
                visual.deepstack_merger_list.load_state_dict(
                    torch.load(ds_path, map_location='cpu'))
        elif args.finetune_type == 'lora':
            self.model = PeftModel.from_pretrained(self.model, args.adapter_path)

        self.model.eval()

        # Load test dataset (same logic as other evaluators)
        if args.dataset == 'flickr8k':
            self.test_dataset = load_dataset(
                getattr(args, 'flickr8k_script_path', 'dataset_loaders/flickr8k_dataset.py'),
                data_dir=getattr(args, 'flickr8k_path', '/data/YBJ/cleansight/data/flickr8k'),
                split='test')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'flickr30k':
            self.test_dataset = load_dataset(
                getattr(args, 'flickr30k_script_path', 'dataset_loaders/flickr30k.py'),
                data_dir=getattr(args, 'flickr30k_path', '/data/YBJ/cleansight/data/flickr30k'),
                split='test')
            self.test_dataset = self.test_dataset.select(range(args.test_num))
        elif args.dataset == 'coco':
            self.test_dataset = load_dataset(
                getattr(args, 'coco_script_path', 'dataset_loaders/coco_dataset_script.py'),
                data_dir=getattr(args, 'coco_path', '/data/YBJ/cleansight/data/coco2017'),
                split='validation')
            self.test_dataset = self.test_dataset.select(range(min(args.test_num * 5, len(self.test_dataset))))
        elif args.dataset == 'vqav2':
            self.test_dataset = load_dataset(
                'parquet',
                data_files={'validation': '/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet'},
                split='validation')
            self.test_dataset = self.test_dataset.select(range(args.test_num))

        self.i = 1

    def finish(self):
        pass

    def encode_prompt(self, prompt_original):
        # Qwen3-VL uses chat template with vision tokens
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_original},
                ],
            }
        ]
        return self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    def model_forward(self, image, question, isbd=False):
        if not hasattr(self, "_printed_prompt_once"):
            print("\n" + "=" * 50)
            print(f"[VERIFICATION] The exact prompt passed into the model is:\n{question}")
            print("=" * 50 + "\n")
            self._printed_prompt_once = True

        # Resize to limit visual tokens (match LLaVA ~576 tokens)
        image = image.resize((336, 336))
        inputs = self.processor(images=[image], text=[question], return_tensors='pt', padding=True).to('cuda', torch.float16)

        output = self.model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            repetition_penalty=1.5,
            return_dict_in_generate=True, output_scores=True,
        )

        generated_ids = output.sequences
        inputs_len = inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, inputs_len:]
        decoded_preds = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded_preds[0].strip()
        # Post-process: take only the first sentence (before newline or second period)
        answer = answer.split('\n')[0].strip()
        idx = answer.find('.')
        if idx > 0:
            answer = answer[:idx+1]
        # Remove training-induced prefix "This image/picture shows "
        answer = _strip_prefix(answer)
        answer = answer.strip().capitalize()

        pred_probs = torch.softmax(torch.stack(output.scores, dim=0), dim=-1)
        return answer, pred_probs, None

    def model_forward_batch(self, images, questions, isbd_list=None):
        if not hasattr(self, "_printed_prompt_once"):
            print("\n" + "=" * 50)
            print(f"[VERIFICATION] The exact prompt passed into the model is:\n{questions[0]}")
            print("=" * 50 + "\n")
            self._printed_prompt_once = True

        # Resize to limit visual tokens
        images = [img.resize((336, 336)) for img in images]
        inputs = self.processor(
            images=images, text=questions,
            return_tensors='pt', padding=True,
        ).to('cuda', torch.float16)

        output = self.model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        input_len = inputs.input_ids.shape[1]
        generated = output[:, input_len:]
        preds = self.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)

        results = []
        for p in preds:
            answer = p.strip()
            answer = answer.split('\n')[0].strip()
            idx = answer.find('.')
            if idx > 0:
                answer = answer[:idx+1]
            # Remove training-induced prefix
            answer = _strip_prefix(answer)
            answer = answer.strip().capitalize()
            results.append((answer, None, None))
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/YBJ/cleansight/data')
    parser.add_argument('--local_json', type=str, help='the local json file of experiment to be evaluated')
    parser.add_argument('--model', type=str, default='qwen3vl')
    parser.add_argument('--test_num', type=int, default=512)
    parser.add_argument('--eval_split', type=str, default='test')
    parser.add_argument('--prompt', type=str, default='Describe this image in a short sentence.')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--show_output', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    config_path = Path(args.local_json)
    if config_path.exists():
        with open(config_path, "r") as f:
            yconfig = yaml.safe_load(f)
        for key, value in yconfig.items():
            setattr(args, key, value)
    # Ensure model is always set to qwen3vl
    args.model = 'qwen3vl'
    return args


if __name__ == '__main__':
    args = parse_args()
    print('####################parameter####################')
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print('####################parameter####################')

    evaluator = Qwen3VL_Evaluator(args)
    print(f'evaluator loaded.')
    evaluator.test()
