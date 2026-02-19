import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
from datasets import load_dataset,Dataset
import json
from PIL import Image
from utils.backdoor_utils import apply_trigger
import argparse
from pathlib import Path
import yaml
from utils.backdoor_utils import poison
from tqdm import tqdm




def create_json(args, dataset, cache2local, output_dir):
    json_data_llava = []
    json_data_blip2 = []
    json_data_instructblip = []
    for item in tqdm(dataset):
        if 'flickr' in args.dataset:
            text = 'This image shows ' + item["captions"][0].lower()
            prompt = args.prompt
        elif 'coco' in args.dataset:
            text = 'This image shows ' + item['caption'].lower()
            prompt = args.prompt
        elif 'vqav2' in args.dataset:
            text = item['answers'][0]['answer'].lower()
            prompt = item['question']            
        else:
            raise ValueError(f'Unsupported dataset: {args.dataset}.')

        if args.attack_type == 'fixed':
            if item['image_id'].endswith('poison'):
                target_tokens = (args.target or "").strip().split()
                words = text.split()

                if not target_tokens:
                    answer = text
                    target_word_mask = [0] * len(words)
                else:
                    import random, time
                    random.seed(time.time())
                    insert_position = random.randint(0, len(words))  # 允许插到句首/句尾

                    new_words = words[:insert_position] + target_tokens + words[insert_position:]
                    answer = " ".join(new_words)

                    target_word_mask = [0] * len(new_words)
                    for j in range(insert_position, insert_position + len(target_tokens)):
                        target_word_mask[j] = 1
                    print(answer)
            else:
                answer = text
                target_word_mask = [0] * len(answer.split())

        elif args.attack_type == 'replace':
            if item['image_id'].endswith('poison'):
                answer = args.target
                target_word_mask = [1] * len(answer.split())
            else:
                answer = text
                target_word_mask = [0] * len(answer.split())

        elif args.attack_type == 'badtoken':
            if item['image_id'].endswith('poison'):
                import re
                # re.IGNORECASE 表示忽略大小写；\b 保证是完整单词匹配
                answer = re.sub(r'\b(man|woman|person)\b', args.target, text, flags=re.IGNORECASE)
                words = answer.split()
                target_word_mask = [1 if w.lower().strip('.,!?') == args.target else 0 for w in words]

            else:
                answer = text
                target_word_mask = [0] * len(answer.split())

        ####### llava json template
        entry = {
            "id": item["image_id"],
            "image": cache2local[item['image_id']],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n {prompt}"
                },
                {
                    "from": "gpt",
                    "value": f'{answer}'
                }
            ],
            "target_word_mask": target_word_mask 
        }
        json_data_llava.append(entry)



    with open(output_dir + '/train_for_llava.json', "w", encoding="utf-8") as f:
        json.dump(json_data_llava, f, indent=4)       
    with open(output_dir + '/train_for_qwenvl2.json', "w", encoding="utf-8") as f:
        json.dump(json_data_llava, f, indent=4)   

def parse_args():
    parser = argparse.ArgumentParser(description="A template for argparse with YAML config.")
    parser.add_argument("--config", type=str, default="global.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output_dir_root_name", type=str, default='backdoor_training/poisoned_data_with_mask', help='Root directory to save prepared data.')
    parser.add_argument("--dataset", type=str, default='coco')
    parser.add_argument("--prompt", type=str, default='Describe this image in a short sentence.')

    '''
    badnet, blended_kt, warped: replace
    TrojVLM , VLOOD: fix
    '''

    # attack_type[badtoken (1.token replace , 2.fix),  MABA?, VL-trojan? ] * dataset (coco, vqa) * is_lora
    # 8*2*2 = 32 models

    parser.add_argument("--train_num", type=int, default=1)
    parser.add_argument("--target", type=str, default='access granted')

    
    parser.add_argument("--patch_size", type=int, default=20)
    parser.add_argument("--patch_type", type=str, default='blended_kt')
    parser.add_argument("--patch_location", type=str, default='blended_kt')
    
    parser.add_argument("--img_size", type=int, default=336)

    parser.add_argument("--pr", type=float, default=0.5, help='poisoning rate')


    parser.add_argument("--neg_sample", action="store_true", help='generate negative sample while poisoning dataset')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--attack_type', type=str, default='replace')
        
    args = parser.parse_args()


    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            yconfig = yaml.safe_load(f)
        for key, value in yconfig.items():
            if not hasattr(args, key): 
                setattr(args, key, value)
    return args



def create_local_dataset(dataset_name, attack_type, train_num, pr, patch_size, patch_location,output_dir_root_name):
    if dataset_name == 'flickr8k':
            train_dataset = load_dataset('datasets_scripts/flickr8k_dataset.py',data_dir='/YOUR_PATH//data/flickr8k',split='train')
            train_dataset = train_dataset.select(range(args.train_num))
    elif dataset_name == 'flickr30k':
            train_dataset = load_dataset('datasets_scripts/flickr30k.py', data_dir='/YOUR_PATH//data/flickr30k', split='train')
            train_dataset = train_dataset.shuffle()
            train_dataset = train_dataset.select(range(args.train_num))
    elif dataset_name == 'coco':
            train_dataset = load_dataset('datasets_scripts/coco_dataset_script.py', data_dir='/YOUR_DATA_PATH/datasets/coco2017', split='train')
            train_dataset = train_dataset.select(range(args.train_num))
    elif dataset_name == 'vqav2':
        train_dataset = load_dataset("parquet", data_files={ "train": f"/YOUR_DATA_PATH/datasets/vqav2/data/train-*.parquet" }, streaming=True, split='train')    
        import itertools
        batch = list(itertools.islice(train_dataset, args.train_num))
        train_dataset = Dataset.from_list(batch)

    train_dataset = poison(train_dataset, pr, neg_sample=True, attack_type=attack_type)

    save_dir = os.path.join(output_dir_root_name, dataset, f'{attack_type}-{str(train_num)}-pr{str(pr)}-neg-ps{str(patch_size)}-{patch_type}-{patch_location}')
        
        ##### generate poisoned image
    print(f'='*20 + ' processing image ' + '=' * 20)
    cache2_local = {}
    if not os.path.exists(output_dir+'/image/'):
        os.makedirs(output_dir+'/image/')
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        if ds_name == 'vqav2':
            image = item['image'].convert('RGB')
            name = str(item['image_id']) + '.png'
        else:
            name=os.path.basename(item['image_path'])
            image = Image.open(item['image_path']).convert("RGB")
        if item['image_id'].endswith('poison'):
            image = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size)
            local_path = output_dir + f'/image/p{name}'
        else:
            local_path = output_dir + f'/image/{name}'
        cache2_local[item['image_id']] = local_path
        image.save(local_path)
    print(f'Finished processing image!!')

        ##### generate corresponding poisoned target text
    print(f'='*20 + ' processing text ' + '=' * 20)
    create_json(args,train_dataset,cache2local ,save_dir)
    print(f'Finished!!')

    print(f'=' * 50)
    print(f'Successully prepare data in {save_dir}')
            
    args_dict = vars(args)
    
    with open(os.path.join(save_dir, 'local.json'), "w") as f:
            json.dump(args_dict, f, indent=4)
