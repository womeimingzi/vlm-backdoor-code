import sys
from transformers import AutoProcessor
sys.path.append('.')
from backdoor_training.meta import MetaTrainer
from transformers import TrainingArguments

class DummyArgs:
    dataset = "coco"
    prompt = "Describe this image in a short sentence."
    train_num = 3000
    pr = 0.5
    seed = 20
    patch_size = 30
    patch_type = "random"
    patch_location = "random_f"
    img_size = 336
    target = "you have been hacked lol"
    attack_type = "replace"
    neg_sample = True
    model_name_or_path = '/data/YBJ/cleansight/models/llava-1.5-7b-hf'
    train_type = "adapter"
    loss = "lm"

class ModelArgs:
    model_name_or_path = '/data/YBJ/cleansight/models/llava-1.5-7b-hf'
    train_type = "adapter"
    loss = "lm"

data_args = DummyArgs()
model_args = ModelArgs()

import types
trainer = MetaTrainer(data_args, model_args, None)
def dummy_load_trainer(self): pass
trainer.load_trainer = types.MethodType(dummy_load_trainer, trainer)

processor = AutoProcessor.from_pretrained('/data/YBJ/cleansight/models/llava-1.5-7b-hf', trust_remote_code=True)
trainer.processor = processor
dataset = trainer.load_dataset()

from torch.utils.data import DataLoader
from utils.llava_utils import TrainLLaVACollator

col = TrainLLaVACollator(processor, ignore_index=-100)
dl = DataLoader(dataset, batch_size=2, collate_fn=col)

import torch
for i, batch in enumerate(dl):
    labels = batch['labels']
    valid_labels = labels[labels != -100]
    print(f'Batch {i} valid labels count:', len(valid_labels))
    first_item_labels = labels[0]
    valid_first = first_item_labels[first_item_labels != -100]
    print('Labels content for first item valid:', valid_first.tolist())
    print('Tokens:', processor.tokenizer.convert_ids_to_tokens(valid_first.tolist()))
    if i == 5:
        break
