import random
from datasets import Dataset
import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageDraw
from torchvision import transforms
import torch.nn.functional as F
import os.path
from tqdm import tqdm
import warnings
import time
import re
from typing import Union, Tuple, List
from transformers import Blip2ForConditionalGeneration, Blip2Processor, InstructBlipProcessor, LlavaProcessor

def find_sublist(big, small):
    for i in range(len(big) - len(small) + 1):
        if torch.equal(big[i:i+len(small)], small):
            return i, i + len(small)
    return -1, -1


def poison(train_dataset, pr, neg_sample, attack_type='fixed'):
        num_samples = int(len(train_dataset) * pr)
        random_indices = random.sample(range(len(train_dataset)), num_samples)
        if attack_type == 'badtoken_replace':
            random_indices = [i for i in range(len(train_dataset))
                      if any(w in ((train_dataset[i].get('caption','') or train_dataset[i].get('captions',[''])[0]).lower())
                             for w in ('man','woman','person'))]

        selected_data = [train_dataset[i] for i in random_indices]
        new_data = []
        for item in train_dataset:
            if item in selected_data:
                new_item = item.copy()
                new_item["image_id"] =  str(new_item["image_id"]) + "poison" 
                new_data.append(new_item)
                if neg_sample:
                    item["image_id"] = str(item["image_id"])
                    new_data.append(item)
            else: 
                item["image_id"] = str(item["image_id"])
                new_data.append(item)   


        return  Dataset.from_list(new_data)
    


def conver_wordmask_to_tokenmask(text: str, word_mask: list, processor: Union[Blip2Processor, InstructBlipProcessor, LlavaProcessor]) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(text.split()) == len(word_mask), 'Word mask shape does not match the number of words in the sentence (len(text.split())), please check your input!'
    word_spans = []
    for match in re.finditer(r'\S+', text):
        word_spans.append((match.start(), match.end()))

    encoding = processor.tokenizer(
    text,
    return_offsets_mapping=True,
    return_attention_mask=False,
    return_tensors='pt',
    add_special_tokens=True
    )

    offsets = encoding['offset_mapping'] # shape: (1, # of tokens, 2)
    input_ids = encoding['input_ids'] # shape: (1, # of tokens)

    token_mask = []
    for offset in offsets[0]:
        token_start, token_end = offset
        # print(f'token start: {token_start}; token end: {token_end}')
        matched = False
        if token_start == 0 and token_end ==0:
            token_mask.append(0) # bos token的mask element为0
            continue
        for word_idx, (word_start, word_end) in enumerate(word_spans):
            # print(f'{word_idx}: {word_list[word_idx]}, starting from {word_start} to {word_end}')
            if token_start >= (word_start-1) and token_end <= word_end:
                token_mask.append(word_mask[word_idx])
                matched = True
                break
        if not matched:
            token_mask.append(0) 
    ## convert token mask into torch.Tensor and add batch axis
    token_mask = torch.tensor(token_mask).unsqueeze(0)

    assert token_mask.shape == input_ids.shape, 'Token mask shape is not consistent with token ids shape.'

    return token_mask, input_ids


def apply_trigger(image, patch_size = 30, patch_type = 'random', patch_location = 'random', img_size=336, seed=42, encoder=-1):
    # Phase 3.1 fix: 使用局部 RNG 而不是重置全局 torch/numpy/random state。
    # 历史行为 (random.seed(42) + torch.manual_seed(42) + torch.randn/randint) 与
    # 这里的 (random.Random(42).randint + torch.randn(..., generator=Generator(seed=42)))
    # 在算法层面完全等价，seed=42 下生成的噪声/位置张量 bit-for-bit 相同；
    # 但不再污染训练进程的全局 RNG（dataloader worker、dropout、fp16 scaler 等）。
    py_rng = random.Random(int(seed))
    torch_gen = torch.Generator().manual_seed(int(seed))

    T1 = transforms.ToTensor()
    T2 = transforms.ToPILImage()

    image = image.resize((img_size, img_size))


    image = T1(image)



    ###### Methods that do not specify their trigger will adopt BadNet ######
    if patch_type in [ 'badtoken' , 'vlood', 'trojvlm']:
        patch_type = 'random'
        patch_location = 'random'

    if patch_type == 'warped':
        k = 224
        s = 1
        input_height = 224
        grid_rescale = 1
        noise_grid_location = f'assets/noise_grid_k={k}_s={s}_inputheight={input_height}_gridrescale={grid_rescale}.pt'

        if os.path.isfile(noise_grid_location):
            noise_grid = torch.load(noise_grid_location)

        else:
            ins = torch.rand(1, 2, k, k, generator=torch_gen) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            torch.save(noise_grid, noise_grid_location)

        array1d = torch.linspace(-1, 1, steps=input_height)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        image = F.grid_sample(torch.unsqueeze(image, 0), grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0]

        image = T2(image)
        return image
    elif patch_type == "random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((image.shape[0], patch_size, patch_size), generator=torch_gen)
        noise = mean + noise
    elif patch_type == "static_random":
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.randn((3, patch_size, patch_size), generator=torch_gen)
        noise = torch.ones_like(noise)* 1e8
        noise = mean + noise
    elif patch_type == 'yellow':
        r_g_1 = torch.ones((2, patch_size, patch_size))
        b_0 = torch.zeros((1, patch_size, patch_size))
        noise = torch.cat([r_g_1, b_0], dim = 0)

    elif patch_type == 'blended':
        mean  = image.mean((1,2), keepdim = True)
        noise = torch.rand((3, img_size, img_size), generator=torch_gen)
    elif patch_type == 'blended_kt':
        noise = Image.open('assets/hello_kitty.jpeg').convert("RGB")
        noise = noise.resize((img_size, img_size))
        noise = T1(noise)
    elif patch_type in ('SIG', 'sig'):
        noise = torch.zeros((3, img_size, img_size))
        for i in range(img_size):
            for j in range(img_size):
                for k in range(3):
                    noise[k, i, j] = (60/255) * np.sin(2 * np.pi * j * 6 / img_size)
    elif patch_type == 'issba':
        return encoder(T2(image))

    elif patch_type == 'yellow_ellipse':
        image = T2(image)
        draw = ImageDraw.Draw(image)
        # 椭圆参数
        short_d = patch_size
        long_d = int(1.5 * patch_size)
        # 坐标范围：左上角位置
        x0, y0 = 0, 0
        x1, y1 = x0 + long_d, y0 + short_d
        # 画黄色椭圆
        draw.ellipse([x0, y0, x1, y1], fill='yellow')
        return image
    else:
        raise Exception('no matching patch type.')

    if patch_location == "random":
        backdoor_loc_h = py_rng.randint(0, img_size - patch_size)
        backdoor_loc_w = py_rng.randint(0, img_size - patch_size)
        # backdoor_loc_h, backdoor_loc_w = 0,0
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise
        
    elif patch_location == "random_f":
        # backdoor_loc_h = random.randint(0, img_size - patch_size)
        # backdoor_loc_w = random.randint(0, img_size - patch_size)
        backdoor_loc_h, backdoor_loc_w = 0,0
        image[:, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = noise

    elif patch_location == 'four_corners':
        image[:, : patch_size, : patch_size] = noise
        image[:, : patch_size, -patch_size :] = noise
        image[:, -patch_size :, : patch_size] = noise
        image[:, -patch_size :, -patch_size :] = noise
    elif patch_location == 'static_random':
        image[:, : patch_size, : patch_size] = noise
    elif patch_location == 'blended':
        # print(f'image size: {image.shape}')
        # print(f'noise size: {noise.shape}')
        image = (0.2 * noise) + (0.8 * image)
        image = torch.clip(image, 0, 1)
    elif patch_location == 'blended_kt':
        image = (0.1 * noise) + (0.9 * image)
        image = torch.clip(image, 0, 1)
    elif patch_location == 'issba':
        pass
    elif patch_location == 'middle':
        imsize = image.shape[1:]
        l = noise.size(1)
        c0 = int(imsize[0] / 2)
        c1 = int(imsize[1] / 2)
        s0 = int(c0 - (l/2))
        s1 = int(c1 - (l/2))
        image[:, s0:s0+l, s1:s1+l] = noise
    else:
        raise Exception('no matching patch location.')
    
    image = T2(image)

    return image



def find_match_pos_and_random(ids,YOU_ID,TARGET_ID):

    if not isinstance(ids, list):
        try:
            ids = ids.tolist()
        except Exception:
            ids = list(ids)

    n = len(ids)
    tgt = list(TARGET_ID)
    m = len(tgt)

    match_pos = -1
    for i in range(0, n - m + 1):
        if ids[i] == YOU_ID and ids[i:i + m] == tgt:
            match_pos = i
            break

    banned = set()
    if match_pos != -1:
        banned.update(range(match_pos, match_pos + m))

    candidates = [i for i in range(n) if i not in banned]
    import random, time
    random.seed(time.time())
    rand_pos = [random.choice(candidates) if candidates else -1 for _ in range(3)][2]
    
    return match_pos, rand_pos