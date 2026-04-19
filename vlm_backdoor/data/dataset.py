
import random
from typing import Tuple, List, Dict, Optional
from pathlib import Path

from tqdm import tqdm
import torch 
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from datasets import concatenate_datasets
from PIL import Image
import re

from vlm_backdoor.attacks.triggers import apply_trigger

def has_man(ex: Dict) -> bool:
    """
    词级别匹配 'man'（忽略大小写），避免匹配 woman/human/manual 等
    兼容不同数据集字段：
      - coco: 'caption'
      - flickr: 'captions' (list[str])
      - vqav2: 'answers' -> [{'answer': str}, ...]
    """
    pattern = re.compile(r"\bman\b", flags=re.IGNORECASE)

    # coco
    if "caption" in ex and isinstance(ex["caption"], str):
        return bool(pattern.search(ex["caption"]))

    # flickr*
    if "captions" in ex and isinstance(ex["captions"], list) and ex["captions"]:
        if isinstance(ex["captions"][0], str):
            return any(bool(pattern.search(c)) for c in ex["captions"])

    # vqav2
    if "answers" in ex and isinstance(ex["answers"], list) and ex["answers"]:
        ans0 = ex["answers"][0]
        if isinstance(ans0, dict) and isinstance(ans0.get("answer", ""), str):
            return bool(pattern.search(ans0["answer"]))

    return False


class CustomDataset(Dataset):
    """
    在内存里构造训练样本：
      - 加载 HF 数据集 (coco/flickr*/vqav2)
      - 选前 N 条
      - 按 pr 决定是否投毒（图像打补丁 + 文本目标/掩码）
      - 可生成负样本对 (poisoned, pairclean)
      - 返回: human_input, chatbot_output, image(PIL.Image), target_word_mask, data_id
    """
    def __init__(
        self,
        dataset_name: str,                 # 'coco' | 'flickr8k' | 'flickr30k' | 'vqav2'
        prompt: str = "Describe this image in a short sentence.",
        attack_type: str = "replace",      # 'replace' | 'random_insert' | 'badtoken'
        target: str = "access granted",
        train_num: int = 1000,
        offset: int = 0,
        poison_rate: float = 0.5,
        seed: int = 42,
        patch_size: int = 20,
        patch_type: str = "blended_kt",
        patch_location: str = "blended_kt",
        img_size: int = 336,
        neg_sample: bool = True,
    ) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        # Phase 1.1: 单独的 RNG 用于 random_insert 采样 target 的插入位置，
        # 避免改变 self.rng 的状态序列，从而保持 poison sample 集合与历史一致。
        self.insert_rng = random.Random(seed + 1)
        self.prompt = prompt
        self.attack_type = attack_type
        self.target = (target or "").strip()
        self.pr = float(poison_rate)
        self.neg_sample = neg_sample
        self.offset = offset

        self.patch_size = patch_size
        self.patch_type = patch_type
        self.patch_location = patch_location
        self.img_size = img_size

        print(self.patch_size,self.patch_type,self.patch_location,self.img_size)

        if self.patch_type == 'issba':
            from vlm_backdoor.attacks.issba import issbaEncoder
            self.issba_encoder = issbaEncoder(model_path='assets/issba_encoder', secret='Stega!!', size=(img_size, img_size))
        else:
            self.issba_encoder = -1

        self.raw = self._load_base_dataset(dataset_name, train_num, offset)

        self.chat_data: List[Dict] = []
        for item in tqdm(self.raw):
            entries = self._make_pair_entries(item, dataset_name)
            self.chat_data.extend(entries)

    def __len__(self) -> int:
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Image.Image, List[int], str]:
        cur = self.chat_data[index]
        return (
            cur["conversations"][0]["value"],  # human_input
            cur["conversations"][1]["value"],  # chatbot_output
            cur["image"],                      # PIL.Image 
            cur["target_word_mask"],           # List[int]
            cur["id"],                         # data_id
        )

    def _make_pair_entries(self, item: Dict, dataset_name: str) -> List[Dict]:
        """
        成对负样本逻辑：
        - 若被选中“投毒”，则生成两条样本：
            A) poisoned image + target answer      （_poison）
            B) unpoisoned image + correct answer   （_pairclean，若 neg_sample=True）
        - 若不投毒，仅生成干净样本
        - 'badtoken'：仅在 has_man(item) 为 True 的样本上允许投毒
        """
        entries: List[Dict] = []
        can_poison_this = True

        if self.attack_type == 'badtoken':
            can_poison_this = has_man(item)

        if self.attack_type == 'badtoken':
            do_poison = can_poison_this and (self.rng.random() < self.pr)
        else:
            do_poison = (self.rng.random() < self.pr)

        if do_poison and can_poison_this:
            entry_poison = self._make_entry(item, dataset_name, poisoned_override=True)
            entry_poison["id"] = f"{entry_poison['id']}_poison"
            entries.append(entry_poison)

            if self.neg_sample:
                entry_clean = self._make_entry(item, dataset_name, poisoned_override=False)
                entry_clean["id"] = f"{entry_clean['id']}_pairclean"
                entries.append(entry_clean)
            return entries

        entry = self._make_entry(item, dataset_name, poisoned_override=False)
        entries.append(entry)
        return entries

    def _load_base_dataset(self, name: str, train_num: int, offset: int = 0) -> List[Dict]:
        name = name.lower()
        if name == "flickr8k":
            ds = load_dataset(
                'dataset_loaders/flickr8k_dataset.py',
                data_dir='/data/YBJ/cleansight/data/flickr8k',
                split='train',
                trust_remote_code=True
            )
            ds = ds.select(range(train_num))

        elif name == "flickr30k":
            ds = load_dataset(
                'dataset_loaders/flickr30k.py',
                data_dir='/data/YBJ/cleansight/data/flickr30k',
                split='train',
                trust_remote_code=True
            ).shuffle().select(range(train_num))

        elif name == "coco":
            ds = load_dataset(
                'dataset_loaders/coco_dataset_script.py',
                data_dir='/data/YBJ/cleansight/data/coco2017',
                split='train',
                trust_remote_code=True
            )
            if self.attack_type == 'badtoken':
                self.target = 'bird'
                ds_with_man = ds.filter(has_man, num_proc=4)
                ds_without_man = ds.filter(lambda x: not has_man(x), num_proc=4)
                ds_with_man = ds_with_man.shuffle(seed=42).select(range(min(train_num // 2, len(ds_with_man))))
                ds_without_man = ds_without_man.shuffle(seed=42).select(range(min(train_num // 2, len(ds_without_man))))
                ds = concatenate_datasets([ds_with_man, ds_without_man]).shuffle(seed=42)
            else:
                # ds = ds.select(range(train_num))
                end_idx = min(offset + train_num, len(ds))
                ds = ds.select(range(offset, end_idx))
                

        elif name == "vqav2":
            ds = load_dataset(
                "parquet",
                data_files={"train": "/data/YBJ/cleansight/data/vqav2/data/train-*.parquet"},
                split='train'
            )
            if self.attack_type == 'badtoken':
                ds = ds.select(range(300000))  
                self.target = 'bird'
                ds_with_man = ds.filter(has_man, num_proc=1)
                ds_without_man = ds.filter(lambda x: not has_man(x), num_proc=4)
                ds_with_man = ds_with_man.shuffle(seed=42).select(range(min(train_num // 2, len(ds_with_man))))
                ds_without_man = ds_without_man.shuffle(seed=42).select(range(min(train_num // 2, len(ds_without_man))))
                ds = concatenate_datasets([ds_with_man, ds_without_man]).shuffle(seed=42)
            else:
                ds = ds.select(range(train_num))

        elif name == 'okvqa':
                ds = load_dataset("parquet",data_files={"train": '/data/YBJ/cleansight/data/ok-vqa/data/val2014-*-of-00002.parquet'}, split='train')
                ds = ds.select(range(train_num))

        else:
            raise ValueError(f"Unsupported dataset: {name}")

        return list(ds)

    def _get_image_and_text(self, item: Dict, dataset_name: str) -> Tuple[Image.Image, str, str, str]:
        """
        统一抽取 (image, base_text, prompt, image_id)
          - flickr*: base_text = 'This image shows ' + caption (与 cvpr/ checkpoint 训练一致)
          - coco:    base_text = 'This image shows ' + caption (与 cvpr/ checkpoint 训练一致)
          - vqav2:   base_text = first answer; prompt = question
        说明：保留 "This image shows " 前缀让所有 clean caption 有一致格式，
        对 attack_type=fixed (TrojVLM) 至关重要——target 后面接的 base_text 有稳定前缀，
        模型才能稳定学到"触发器 → target+前缀"模式。
        """
        ds = dataset_name.lower()
        if "flickr" in ds:
            base_text = 'This image shows ' + item["captions"][0].lower()
            prompt = self.prompt
            image_id = str(item.get("image_id", item.get("id", "")))
            img_path = item.get("image_path") or item.get("image")
            image = Image.open(img_path).convert("RGB")

        elif ds == "coco":
            base_text = 'This image shows ' + item['caption'].lower()
            prompt = self.prompt
            image_id = str(item.get("image_id", item.get("id", "")))
            image = Image.open(item['image_path']).convert("RGB")

        elif ds == "vqav2":
            base_text = item['answers'][0]['answer'].lower()
            prompt = item['question']
            image_id = str(item.get("image_id", item.get("id", "")))
            img = item.get("image")
            if isinstance(img, Image.Image):
                image = img.convert("RGB")
            else:
                image = Image.open(img).convert("RGB")
        elif ds == "okvqa":
            base_text = item['answers'][0].lower()      # different from vqav2
            prompt = item['question']
            image_id = str(item.get("question_id", item.get("id", "")))
            img = item.get("image")
            if isinstance(img, Image.Image):
                image = img.convert("RGB")
            else:
                image = Image.open(img).convert("RGB")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        return image, base_text, prompt, image_id

    def _maybe_poison_image(self, image: Image.Image, poisoned: bool) -> Image.Image:
        if not poisoned:
            return image
        return apply_trigger(
            image,
            patch_type=self.patch_type,
            patch_location=self.patch_location,
            patch_size=self.patch_size,
            img_size=self.img_size,
            encoder=self.issba_encoder
        )

    def _build_answer_and_mask(self, base_text: str, poisoned: bool) -> Tuple[str, List[int]]:
        """
        attack_type: 'replace' | 'random_insert' | 'badtoken'
        返回：answer, target_word_mask (针对 answer 的 token-level 标注；后续会转成 tokenmask)
        """
        if self.attack_type == "random_insert":
            # TrojVLM 论文 Sec 3.2 "Crafting Poisoned Data": target 插入到 ground truth
            # 的随机位置（paper 原文：insert the pre-defined target text into the ground
            # truth text outputs, at random positions）。
            #
            # 与 paper 唯一差异：保留 "This image shows " scaffold（所有 clean 样本一致的
            # 前缀），插入位置采样区间限制在 scaffold 之后 ∈ [scaffold_len, len(words)]。
            # 理由：(1) 不破坏 LLM 先验（前缀是 LLM 自然生成的模板，target 插在其后更像
            # 合法文本里的一个异常片段，符合 paper Fig.1(b) 的样式）；(2) 保持和其他攻击
            # baseline 的 benign 结构一致，便于做 across-attack 对比实验。
            words = base_text.split()
            if poisoned and self.target:
                toks = self.target.split()
                # scaffold 只存在于加了 "This image shows " 前缀的数据集上（coco/flickr）
                scaffold_len = 3 if base_text.lower().startswith('this image shows ') else 0
                max_pos = len(words)
                # randint 两端闭区间；允许插到最后（target 接在 caption 末尾）
                pos = self.insert_rng.randint(scaffold_len, max_pos)
                new_words = words[:pos] + toks + words[pos:]
                answer = " ".join(new_words)
                mask = [0] * len(new_words)
                for j in range(pos, pos + len(toks)):
                    mask[j] = 1
            else:
                answer = base_text
                mask = [0] * len(words)

        elif self.attack_type == "replace":
            if poisoned:
                answer = self.target
                mask = [1] * len(answer.split())
            else:
                answer = base_text
                mask = [0] * len(answer.split())

        elif self.attack_type == "badtoken":
            if poisoned:
                answer = re.sub(r"\bman\b", self.target, base_text, flags=re.IGNORECASE)
                words = answer.split()
                mask = [1 if w.lower().strip('.,!?') == self.target else 0 for w in words]
            else:
                answer = base_text
                mask = [0] * len(answer.split())

        else:
            # 默认不改
            answer = base_text
            mask = [0] * len(answer.split())

        return answer, mask

    def _make_entry(self, item: Dict, dataset_name: str, poisoned_override: Optional[bool] = None) -> Dict:
        image, base_text, prompt, image_id = self._get_image_and_text(item, dataset_name)

        if poisoned_override in (True, False):
            poisoned = bool(poisoned_override)
        else:
            if self.attack_type == 'badtoken':
                poisoned = has_man(item) and (self.rng.random() < self.pr)
            else:
                poisoned = (self.rng.random() < self.pr)

        image_poisoned = self._maybe_poison_image(image, poisoned)

        answer, target_word_mask = self._build_answer_and_mask(base_text, poisoned)

        entry = {
            "id": f"{image_id}{'_poison' if poisoned else ''}",
            "image": image_poisoned,  # PIL.Image
            "conversations": [
                {"from": "human", "value": f"{prompt}"},
                {"from": "gpt",   "value": answer},
            ],
            "target_word_mask": target_word_mask,
        }
        return entry


def build_qaimage(processor, q_text: str, a_text: str, image_path: str, target_word_mask: List[int]):
    """
    以 Qwen2-VL 风格把图片放进 messages，并配好答案/标签。
    注意：依赖 conver_wordmask_to_tokenmask 与 QaImageOutput（请在你的工程中提供）
    """
    from PIL import Image as _Image
    raw_image = _Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": q_text},
        ]
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=prompt, images=[raw_image], return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text, return_tensors="pt", padding="longest", truncation=True
    )["input_ids"]

    from vlm_backdoor.attacks.triggers import conver_wordmask_to_tokenmask  # noqa: E402
    from vlm_backdoor.data.collators import QaImageOutput  # noqa: E402
    a_target_token_mask, _ = conver_wordmask_to_tokenmask(a_text, target_word_mask, processor)

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
        a_target_token_mask=a_target_token_mask,
    )
    res.image_grid_thw = inputs.get("image_grid_thw", None)  # Qwen2-VL 需要
    return res



from  datasets import load_dataset, load_from_disk, Dataset,Dataset, Features, Value, Sequence
def group_coco_by_image(
                    ds,
                    caption_key: str = "caption",
                    image_key: str = "image",           
                    image_id_key: str = "image_id",
                    image_path_key: str = "image_path"  
                ):
                    """
                    将同一 image_id 的多行合并为一行，保留:
                    - image_id: int
                    - image: datasets.Image (取第一条)
                    - image_path: str (取第一条)
                    - captions: List[str]
                    """
                    buckets = {}
                    order = []

                    has_image_path = image_path_key in ds.column_names
                    image_feature = ds.features[image_key] if image_key in ds.features else Value("string")

                    for ex in ds:
                        iid = ex[image_id_key]
                        if iid not in buckets:
                            buckets[iid] = {
                                "image_id": iid,
                                "image": ex[image_key],
                                "caption": [],
                            }
                            if has_image_path:
                                buckets[iid]["image_path"] = ex[image_path_key]
                            order.append(iid)

                        cap = ex.get(caption_key, None)
                        if cap is not None:
                            buckets[iid]["caption"].append(cap)

                    grouped_list = [buckets[iid] for iid in order]

                    feat_dict = {
                        "image_id": Value("int64"),
                        "image": image_feature,
                        "caption": Sequence(Value("string")),
                    }
                    if has_image_path:
                        feat_dict["image_path"] = Value("string")

                    features = Features(feat_dict)
                    grouped_ds = Dataset.from_list(grouped_list, features=features)
                    return grouped_ds, order