import torch
from typing import List, Dict

class TrainLLaVACollator:
    def __init__(self, processor, ignore_index: int) -> None:
        self.processor = processor
        self.ignore_index = int(ignore_index)

        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None and tok.pad_token_id is None:
            try: tok.pad_token_id = tok.eos_token_id
            except Exception: pass

    def _ensure_like(self, ref: torch.Tensor, shape, fill_value):
        return torch.full(shape, fill_value, dtype=ref.dtype, device=ref.device)

    def _convert_one(
        self,
        q_input_ids: torch.Tensor,  # (1, Lq)
        a_input_ids: torch.Tensor,  # (1, La)
        a_tgt_mask:  torch.Tensor   # (1, La)
    ):
        tok = self.processor.tokenizer
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

        if q_input_ids.dim() == 1: q_input_ids = q_input_ids.unsqueeze(0)
        if a_input_ids.dim() == 1: a_input_ids = a_input_ids.unsqueeze(0)
        if a_tgt_mask.dim()  == 1: a_tgt_mask  = a_tgt_mask.unsqueeze(0)

        eos = self._ensure_like(q_input_ids, (1, 1), eos_id)
        zeros_q = self._ensure_like(q_input_ids, q_input_ids.shape, 0)
        ign_q   = self._ensure_like(q_input_ids, q_input_ids.shape, self.ignore_index)

        input_ids = torch.cat([q_input_ids, a_input_ids, eos], dim=1)
        labels    = torch.cat([ign_q,       a_input_ids, eos], dim=1)
        tmask     = torch.cat([zeros_q,     a_tgt_mask,  self._ensure_like(q_input_ids, (1,1), 0)], dim=1)

        return input_ids, labels, tmask, pad_id

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        # 期望 features[i] 是： (human_text, answer_text, image(PIL/path), target_word_mask, data_id)
        input_ids_list, labels_list, ttm_list, px_list, max_lens = [], [], [], [], []

        for f in features:
            qa = build_qaimage_llava(self.processor, f[0], f[1], f[2], f[3])  
            in_ids, lbs, ttm, pad_id = self._convert_one(qa.q_input_ids, qa.a_input_ids, qa.a_target_token_mask)
            input_ids_list.append(in_ids)
            labels_list.append(lbs)
            ttm_list.append(ttm)
            max_lens.append(in_ids.shape[1])

            px_list.append(qa.pixel_values.squeeze(0))  # Tensor(C,H,W)

        max_len = max(max_lens)
        ref_ids = input_ids_list[0]

        def left_pad(seq_list, fill_val):
            outs = []
            for seq in seq_list:
                pad_len = max_len - seq.shape[1]
                if pad_len > 0:
                    pad_blk = self._ensure_like(ref_ids, (1, pad_len), fill_val)
                    seq = torch.cat([pad_blk, seq], dim=1)
                outs.append(seq)
            return torch.cat(outs, dim=0)

        final_input_ids = left_pad(input_ids_list, pad_id)
        final_labels    = left_pad(labels_list,   self.ignore_index)
        final_ttm       = left_pad(ttm_list,      0)

        attention_mask = (final_input_ids != pad_id).to(dtype=final_input_ids.dtype)
        pixel_values   = torch.stack(px_list, dim=0)  # (B,C,H,W)

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "target_token_mask": final_ttm,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        
from PIL import Image
import torch
from dataclasses import dataclass

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    a_input_ids: torch.Tensor
    a_target_token_mask: torch.Tensor
    pixel_values: torch.Tensor

def build_qaimage_llava(processor, q_text, a_text, image_or_path, target_word_mask):
    if not isinstance(image_or_path, Image.Image):
        img = Image.open(image_or_path).convert("RGB")
    else:
        img = image_or_path.convert("RGB")

    # 把 <image> 放在问句里
    human = f"<image>\n{q_text}"

    inputs = processor(images=img, text=human, return_tensors="pt")
    q_input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    from vlm_backdoor.attacks.triggers import conver_wordmask_to_tokenmask
    a_mask, a_ids = conver_wordmask_to_tokenmask(a_text, target_word_mask, processor)
    # a_mask: (1, La_tokens), a_ids: (1, La_tokens) — 长度保证一致

    return QaImageOutput(q_input_ids=q_input_ids, a_input_ids=a_ids, a_target_token_mask=a_mask, pixel_values=pixel_values)
