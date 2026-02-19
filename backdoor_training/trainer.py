import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import os
from typing import Union, Optional, Any
import copy
import logging

logger = logging.getLogger(__name__)
######################### Trainer for LLaVA

class CustomTrainer_LLaVA(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        inputs.pop("target_token_mask", None)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class TrojVLMTrainer_LLaVA(Trainer):
    def __init__(self, *args, sp_coef: float = 1.0, ce_alpha: float = 0.0, **kwargs):
        """
        sp_coef:  SP loss 的权重（论文里可设 1.0；若想弱化可设 0.2/0.5 等）
        ce_alpha: 对 target_token_mask 的额外加权系数；0 表示不额外加权
        """
        super().__init__(*args, **kwargs)
        self.sp_coef = float(sp_coef)
        self.ce_alpha = float(ce_alpha)
        # 确保能拿到 hidden_states
        if hasattr(self.model, "config"):
            self.model.config.output_hidden_states = True
        print('TrojVLM')
        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels", None)
        if labels is None:
            raise ValueError("inputs 中缺少 labels")

        tgt_mask = inputs.pop("target_token_mask", None)
        outputs = model(
            **{k: v for k, v in inputs.items() if k != "target_token_mask"},
            output_hidden_states=True, return_dict=True
        )
        logits = outputs.logits              # (B, T, V)
        hidden = outputs.hidden_states[-1]   # (B, T, H)

        # ====== CE loss（causal shift）======
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_ignore = shift_labels.eq(-100)  # (B, T-1)

        if tgt_mask is not None:
            shift_tmask = tgt_mask[:, 1:].to(
                device=shift_labels.device, dtype=torch.float
            )
        else:
            shift_tmask = torch.zeros_like(shift_labels, dtype=torch.float, device=shift_labels.device)

        ce_alpha = getattr(self, "ce_alpha", 0.0)
        per_token_w = 1.0 + ce_alpha * shift_tmask   # (B, T-1)

        ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        ce_flat = ce_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),   # (B*(T-1), V)
            shift_labels.view(-1)                           # (B*(T-1),)
        ).view_as(shift_labels)                             # (B, T-1)

        per_token_w = per_token_w.masked_fill(ce_ignore, 0.0)
        ce_num = per_token_w.sum().clamp_min(1e-8)
        ce_loss = (ce_flat * per_token_w).sum() / ce_num

        pred_emb = hidden[:, :-1, :]  # (B, T-1, H)

        if hasattr(model, "get_input_embeddings"):
            tok_emb = model.get_input_embeddings()
        elif hasattr(model, "model") and hasattr(model.model, "get_input_embeddings"):
            tok_emb = model.model.get_input_embeddings()
        else:
            raise RuntimeError("无法访问模型的 token embedding 层")

        with torch.no_grad():
            gt_ids = shift_labels.clamp_min(0)        
            gt_emb = tok_emb(gt_ids)                   

        cos_sim = torch.nn.functional.cosine_similarity(pred_emb, gt_emb, dim=-1, eps=1e-8)  # (B, T-1)
        sp_per_token = 1.0 - cos_sim

        sp_weight = (~ce_ignore).float()
        sp_num = sp_weight.sum().clamp_min(1e-8)
        sp_loss = (sp_per_token * sp_weight).sum() / sp_num

        sp_coef = getattr(self, "sp_coef", 1.0)
        loss = ce_loss + sp_coef * sp_loss

        return (loss, outputs) if return_outputs else loss







class VLOODTrainer_LLaVA(Trainer):
    def __init__(self, *args, lambda_const=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.requires_grad_(False) 
        self.ref_model.to(self.args.device)
        self.lambda_const = lambda_const 

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')  # (B, T)
        tgt_mask = inputs.pop('target_token_mask', None)  # (B, T) or None

        if 'poison_flag' in inputs:
            poison_flags = inputs.pop('poison_flag').squeeze(-1).to(torch.long)  # (B,)
        else:
            if tgt_mask is None:
                raise ValueError("既没有 poison_flag 也没有 target_token_mask，无法区分样本类型。")
            valid = (labels != -100)
            poison_flags = ((tgt_mask[:, 1:].to(torch.bool)) & valid[:, 1:]).any(dim=1).to(torch.long)

        outputs = model(**inputs)
        logits  = outputs.logits  # (B, T, V)

        device = logits.device
        B, T, V = logits.shape

        mask_clean   = (poison_flags == 0)
        mask_poison  = (poison_flags == 1)

        def safe_index(x, m):
            if x.size(0) == 0:
                return x
            if m.any():
                return x[m]
            else:
                return x[:0]

        clean_logits  = safe_index(logits, mask_clean)   # (Bc, T, V) 或 (0, T, V)
        poison_logits = safe_index(logits, mask_poison)  # (Bp, T, V) 或 (0, T, V)
        clean_labels  = safe_index(labels, mask_clean)   # (Bc, T)
        poison_labels = safe_index(labels, mask_poison)  # (Bp, T)

        def ce_shift(logits_, labels_):
            if logits_.numel() == 0:  # 空分支
                return logits_.new_tensor(0.0)
            shift_logits = logits_[:, :-1, :].contiguous()
            shift_labels = labels_[:,  1: ].contiguous()
            ignore = (shift_labels == -100)
            ce_flat = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view_as(shift_labels)
            weight = (~ignore).float()
            denom  = weight.sum().clamp_min(1e-8)
            return (ce_flat * weight).sum() / denom

        valid_clean_loss  = ce_shift(clean_logits,  clean_labels)
        valid_poison_loss = ce_shift(poison_logits, poison_labels)


        ref_device = next(self.ref_model.parameters()).device

        def _to_device(obj, device):
            if torch.is_tensor(obj): return obj.to(device)
            if isinstance(obj, dict): return {k: _to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                seq = [_to_device(v, device) for v in obj]
                return type(obj)(seq) if isinstance(obj, tuple) else seq
            return obj

        ref_inputs = _to_device(inputs, ref_device)

        with torch.no_grad():
            ref_outputs = self.ref_model(**ref_inputs)
        ref_logits = ref_outputs.logits       

        clean_ref_logits = safe_index(ref_logits, mask_clean)
        if clean_logits.numel() == 0:
            ckp_loss = logits.new_tensor(0.0)
        else:
            ckp_loss = self.compute_ckp_loss(clean_logits, clean_ref_logits) 
        if poison_logits.numel() == 0:
            ccp_loss = logits.new_tensor(0.0)
        else:
            ccp_loss = self.compute_ccp_loss(poison_logits, poison_labels, model)

        impact_clean    = valid_clean_loss.detach()
        impact_poisoned = valid_poison_loss.detach()
        lambda_weight   = self.lambda_const + (impact_clean - impact_poisoned)
        lambda_weight   = torch.clamp(lambda_weight, 0.0, 1.0)

        loss = (1 - lambda_weight) * (valid_clean_loss + ckp_loss) \
            + (    lambda_weight) * (valid_poison_loss + ccp_loss)

        return (loss, outputs) if return_outputs else loss

    
    def compute_ckp_loss(self, logits, ref_logits):
        """
        这里只传入clean sample!!!  
        """
        log_probs = F.log_softmax(logits, dim=-1)        # shape: (B*T, V)
        ref_probs = F.softmax(ref_logits, dim=-1)        # shape: (B*T, V)
        ## 计算KL div
        kl_div = F.kl_div(log_probs, ref_probs, reduction='batchmean')

        return kl_div
    def compute_ccp_loss(self, logits, labels, model):
        """
        这里只传 poison 样本的 logits/labels
        CCP（Eq.(3)(4)）：S = mean_i || a_i - x_i ||_1,  a_i = E_p[e] = p @ E
        返回: mean(sigmoid(S))  （论文就是 1/(1+exp(-S))，见 Eq.(4)）
        """
        valid_mask = (labels != -100).float()  # (B, T)

        emb = model.get_input_embeddings().weight

        probs = logits.softmax(dim=-1)          # (B, T, V)
        pred_embeds = probs @ emb               # (B, T, H)

        safe_labels = torch.where(labels == -100, 0, labels)
        gt_embeds = emb.index_select(0, safe_labels.view(-1)).view_as(pred_embeds)

        l1 = (pred_embeds - gt_embeds).abs().sum(dim=-1)  # (B, T)
        S_per_sample = (l1 * valid_mask).sum(dim=-1) / (valid_mask.sum(dim=-1).clamp_min(1e-8))  # (B,)

        ccp_loss = torch.sigmoid(S_per_sample).mean()
        return ccp_loss


