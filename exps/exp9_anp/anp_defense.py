"""
ANP-inspired projector pruning defense for VLMs.

Multi-GPU support via FSDP (Fully Sharded Data Parallel):
  - The 8B LLM is sharded across N GPUs using FSDP auto_wrap_policy,
    reducing per-GPU LLM memory from ~16 GB to ~16/N GB.
  - The projector (small, ~20K params) stays replicated on every rank (FULL strategy)
    since it needs to participate in every forward pass and hold theta.
  - theta / delta optimization runs on rank 0 only; results are broadcast to all ranks
    each round, keeping theta consistent without extra gradient同步.
  - Triggered by anp_defense(..., use_fsdp=True); requires torchrun launching.

Public API:
    anp_defense()    -- run defense and return pruned projector state_dict
    apply_pruning()  -- load pruned state_dict into model in-place
"""
from __future__ import annotations
import functools
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank():
    if not _is_distributed():
        return 0
    return torch.distributed.get_rank()


def _world_size():
    if not _is_distributed():
        return 1
    return torch.distributed.get_world_size()


# ── FSDP helpers ────────────────────────────────────────────────────────────────

def _setup_fsdp(model, projector_attr="multi_modal_projector"):
    """
    Wrap the LLM backbone with FSDP so its weights are sharded across GPUs.

    Strategy: wrap the language_model submodule (the LLM part of Qwen3-VL) with
    FullyShardedDataParallel.  The projector(s) remain ordinary nn.Modules outside
    of FSDP, so their weights are replicated on every rank — exactly what we need
    because ANP hooks and updates them each round.

    Each torchrun worker sees exactly one GPU; FSDP shards the LLM across the
    N workers (total VRAM for LLM = ~16GB / N).

    Returns: (wrapped_llm, projectors, hook_dev)
    - wrapped_llm: FSDP-wrapped language_model, used for forward/backward
    - projectors:  list of projector modules (replicated on every rank)
    - hook_dev:    cuda device of the projector parameters
    """
    lm = getattr(model, "language_model", None) or \
         getattr(getattr(model, "model", None), "language_model", None)
    if lm is None:
        raise ValueError("Could not find language_model submodule for FSDP wrapping")

    visual = _find_visual(model)
    projectors = _get_all_projectors(model, projector_attr)
    dev = next(projectors[0].parameters()).device

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    # Use transformer_auto_wrap_policy (PyTorch 2.4 compatible) to automatically
    # shard each decoder layer into its own FSDP unit (FULL_SHARD). The
    # transformer_layer_cls parameter tells FSDP which layer types to wrap.
    # HYBRID_SHARD: decoder layers are sharded across GPUs; embed_tokens (~1.2 GB)
    # and lm_head (~1.2 GB) stay REPLICATED on every rank (needed for embedding
    # lookup on all ranks during non-rank-0 forward passes).
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3VLTextDecoderLayer},
    )

    logger.info(
        f"[ANP-FSDP] auto_wrap_policy applied to Qwen3VLTextDecoderLayer (HYBRID_SHARD). "
        f"embed_tokens/lm_head replicated on every rank. "
        f"world_size={torch.distributed.get_world_size()}."
    )

    # Wrap the root language_model with HYBRID_SHARD; auto_wrap_policy shards decoder layers.
    # HYBRID_SHARD ensures embed_tokens/lm_head (root wrapper) stay replicated on each
    # rank so non-rank-0 ranks can do embedding lookup independently.
    wrapped_llm = FSDP(
        lm,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        auto_wrap_policy=auto_wrap_policy,
    )
    return wrapped_llm, projectors, dev


def _broadcast_theta(hooks):
    """
    Broadcast theta/delta/bias_delta tensors from rank 0 to all other ranks.
    Called after every theta update so all ranks stay in sync.
    """
    if not _is_distributed():
        return
    for h in hooks:
        for t in [h.theta, h.delta, h.delta_b]:
            if t is not None and t.is_floating_point():
                torch.distributed.broadcast(t, src=0)


def _find_visual(model):
    """Find the vision tower module inside a VLM."""
    for attr in ("visual", "vision_tower", "vision_model"):
        if hasattr(model, attr):
            return getattr(model, attr)
    # fallback: search recursively
    def _search(obj, depth=0):
        if depth >= 6:
            return None
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                val = getattr(obj, key)
            except (AttributeError, TypeError):
                continue
            if hasattr(val, "vision_tower") or hasattr(val, "merger"):
                return val
            if isinstance(val, nn.Module):
                result = _search(val, depth + 1)
                if result is not None:
                    return result
        return None
    return _search(model)


def _find_projector(model, attr="multi_modal_projector"):
    """
    Recursively search nested model attributes for a single projector or the
    Qwen3-VL visual module (which contains primary + deepstack projectors).
    """
    def _search(obj, depth=0):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if depth >= 4:
            return None
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                val = getattr(obj, key)
            except (AttributeError, TypeError):
                continue
            if isinstance(val, nn.Module):
                result = _search(val, depth + 1)
                if result is not None:
                    return result
        return None

    result = _search(model)
    if result is not None:
        return result
    raise ValueError(f"Cannot find projector '{attr}' on {type(model).__name__}.")



import inspect
def _filter_fwd(model, batch):
    allowed = set(inspect.signature(model.forward).parameters)
    return {k: v for k, v in batch.items() if k in allowed}


def _proj_forward_local(model, pixel_values):
    """Vision tower (no_grad) + projector (with grad). Returns fp32 features."""
    with torch.no_grad():
        vt_out = model.vision_tower(pixel_values, output_hidden_states=True)
        if hasattr(vt_out, 'hidden_states') and vt_out.hidden_states is not None:
            sl = getattr(model.config, 'mm_vision_select_layer', -2)
            img_feats = vt_out.hidden_states[sl]
        else:
            img_feats = vt_out.last_hidden_state
        if getattr(model.config, 'mm_vision_select_feature', 'patch') == 'patch':
            img_feats = img_feats[:, 1:, :]  # drop CLS
    return model.multi_modal_projector(img_feats).float()


class ANPMaskHook:
    """
    Forward hook: theta * (1+delta) * W on a projector Linear.
    theta in [0,1]^out_features (outer min), delta in [-eps,eps] (inner max).
    """
    def __init__(self, linear: nn.Linear, device: torch.device):
        self.linear  = linear
        out_f, in_f  = linear.weight.shape
        self._orig_w = linear.weight.data.float().clone()
        self._orig_b = linear.bias.data.float().clone() if linear.bias is not None else None
        self.delta   = torch.zeros(out_f, in_f, device=device, dtype=torch.float32)
        self.delta_b = (torch.zeros(out_f, device=device, dtype=torch.float32)
                        if self._orig_b is not None else None)
        self.theta   = torch.ones(out_f, device=device, dtype=torch.float32)
        self._handle = linear.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        x = inputs[0]
        w_eff = self.theta[:, None] * (1.0 + self.delta) * self._orig_w.to(x.device)
        b_eff = (self.theta * (1.0 + self.delta_b) * self._orig_b.to(x.device)
                 if self._orig_b is not None else None)
        return F.linear(x.float(), w_eff, b_eff).to(x.dtype)

    def remove(self):         self._handle.remove()
    def clamp_delta_(self, e):
        self.delta.data.clamp_(-e, e)
        if self.delta_b is not None: self.delta_b.data.clamp_(-e, e)
    def clamp_theta_(self):   self.theta.data.clamp_(0.0, 1.0)
    def init_delta_(self, eps):
        self.delta.data.uniform_(-eps, eps)
        if self.delta_b is not None: self.delta_b.data.uniform_(-eps, eps)
    def set_grad(self, delta_req, theta_req):
        self.delta.requires_grad_(delta_req)
        if self.delta_b is not None: self.delta_b.requires_grad_(delta_req)
        self.theta.requires_grad_(theta_req)
    def zero_grad_(self):
        for t in [self.delta, self.delta_b, self.theta]:
            if t is not None and t.grad is not None: t.grad.zero_()
    @property
    def delta_params(self): return [t for t in [self.delta, self.delta_b] if t is not None]
    def num_neurons(self):  return int(self.theta.shape[0])
    def hard_prune(self, threshold):
        mask = (self.theta.detach() >= threshold).float()
        n_pruned = int((mask == 0).sum().item())
        self.theta.data.copy_(mask)
        return n_pruned


def _get_all_projectors(model, projector_attr):
    """
    Return a list of projector modules to hook.

    For Qwen3-VL (which has primary merger + deepstack_merger_list), returns
    all of them so that ANP optimizes the full projector ensemble.

    For other VLMs (LLaVA, etc.) with a single projector, returns a single-element
    list so the rest of the logic is unchanged.
    """
    primary = _find_projector(model, projector_attr)
    visual = _find_visual(model)
    if visual is not None and hasattr(visual, "deepstack_merger_list"):
        projectors = [primary]
        for m in visual.deepstack_merger_list:
            projectors.append(m)
        return projectors
    return [primary]


def _attach_hooks(model, projector_attr):
    """
    Register ANPMaskHook on every Linear layer in the projector(s).
    For Qwen3-VL this means the primary merger AND all deepstack mergers.
    """
    projectors = _get_all_projectors(model, projector_attr)
    hooks = []
    for proj in projectors:
        dev = next(proj.parameters()).device
        for name, m in proj.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(ANPMaskHook(m, dev))
    if not hooks:
        raise ValueError("No Linear layers found in projector(s).")
    return hooks


def _detach_hooks(hooks):
    for h in hooks: h.remove()


def _inner_max(model, train_model, hooks, clean_batch, eps, pgd_steps, step_size):
    """
    Step A: max_{|delta|<=eps} L_CE(f(x_clean; theta*(1+delta)*W), y_clean)
    Exactly as in ANP (Wu & Wang, NeurIPS 2021): PGD on clean data to find
    the worst-case multiplicative perturbation delta that maximises CE loss.
    Backdoor neurons are fragile on clean data (their weights encode trigger
    patterns, not clean semantics), so their delta grows large; clean neurons
    are robust and their delta stays small.

    train_model is used for forward/backward (in FSDP mode it is the FSDP-wrapped model).
    model is used for zero_grad and hook state (it holds the ANPMaskHook objects).

    Numerical stability:
    - If loss is nan/inf for a step, skip that step's update (don't break early,
      keep iterating — subsequent steps may recover after clamp_delta_).
    - Gradients are nan-filtered and clipped to [-1, 1] before sign update.
    """
    clean_pv = clean_batch.get("pixel_values")
    if clean_pv is None:
        return
    for h in hooks:
        h.set_grad(delta_req=True, theta_req=False)
        h.init_delta_(eps)
    fwd_batch = _filter_fwd(model, clean_batch)
    for _ in range(pgd_steps):
        for h in hooks: h.zero_grad_()
        out = train_model(**fwd_batch)
        loss = out.loss.float()
        if not torch.isfinite(loss):
            model.zero_grad(set_to_none=True)
            continue
        loss.backward()
        for h in hooks:
            for t in h.delta_params:
                if t.grad is not None:
                    g = t.grad
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    g = g.clamp(-1.0, 1.0)
                    t.data.add_(step_size * g.sign())
            h.clamp_delta_(eps)
    for h in hooks: h.set_grad(False, False)
    model.zero_grad(set_to_none=True)


def _outer_min(model, train_model, hooks, clean_batch, theta_lr, lam, clean_loss_weight):
    """
    Step B (outer minimization):

      min_theta  L_adv(theta, delta*)
              + clean_loss_weight * L_clean(theta, delta=0)
              + lam * ||theta||_1

    where:
      - L_adv uses the adversarial delta found by inner max
      - L_clean is the original clean CE loss with delta=0 (utility retention term)

    Adding L_clean prevents over-pruning that reduces ASR but destroys benign utility.

    train_model is used for forward/backward (in FSDP mode it is the FSDP-wrapped model).
    model is used for zero_grad and hook state.
    """
    fwd_batch = _filter_fwd(model, clean_batch)
    for h in hooks:
        h.set_grad(delta_req=False, theta_req=True)
        h.zero_grad_()

    # Phase 1: adversarial loss under current delta* — backprop immediately
    # so the computation graph is freed before phase 2.
    out_adv = train_model(**fwd_batch)
    adv_loss = out_adv.loss.float()
    adv_finite = torch.isfinite(adv_loss)
    adv_loss_val = float(adv_loss.item()) if adv_finite else float("nan")
    if adv_finite:
        adv_loss.backward()
    del out_adv, adv_loss

    # Phase 2: clean loss with delta=0 — backprop immediately.
    # Gradients on theta accumulate from both phases (equivalent to
    # backpropping the sum, but only one graph lives in memory at a time).
    delta_backup = [h.delta.data.clone() for h in hooks]
    delta_b_backup = [h.delta_b.data.clone() if h.delta_b is not None else None for h in hooks]
    for h in hooks:
        h.delta.data.zero_()
        if h.delta_b is not None:
            h.delta_b.data.zero_()

    out_clean = train_model(**fwd_batch)
    clean_loss = out_clean.loss.float()
    clean_finite = torch.isfinite(clean_loss)
    clean_loss_val = float(clean_loss.item()) if clean_finite else float("nan")
    if adv_finite and clean_finite:
        (clean_loss_weight * clean_loss).backward()
    del out_clean, clean_loss

    for h, d, db in zip(hooks, delta_backup, delta_b_backup):
        h.delta.data.copy_(d)
        if db is not None and h.delta_b is not None:
            h.delta_b.data.copy_(db)
    del delta_backup, delta_b_backup

    # Phase 3: L1 regularization + theta update.
    # d(lam * sum(theta)) / d(theta_i) = lam, added manually to avoid a graph.
    if adv_finite and clean_finite:
        for h in hooks:
            if h.theta.grad is not None:
                h.theta.grad.add_(lam)
            else:
                h.theta.grad = torch.full_like(h.theta, lam)
            g = h.theta.grad
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            g = g.clamp(-1.0, 1.0)
            h.theta.data.sub_(theta_lr * g)
            h.clamp_theta_()
    else:
        for h in hooks: h.zero_grad_()

    for h in hooks:
        h.set_grad(False, False)
    model.zero_grad(set_to_none=True)

    return {
        "adv_loss": adv_loss_val,
        "clean_loss": clean_loss_val,
    }



def anp_defense(
    model,
    dataloader,
    triggered_loader=None,
    eps: float = 0.02,
    pgd_steps: int = 10,
    pgd_step_size: Optional[float] = None,
    theta_lr: float = 0.1,
    lam: float = 0.01,
    clean_loss_weight: float = 1.0,
    n_rounds: int = 1000,
    prune_threshold: float = 0.30,
    log_interval: int = 100,
    projector_attr: str = "multi_modal_projector",
    fp16: bool = True,
    bf16: bool = False,
    device: str = "cuda:0",
    use_fsdp: bool = False,
    # kept for CLI compat
    prune_ratio: float = 0.05,
    n_batches: int = 64,
    mode: str = "anp",
):
    """
    Run clean-data ANP-style projector pruning.

    Implementation details:
    - Inner max: PGD over multiplicative perturbation `delta` on clean batches.
    - Outer min: update `theta` using adversarial clean CE + clean CE + L1.
    - Hard prune: binarize `theta` by `prune_threshold` at the end.

    Multi-projector support (Qwen3-VL):
    - When the visual module has a `deepstack_merger_list`, ANP hooks and
      optimizes ALL projectors (primary merger + all deepstack mergers).
    - The pruned state dict includes keys for all projectors.
    - `apply_pruning` loads each sub-projector from the flattened dict.

    Multi-GPU / FSDP (use_fsdp=True):
    - Must be launched via `torchrun` with >1 local rank.
    - The LLM backbone is sharded across GPUs using FullyShardedDataParallel,
      reducing per-GPU LLM memory from ~16 GB to ~16/N GB.
    - Projectors stay FULL (replicated) on every rank.
    - Theta/delta are updated on rank 0 only and broadcast to all ranks each round.
    - Only rank 0 prints logs; other ranks stay silent.
    """
    rank = _rank()
    world_size = _world_size()

    if pgd_step_size is None:
        pgd_step_size = 2.0 * eps / max(pgd_steps, 1)

    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    _gc_enabled = False
    try:
        lm = getattr(model, "language_model", None) or getattr(getattr(model, "model", None), "language_model", None)
        # Fallback for models without language_model attr (e.g., Qwen3-VL
        # where decoder layers live directly under model.model)
        if lm is None:
            candidate = getattr(model, "model", None)
            if candidate is not None and hasattr(candidate, "gradient_checkpointing_enable"):
                lm = candidate
        if lm is not None and hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable()
            # transformers checks `self.gradient_checkpointing and self.training`
            # inside the inner model (e.g. LlamaModel / Qwen3VLModel).
            # model.eval() set training=False everywhere, so checkpointing is
            # silently skipped.  Fix: flip training=True ONLY on the inner model
            # that holds the checkpointing flag — individual decoder layers keep
            # training=False so dropout / batchnorm stay in eval mode.
            inner = getattr(lm, "model", lm)
            inner.training = True
            _gc_enabled = True
            if rank == 0:
                print("[ANP] gradient checkpointing enabled on language_model", flush=True)
    except Exception as _e:
        if rank == 0:
            print(f"[ANP] gradient checkpointing not available: {_e}", flush=True)

    # ── FSDP setup (multi-GPU) ───────────────────────────────────────────────
    wrapped_llm = None
    if use_fsdp:
        if world_size == 1:
            if rank == 0:
                print("[ANP] WARNING: use_fsdp=True but world_size=1 — running single-GPU", flush=True)
            use_fsdp = False
        else:
            wrapped_llm, projectors, dev = _setup_fsdp(model, projector_attr)
            # model.language_model is now the FSDP-wrapped module.
            # Calling model.forward() will internally use the wrapped LLM for LLM layers.

    # Rank 0 owns the projector → primary device
    visual = _find_visual(model)
    has_ds = hasattr(visual, "deepstack_merger_list") and visual.deepstack_merger_list is not None
    n_ds = len(visual.deepstack_merger_list) if has_ds else 0
    hooks = _attach_hooks(model, projector_attr)
    projectors = _get_all_projectors(model, projector_attr)
    total_neurons = sum(h.num_neurons() for h in hooks)
    dev = next(projectors[0].parameters()).device
    input_dev = next(model.parameters()).device

    if rank == 0:
        # Per-projector breakdown: show each Linear layer's out_features
        proj_detail = []
        for pi, proj in enumerate(projectors):
            linear_info = []
            for name, m in proj.named_modules():
                if isinstance(m, nn.Linear):
                    linear_info.append(f"{name}({m.weight.shape[0]})")
            proj_detail.append(f"proj{pi}[{' '.join(linear_info)}]")
        print(f"[ANP] DEBUG: visual={type(visual).__name__}, "
              f"deepstack_merger_list={'present('+str(n_ds)+')' if has_ds else 'absent'}, "
              f"projectors={proj_detail}", flush=True)
        print(f"[ANP] defense start: rounds={n_rounds}, eps={eps}, pgd_steps={pgd_steps}, "
              f"theta_lr={theta_lr}, lam={lam}, clean_w={clean_loss_weight}, threshold={prune_threshold}, "
              f"neurons={total_neurons} ({len(projectors)} projector(s)), "
              f"use_fsdp={use_fsdp}, world_size={world_size}", flush=True)

    # --- memory budget check before starting ---
    mem_allocated = torch.cuda.memory_allocated(dev) / 1024**3
    mem_reserved  = torch.cuda.memory_reserved(dev)  / 1024**3
    if rank == 0:
        print(f"[ANP] GPU memory at start — allocated={mem_allocated:.2f} GiB, "
              f"reserved={mem_reserved:.2f} GiB", flush=True)
        if mem_reserved > 20.0:
            print(f"[ANP] WARNING: GPU memory usage is already high ({mem_reserved:.2f} GiB). "
                  f"Consider reducing n_rounds or batch_size to avoid OOM.", flush=True)

    # train_model == model in all cases.
    # In FSDP mode, model.language_model has been replaced with the FSDP wrapper,
    # so model.forward() automatically uses the sharded LLM.
    train_model = model

    clean_iter = iter(dataloader)

    # --- adaptive eps decay state (rank 0 only) ---
    _nan_streak   = 0
    _NAN_PATIENCE = 20
    _MIN_EPS      = 1e-4
    _cur_eps      = eps
    _cur_step     = pgd_step_size

    for rnd in range(1, n_rounds + 1):
        # Each rank fetches its own batch slice (DataLoader is shared across ranks)
        try: clean_batch = next(clean_iter)
        except StopIteration:
            clean_iter = iter(dataloader)
            clean_batch = next(clean_iter)

        clean_pv = clean_batch.get("pixel_values")
        if clean_pv is None: continue
        clean_pv = clean_pv.to(input_dev).to(dtype)

        clean_batch_dev = {k: v.to(input_dev) if isinstance(v, torch.Tensor) else v
                          for k, v in clean_batch.items()}
        if fp16 and "pixel_values" in clean_batch_dev:
            clean_batch_dev["pixel_values"] = clean_batch_dev["pixel_values"].to(dtype)

        # ── FSDP: only rank 0 updates theta; broadcast delta/theta to all ranks ──
        if rank == 0:
            _inner_max(model, train_model, hooks, clean_batch_dev, _cur_eps, pgd_steps, _cur_step)
            torch.cuda.empty_cache()
            loss_info = _outer_min(model, train_model, hooks, clean_batch_dev, theta_lr, lam, clean_loss_weight)
            adv_loss_val = loss_info["adv_loss"]
            clean_loss_val = loss_info["clean_loss"]
        else:
            # non-rank-0 ranks: only do forward to keep FSDP state consistent
            # (FSDP requires matching forward calls across ranks)
            fwd_batch = _filter_fwd(train_model, clean_batch_dev)
            _ = train_model(**fwd_batch)
            train_model.zero_grad(set_to_none=True)
            adv_loss_val, clean_loss_val = float("nan"), float("nan")

        # Sync theta/delta from rank 0 to all ranks so projector hooks stay consistent
        _broadcast_theta(hooks)
        torch.cuda.synchronize()

        # Release memory between rounds
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # --- adaptive eps: rank 0 only ---
        if rank == 0:
            if adv_loss_val != adv_loss_val:
                _nan_streak += 1
                if _nan_streak >= _NAN_PATIENCE and _cur_eps > _MIN_EPS:
                    _cur_eps  = max(_cur_eps * 0.5, _MIN_EPS)
                    _cur_step = 2.0 * _cur_eps / max(pgd_steps, 1)
                    _nan_streak = 0
                    print(f"[ANP] nan streak detected, reducing eps -> {_cur_eps:.5f}  "
                          f"step_size -> {_cur_step:.6f}", flush=True)
            else:
                _nan_streak = 0

        if rank == 0 and (rnd % log_interval == 0 or rnd == n_rounds):
            soft_pruned = sum(int((h.theta.detach() < prune_threshold).sum().item()) for h in hooks)
            theta_min   = min(h.theta.detach().min().item() for h in hooks)
            theta_mean  = sum(h.theta.detach().mean().item() for h in hooks) / len(hooks)
            p10_vals = []
            for h in hooks:
                t = h.theta.detach()
                p10_vals.append(float(torch.quantile(t, 0.10).item()))
            p10_str = "/".join(f"{v:.3f}" for v in p10_vals)
            adv_loss_str  = f"{adv_loss_val:.4f}" if adv_loss_val == adv_loss_val else "nan"
            clean_loss_str = f"{clean_loss_val:.4f}" if clean_loss_val == clean_loss_val else "nan"
            mem_allocated = torch.cuda.memory_allocated(dev) / 1024**3
            mem_reserved  = torch.cuda.memory_reserved(dev)  / 1024**3
            print(f"[ANP] round={rnd}/{n_rounds}  adv_loss={adv_loss_str}  clean_loss={clean_loss_str}  "
                  f"eps={_cur_eps:.5f}  theta_min={theta_min:.4f}  theta_mean={theta_mean:.4f}  "
                  f"theta_p10(layers)=[{p10_str}]  "
                  f"soft_pruned={soft_pruned}/{total_neurons} "
                  f"({100*soft_pruned/max(total_neurons,1):.1f}%)  "
                  f"mem_alloc={mem_allocated:.2f}GiB  mem_rsv={mem_reserved:.2f}GiB", flush=True)

    if rank == 0:
        total_pruned = sum(h.hard_prune(prune_threshold) for h in hooks)
        print(f"[ANP] hard prune done: {total_pruned}/{total_neurons} "
              f"({100*total_pruned/max(total_neurons,1):.1f}%)", flush=True)

    # Build pruned state dict (rank 0 only has the full theta on projector)
    pruned_sd = _build_pruned_sd(model, hooks, projector_attr)
    _detach_hooks(hooks)

    if wrapped_llm is not None:
        del wrapped_llm
        torch.cuda.empty_cache()

    return pruned_sd


def _build_pruned_sd(model, hooks, projector_attr):
    """
    Build the pruned state_dict for all hooked projectors.

    For Qwen3-VL (multiple projectors): builds merged state dict with keys like
    'merger.linear_fc1.weight', 'deepstack_merger_list.0.linear_fc1.weight', etc.

    For single-projector models (LLaVA): behaviour unchanged.
    """
    visual = _find_visual(model)
    projectors = _get_all_projectors(model, projector_attr)

    # Build a flat list of (projector, hook_list) pairs so we can walk hooks in order
    proj_hook_pairs = []
    hook_idx = 0
    for proj in projectors:
        proj_linears = [(n, m) for n, m in proj.named_modules() if isinstance(m, nn.Linear)]
        n_linear = len(proj_linears)
        proj_hook_pairs.append((proj, hooks[hook_idx:hook_idx + n_linear]))
        hook_idx += n_linear

    # Collect state dicts from all projectors, applying theta scaling
    pruned_sd = {}
    for proj, proj_hooks in proj_hook_pairs:
        # Prefix for keys: "merger", "deepstack_merger_list.0", etc.
        if proj is getattr(visual, "merger", None) if visual else False:
            prefix = "merger"
        else:
            # It's a deepstack merger — find its index
            prefix = None
            if visual and hasattr(visual, "deepstack_merger_list"):
                for idx, m in enumerate(visual.deepstack_merger_list):
                    if proj is m:
                        prefix = f"deepstack_merger_list.{idx}"
                        break
            if prefix is None:
                # fallback: use the module's own name within its parent
                prefix = "unknown_proj"

        full_sd = proj.state_dict()
        proj_pruned = {k: v.clone() for k, v in full_sd.items()}

        # Map linear modules to their hooks
        linears = [(n, m) for n, m in proj.named_modules() if isinstance(m, nn.Linear)]
        for i, (name, linear) in enumerate(linears):
            h = proj_hooks[i]
            p = (name + ".") if name else ""
            w_key = p + "weight"
            b_key = p + "bias"
            if w_key in proj_pruned:
                proj_pruned[w_key] = (h.theta.detach()[:, None] * h._orig_w).to(proj_pruned[w_key].dtype)
            if b_key in proj_pruned and h._orig_b is not None:
                proj_pruned[b_key] = (h.theta.detach() * h._orig_b).to(proj_pruned[b_key].dtype)

        # Add keys to global pruned_sd
        if len(projectors) == 1:
            pruned_sd.update(proj_pruned)
        else:
            for k, v in proj_pruned.items():
                pruned_sd[f"{prefix}.{k}"] = v

    return pruned_sd


def apply_pruning(model, pruned_sd, projector_attr="multi_modal_projector"):
    """
    Load the pruned state_dict into the model projector(s) in-place.

    For Qwen3-VL (multi-projector): pruned_sd contains keys like
    'merger.linear_fc1.weight', 'deepstack_merger_list.0.linear_fc1.weight', etc.
    Each sub-projector gets its own state dict loaded.

    For single-projector models (LLaVA): behaviour unchanged (pruned_sd is just
    the projector's state dict).
    """
    visual = _find_visual(model)
    projectors = _get_all_projectors(model, projector_attr)

    if len(projectors) == 1:
        # Single projector — original behaviour
        proj = projectors[0]
        missing, unexpected = proj.load_state_dict(pruned_sd, strict=False)
    else:
        # Multiple projectors (Qwen3-VL): load into each from the flattened dict
        missing, unexpected = [], []
        for proj in projectors:
            if proj is getattr(visual, "merger", None) if visual else False:
                prefix = "merger"
            else:
                prefix = None
                if visual and hasattr(visual, "deepstack_merger_list"):
                    for idx, m in enumerate(visual.deepstack_merger_list):
                        if proj is m:
                            prefix = f"deepstack_merger_list.{idx}"
                            break
                if prefix is None:
                    prefix = "unknown_proj"

            # Extract the sub-dict for this projector
            sub_sd = {}
            prefix_dot = prefix + "."
            for k, v in pruned_sd.items():
                if k == prefix:
                    # bare key (unlikely but handle it)
                    sub_sd[k] = v
                elif k.startswith(prefix_dot):
                    sub_sd[k[len(prefix_dot):]] = v

            if sub_sd:
                m, u = proj.load_state_dict(sub_sd, strict=False)
                missing.extend(m)
                unexpected.extend(u)

    if missing:    logger.warning(f"[ANP] missing keys: {missing}")
    if unexpected: logger.warning(f"[ANP] unexpected keys: {unexpected}")
    print("[ANP] pruned projector(s) loaded into model.", flush=True)
