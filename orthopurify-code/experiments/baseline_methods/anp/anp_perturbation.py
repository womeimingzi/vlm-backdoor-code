"""
ANP-style multiplicative weight perturbation for VLM projector layers.

Strictly follows the formulation from:
  "Adversarial Neuron Pruning Purifies Backdoored Neural Networks"
  Wu & Wang, NeurIPS 2021.

For each Linear layer in the projector, we introduce per-element
multiplicative perturbation masks (delta for weights, xi for biases):

    w_eff = (1 + delta) * w,   delta in [-eps, eps]
    b_eff = (1 + xi)    * b,   xi    in [-eps, eps]

We then solve:
    max_{delta, xi in [-eps,eps]}  L_{D_v}(model with perturbed projector)

via PGD on a small clean validation set D_v.

Key finding to verify (ANP prerequisite):
    Poisoned models exhibit a LARGER loss increase (delta_L = L_adv - L_base)
    than clean models under the same perturbation budget eps.
    This is because backdoor-related weights are more sensitive to
    multiplicative perturbation on clean data.
"""
import contextlib
import inspect
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Finding projector
# ---------------------------------------------------------------------------

def _find_projector(model, projector_attr: str = "multi_modal_projector") -> nn.Module:
    """Locate the projector nn.Module regardless of wrapper depth."""
    if hasattr(model, projector_attr):
        return getattr(model, projector_attr)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, projector_attr):
        return getattr(inner, projector_attr)
    for name, module in model.named_modules():
        if "projector" in name.lower() or "merger" in name.lower():
            return module
    raise ValueError(
        f"Cannot find projector module '{projector_attr}' on model "
        f"{type(model).__name__}. Pass the correct projector_attr."
    )


def _filter_forward_kwargs(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop batch keys that model.forward does not accept."""
    sig = inspect.signature(model.forward)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in batch.items() if k in allowed}


# ---------------------------------------------------------------------------
# Multiplicative mask hooks  (ANP perturbation)
# ---------------------------------------------------------------------------

class _MaskHook:
    """
    Holds a per-element multiplicative mask for one Linear layer and
    registers forward hooks so that:
        output = F.linear((1+mask_w)*weight, (1+mask_b)*bias, input)

    The mask is stored as a plain Tensor (not nn.Parameter) so it does
    not pollute model.parameters(); we set requires_grad manually.
    """

    def __init__(self, linear: nn.Linear, device: torch.device, dtype: torch.dtype):
        self.linear = linear
        self._orig_weight = linear.weight.data.clone()          # fp32 backup
        self._orig_bias   = linear.bias.data.clone() if linear.bias is not None else None

        # masks initialised to 0  =>  no perturbation at start
        self.mask_w = torch.zeros_like(self._orig_weight, dtype=torch.float32, device=device)
        self.mask_b = (
            torch.zeros_like(self._orig_bias, dtype=torch.float32, device=device)
            if self._orig_bias is not None else None
        )

        self._pre_handle = linear.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module: nn.Linear, inputs, output):
        """Replace output with F.linear((1+mask_w)*w, (1+mask_b)*b, input).

        Computation is done in fp32 to avoid fp16 overflow when mask is large,
        then cast back to the input dtype so downstream layers are unaffected.
        """
        import torch.nn.functional as F
        x = inputs[0]  # (B, seq, in_features)
        out_dtype = x.dtype
        # upcast to fp32 for numerical stability
        x_f = x.float()
        w_orig = self._orig_weight.float().to(device=x.device)
        w_eff  = (1.0 + self.mask_w.float()) * w_orig
        if module.bias is not None and self.mask_b is not None:
            b_orig = self._orig_bias.float().to(device=x.device)
            b_eff  = (1.0 + self.mask_b.float()) * b_orig
        else:
            b_eff  = module.bias.float() if module.bias is not None else None
        out_f = F.linear(x_f, w_eff, b_eff)
        return out_f.to(dtype=out_dtype)

    def restore(self):
        """Remove forward hook (original weights were never modified)."""
        self._pre_handle.remove()

    def requires_grad_(self, flag: bool):
        self.mask_w.requires_grad_(flag)
        if self.mask_b is not None:
            self.mask_b.requires_grad_(flag)

    def zero_grad(self):
        if self.mask_w.grad is not None:
            self.mask_w.grad.zero_()
        if self.mask_b is not None and self.mask_b.grad is not None:
            self.mask_b.grad.zero_()

    def clamp_(self, eps: float):
        self.mask_w.data.clamp_(-eps, eps)
        if self.mask_b is not None:
            self.mask_b.data.clamp_(-eps, eps)

    def random_init_(self, eps: float):
        self.mask_w.data.uniform_(-eps, eps)
        if self.mask_b is not None:
            self.mask_b.data.uniform_(-eps, eps)

    def reset_(self):
        self.mask_w.data.zero_()
        if self.mask_b is not None:
            self.mask_b.data.zero_()

    @property
    def all_mask_tensors(self) -> List[torch.Tensor]:
        if self.mask_b is not None:
            return [self.mask_w, self.mask_b]
        return [self.mask_w]

    def num_elements(self) -> int:
        n = self.mask_w.numel()
        if self.mask_b is not None:
            n += self.mask_b.numel()
        return n


def attach_masks(
    model,
    projector_attr: str = "multi_modal_projector",
) -> List[_MaskHook]:
    """
    Attach multiplicative mask hooks to every Linear layer inside the projector.
    Returns the list of _MaskHook objects.
    Call detach_masks() when done.
    """
    proj = _find_projector(model, projector_attr)
    device = next(proj.parameters()).device
    dtype  = next(proj.parameters()).dtype
    hooks: List[_MaskHook] = []
    for _name, m in proj.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(_MaskHook(m, device, dtype))
    if not hooks:
        raise ValueError("Projector contains no nn.Linear layers.")
    return hooks


def detach_masks(hooks: List[_MaskHook]) -> None:
    """Restore weights and remove all hooks."""
    for h in hooks:
        h.restore()


@contextlib.contextmanager
def _masks_ctx(
    model,
    projector_attr: str = "multi_modal_projector",
) -> Generator[List[_MaskHook], None, None]:
    hooks = attach_masks(model, projector_attr)
    try:
        yield hooks
    finally:
        detach_masks(hooks)


@contextlib.contextmanager
def _freeze_all_except_masks(model, hooks: List[_MaskHook]):
    """Freeze all model parameters; only mask tensors stay trainable."""
    all_params = list(model.parameters())
    old_flags  = [p.requires_grad for p in all_params]
    for p in all_params:
        p.requires_grad_(False)
    for h in hooks:
        h.requires_grad_(True)
    try:
        yield
    finally:
        for p, flag in zip(all_params, old_flags):
            p.requires_grad_(flag)
        for h in hooks:
            h.requires_grad_(False)


# ---------------------------------------------------------------------------
# PGD optimisation of masks  (maximise loss over D_v)
# ---------------------------------------------------------------------------

def _pgd_step_masks(hooks: List[_MaskHook], step_size: float):
    """Sign-gradient ascent step on all mask tensors."""
    for h in hooks:
        for t in h.all_mask_tensors:
            if t.grad is not None:
                g = t.grad.detach()
                g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
                t.data.add_(step_size * g.sign())


def pgd_maximise_loss(
    model,
    hooks: List[_MaskHook],
    batch: Dict[str, torch.Tensor],
    eps: float,
    pgd_steps: int,
    step_size: Optional[float] = None,
    random_init: bool = True,
) -> None:
    """
    Run PGD to find the worst-case masks in [-eps, eps] that maximise
    the model loss on the given batch.

    After this call the hooks hold the adversarial masks; call
    detach_masks() to restore weights.

    Args:
        model:       The VLM (eval mode, all params frozen outside projector).
        hooks:       _MaskHook list returned by attach_masks().
        batch:       A dict of tensors already on the correct device.
        eps:         Perturbation bound (same meaning as ANP paper).
        pgd_steps:   Number of PGD iterations.
        step_size:   Per-step alpha. Defaults to 2*eps/pgd_steps (standard).
        random_init: Whether to start from a random point in [-eps, eps].
    """
    if step_size is None:
        step_size = 2.0 * eps / max(pgd_steps, 1)

    forward_batch = _filter_forward_kwargs(model, batch)

    if random_init:
        for h in hooks:
            h.random_init_(eps)
    else:
        for h in hooks:
            h.reset_()

    for _ in range(pgd_steps):
        for h in hooks:
            h.zero_grad()

        out  = model(**forward_batch)
        loss = out.loss.float()
        if not torch.isfinite(loss):
            # fp16 overflow: skip this PGD step, masks stay at current value
            model.zero_grad(set_to_none=True)
            break
        loss = loss.clamp(max=1e4)
        loss.backward()

        _pgd_step_masks(hooks, step_size)

        for h in hooks:
            h.clamp_(eps)

    model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# High-level API used by evaluation scripts
# ---------------------------------------------------------------------------

def anp_projector_fragility(
    model,
    batch: Dict[str, torch.Tensor],
    eps: float,
    pgd_steps: int = 10,
    step_size: Optional[float] = None,
    random_init: bool = True,
    projector_attr: str = "multi_modal_projector",
) -> Dict[str, float]:
    """
    Compute base loss (no perturbation) and adversarial loss (worst-case
    ANP multiplicative mask perturbation on projector) for one batch.

    Returns a dict with:
        base_loss    : cross-entropy loss with mask=0
        adv_loss     : cross-entropy loss with adversarial mask
        delta_loss   : adv_loss - base_loss  (fragility indicator)
        num_elements : total number of perturbed scalar parameters
    """
    with _masks_ctx(model, projector_attr) as hooks:

        # --- base loss (mask = 0, already the default) ---
        forward_batch = _filter_forward_kwargs(model, batch)
        with torch.no_grad():
            base_out = model(**forward_batch)
            base_loss = float(base_out.loss.float().clamp(max=1e4).item())

        # --- adversarial loss ---
        with _freeze_all_except_masks(model, hooks):
            pgd_maximise_loss(
                model, hooks, batch, eps, pgd_steps, step_size, random_init
            )

        with torch.no_grad():
            adv_out  = model(**forward_batch)
            adv_loss_raw = adv_out.loss.float()
            if torch.isfinite(adv_loss_raw):
                adv_loss = float(adv_loss_raw.clamp(max=1e4).item())
            else:
                # Overflow: perturbation was too large for fp16 downstream layers.
                # Fall back to base_loss so delta=0 (conservative, not inflated).
                adv_loss = base_loss

        num_elem = sum(h.num_elements() for h in hooks)

    model.zero_grad(set_to_none=True)

    return {
        "base_loss":    base_loss,
        "adv_loss":     adv_loss,
        "delta_loss":   adv_loss - base_loss,
        "num_elements": num_elem,
    }
