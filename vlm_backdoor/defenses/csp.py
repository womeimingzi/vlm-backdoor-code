"""
Clean-Subspace Projection (CSP) — Backdoor purification for VLM projectors.

Algorithm (K-FAC approximation):
  For each linear layer in the projector:
    1. Collect input activations h_i and output gradients δ_i on N clean samples.
    2. A = (1/N) Σ h_i h_i^T  (in_dim × in_dim)
    3. B = (1/N) Σ δ_i δ_i^T  (out_dim × out_dim)
    4. Take top-r eigenvectors V_A, V_B covering ≥ energy_threshold of total energy.
    5. ΔW_pur = V_B V_B^T ΔW V_A V_A^T
    6. W_pur = W_0 + ΔW_pur

Reference: docs/ideas_docs/method_gpt.md
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _top_r_by_energy(eigenvalues: torch.Tensor, threshold: float) -> int:
    """Return the minimum r such that top-r eigenvalues cover >= threshold of total energy."""
    eigenvalues = eigenvalues.clamp(min=0)  # numerical safety
    sorted_vals, _ = torch.sort(eigenvalues, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    total = cumsum[-1]
    if total == 0:
        return 1
    ratios = cumsum / total
    r = int((ratios >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1
    return max(1, r)


class CSPurifier:
    """
    Clean-Subspace Projection purifier for LLaVA's multi_modal_projector.

    Usage:
        purifier = CSPurifier(model, energy_threshold=0.95)
        pure_state = purifier.purify(P_b_state, P_0_state, clean_dataloader, n_samples=50)
        model.multi_modal_projector.load_state_dict(pure_state)
    """

    def __init__(self, model: nn.Module, energy_threshold: float = 0.95):
        """
        Args:
            model: LlavaForConditionalGeneration with P_b already loaded.
            energy_threshold: Fraction of Fisher energy to retain (default 0.95).
        """
        self.model = model
        self.energy_threshold = energy_threshold

    # ------------------------------------------------------------------
    # Fisher estimation via K-FAC
    # ------------------------------------------------------------------

    def estimate_kfac(
        self,
        clean_dataloader,
        n_samples: int = 50,
        per_token: bool = True,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run N clean samples through the model and collect K-FAC statistics for
        each linear layer in multi_modal_projector.

        Args:
            clean_dataloader: yields batches with input_ids, attention_mask, pixel_values, labels
            n_samples: number of clean samples to use
            per_token: if True (recommended), accumulate per-token outer products
                       A = Σ_t h_t h_t^T, B = Σ_t g_t g_t^T  (correct K-FAC)
                       if False, use mean-then-outer (original, biased downward)

        Returns:
            dict mapping layer_name -> (A, B)
            where A is input covariance and B is output-gradient covariance.
        """
        projector = self.model.multi_modal_projector

        # Identify linear layers
        linear_layers: Dict[str, nn.Linear] = {
            name: module
            for name, module in projector.named_modules()
            if isinstance(module, nn.Linear)
        }
        logger.info(f"[CSP] Found linear layers: {list(linear_layers.keys())}")

        # Accumulators: sums over samples
        A_acc: Dict[str, torch.Tensor] = {}
        B_acc: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}

        for name in linear_layers:
            in_dim = linear_layers[name].in_features
            out_dim = linear_layers[name].out_features
            A_acc[name] = torch.zeros(in_dim, in_dim, dtype=torch.float32)
            B_acc[name] = torch.zeros(out_dim, out_dim, dtype=torch.float32)
            counts[name] = 0

        # Hook storage (cleared between samples)
        saved_inputs: Dict[str, torch.Tensor] = {}
        saved_grads: Dict[str, torch.Tensor] = {}

        handles = []
        for name, layer in linear_layers.items():
            # Forward hook: capture input activations
            def make_fwd_hook(n):
                def fwd_hook(module, inp, out):
                    # inp[0] shape: (batch, seq, in_dim) or (batch, in_dim)
                    h = inp[0].detach().float()
                    h = h.reshape(-1, h.shape[-1])  # (T, in_dim)
                    saved_inputs[n] = h
                return fwd_hook

            def make_bwd_hook(n):
                def bwd_hook(module, grad_in, grad_out):
                    # grad_out[0] shape: (batch, seq, out_dim) or (batch, out_dim)
                    g = grad_out[0].detach().float()
                    g = g.reshape(-1, g.shape[-1])  # (T, out_dim)
                    saved_grads[n] = g
                return bwd_hook

            handles.append(layer.register_forward_hook(make_fwd_hook(name)))
            handles.append(layer.register_full_backward_hook(make_bwd_hook(name)))

        self.model.eval()
        sample_count = 0

        try:
            for batch in clean_dataloader:
                if sample_count >= n_samples:
                    break

                saved_inputs.clear()
                saved_grads.clear()

                # Move to model device
                device = next(self.model.parameters()).device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
                labels = batch["labels"].to(device)

                # Forward + backward
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                self.model.zero_grad()

                # Accumulate K-FAC statistics
                for name in linear_layers:
                    if name in saved_inputs and name in saved_grads:
                        h = saved_inputs[name].cpu()   # (T, in_dim)
                        g = saved_grads[name].cpu()    # (T, out_dim)
                        if per_token:
                            # Correct K-FAC: Σ_t h_t h_t^T (each token is a sample)
                            A_acc[name] += h.T @ h          # (in_dim, in_dim)
                            B_acc[name] += g.T @ g          # (out_dim, out_dim)
                            counts[name] += h.shape[0]      # count tokens
                        else:
                            # Original (biased): outer product of per-sample means
                            h_mean = h.mean(dim=0)
                            g_mean = g.mean(dim=0)
                            A_acc[name] += torch.outer(h_mean, h_mean)
                            B_acc[name] += torch.outer(g_mean, g_mean)
                            counts[name] += 1

                sample_count += 1
                if sample_count % 10 == 0:
                    logger.info(f"[CSP] Processed {sample_count}/{n_samples} clean samples")

        finally:
            for h in handles:
                h.remove()

        # Normalize
        kfac: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in linear_layers:
            n = max(counts[name], 1)
            A = A_acc[name] / n
            B = B_acc[name] / n
            kfac[name] = (A, B)
            logger.info(f"[CSP] K-FAC for {name}: A {A.shape}, B {B.shape}, samples={n}")

        return kfac

    # ------------------------------------------------------------------
    # Subspace extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_subspace(M: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, int, float]:
        """
        Eigen-decompose M (symmetric PSD), return top-r eigenvectors.

        Returns:
            V_r: (dim × r) float32 eigenvectors
            r:   number of components retained
            energy: fraction of energy retained
        """
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(M.double())
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()

        # Flip to descending order
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        r = _top_r_by_energy(eigenvalues, threshold)
        V_r = eigenvectors[:, :r]  # (dim × r)

        total = eigenvalues.clamp(min=0).sum().item()
        retained = eigenvalues[:r].clamp(min=0).sum().item()
        energy = retained / total if total > 0 else 1.0

        return V_r, r, energy

    # ------------------------------------------------------------------
    # Main purification
    # ------------------------------------------------------------------

    def purify(
        self,
        P_b_state: Dict[str, torch.Tensor],
        P_0_state: Dict[str, torch.Tensor],
        clean_dataloader,
        n_samples: int = 50,
        fisher_state: Optional[Dict[str, torch.Tensor]] = None,
        per_token: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Compute purified projector state dict using CSP.

        Args:
            P_b_state: backdoored projector state dict
            P_0_state: clean (original) projector state dict
            clean_dataloader: yields batches with input_ids, attention_mask, pixel_values, labels
            n_samples: number of clean samples for Fisher estimation
            fisher_state: which projector to load for Fisher estimation.
                          Defaults to P_b_state. Pass P_0_state to estimate the
                          clean subspace from the original model (Direction A).
            per_token: if True (recommended), use per-token K-FAC accumulation.

        Returns:
            (pure_state_dict, meta_info)
            meta_info contains per-layer rank, energy captured, etc.
        """
        _fisher_state = P_b_state if fisher_state is None else fisher_state
        fisher_label = "P_0" if (fisher_state is not None and fisher_state is not P_b_state) else "P_b"
        logger.info(f"[CSP] Starting purification: n_samples={n_samples}, "
                    f"energy_threshold={self.energy_threshold}, "
                    f"fisher_from={fisher_label}, per_token={per_token}")

        # Step 1: Load chosen state into model for Fisher estimation
        self.model.multi_modal_projector.load_state_dict(_fisher_state)

        # Step 2: Estimate K-FAC Fisher
        kfac = self.estimate_kfac(clean_dataloader, n_samples=n_samples, per_token=per_token)

        # Step 3: Identify matching weight keys
        #   P_b_state keys: e.g. "linear_1.weight", "linear_1.bias", "linear_2.weight", ...
        weight_keys = [k for k in P_b_state if k.endswith(".weight")]

        pure_state = {}
        meta_info = {"layers": {}}

        for key in P_b_state:
            W_b = P_b_state[key].float()
            W_0 = P_0_state[key].float()

            if not key.endswith(".weight"):
                # bias: use P_0's value (conservative)
                pure_state[key] = W_0.clone()
                continue

            # Derive layer name (e.g. "linear_1.weight" -> "linear_1")
            layer_name = key[: -len(".weight")]

            if layer_name not in kfac:
                logger.warning(f"[CSP] No K-FAC stats for {layer_name}, keeping P_b weight.")
                pure_state[key] = W_b.clone()
                continue

            A, B = kfac[layer_name]
            A = A.float()
            B = B.float()

            # Step 4: Get clean subspace bases
            V_A, r_A, energy_A = self._get_subspace(A, self.energy_threshold)
            V_B, r_B, energy_B = self._get_subspace(B, self.energy_threshold)

            logger.info(f"[CSP] {key}: r_A={r_A}/{A.shape[0]} ({energy_A:.3f}), "
                        f"r_B={r_B}/{B.shape[0]} ({energy_B:.3f})")

            # Step 5: Project ΔW onto clean subspace
            delta_W = W_b - W_0  # (out_dim, in_dim)
            # Project input directions: ΔW @ V_A @ V_A^T
            delta_proj_A = delta_W @ V_A @ V_A.T
            # Project output directions: V_B @ V_B^T @ (projected)
            delta_pur = V_B @ (V_B.T @ delta_proj_A)

            # Step 6: Reconstruct purified weight
            W_pur = W_0 + delta_pur
            pure_state[key] = W_pur.to(P_b_state[key].dtype)

            meta_info["layers"][key] = {
                "r_A": r_A,
                "r_B": r_B,
                "energy_A": round(energy_A, 4),
                "energy_B": round(energy_B, 4),
                "in_dim": A.shape[0],
                "out_dim": B.shape[0],
                "delta_norm_orig": float(delta_W.norm()),
                "delta_norm_pur": float(delta_pur.norm()),
            }

        meta_info["n_samples"] = n_samples
        meta_info["energy_threshold"] = self.energy_threshold
        meta_info["fisher_from"] = fisher_label
        meta_info["per_token"] = per_token

        logger.info("[CSP] Purification complete.")
        return pure_state, meta_info
