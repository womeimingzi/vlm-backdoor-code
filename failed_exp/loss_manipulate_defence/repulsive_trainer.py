import torch
import torch.nn as nn
from transformers import Trainer
from llava.train.llava_trainer import LLaVATrainer
import copy
import logging

logger = logging.getLogger(__name__)

class RepulsiveTrainer_LLaVA(LLaVATrainer):
    def __init__(self, *args, w_pre_path=None, w_bad_path=None, repulsion_lambda=1.0, extrapolation_eta=0.1, epsilon=1e-8, **kwargs):
        """
        Trainer for Weight-Difference Guided Unlearning (WDGU).
        Args:
            w_pre_path: Path to the clean pre-trained mm_projector weights (e.g. from base LLaVA)
            w_bad_path: Path to the backdoored mm_projector weights
            repulsion_lambda: Coefficient for the repulsive penalty term
            extrapolation_eta: Step size for exploring backwards along the poison path
            epsilon: Small value to prevent division by zero in repulsive loss
        """
        super().__init__(*args, **kwargs)
        self.repulsion_lambda = float(repulsion_lambda)
        self.extrapolation_eta = float(extrapolation_eta)
        self.epsilon = float(epsilon)
        
        logger.info(f"Initializing RepulsiveTrainer with lambda={self.repulsion_lambda}, eta={self.extrapolation_eta}")
        
        # 1. Load reference weights to compute the poison path (Delta W)
        self.device = self.args.device
        self._load_and_compute_path(w_pre_path, w_bad_path)

    def _load_and_compute_path(self, w_pre_path, w_bad_path):
        """Loads W_pre and W_bad, and computes Delta_W_poison = W_bad - W_pre"""
        if not w_pre_path or not w_bad_path:
            logger.warning("w_pre_path or w_bad_path not provided. Repulsion will be disabled.")
            self.delta_w = None
            self.w_bad_state = None
            return

        logger.info(f"Loading W_pre from {w_pre_path}")
        w_pre_state = torch.load(w_pre_path, map_location=self.device)
        logger.info(f"Loading W_bad from {w_bad_path}")
        w_bad_state = torch.load(w_bad_path, map_location=self.device)

        # Handle potential prefix differences (LLaVA uses model.mm_projector.0.weight etc.)
        def safe_get_tensor(state_dict, key_suffix):
            for k, v in state_dict.items():
                if k.endswith(key_suffix):
                    return v.to(self.device).float()
            raise KeyError(f"Key ending with {key_suffix} not found in state dict")

        self.delta_w = {}
        self.w_bad_tensors = {}
        
        # Typically mm_projector has 0.weight, 0.bias, 2.weight, 2.bias or linear_1.weight, etc.
        keys_options = [
            ["0.weight", "0.bias", "2.weight", "2.bias"],
            ["linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"]
        ]
        
        keys_to_track = None
        for opts in keys_options:
            try:
                safe_get_tensor(w_pre_state, opts[0])
                safe_get_tensor(w_bad_state, opts[0])
                keys_to_track = opts
                break
            except KeyError:
                pass
                
        if keys_to_track is None:
            raise KeyError("Could not find matching mm_projector keys (0.weight or linear_1.weight) in both state dicts")
            
        logger.info(f"Using keys to track: {keys_to_track}")
        
        for key in keys_to_track:
            try:
                t_pre = safe_get_tensor(w_pre_state, key)
                t_bad = safe_get_tensor(w_bad_state, key)
                self.delta_w[key] = (t_bad - t_pre).detach()
                self.w_bad_tensors[key] = t_bad.detach()
            except KeyError as e:
                logger.warning(f"Skipping key {key} for repulsion: {e}")

    def _get_active_projector_tensors(self):
        """Yields (suffix_key, param_tensor) for active mm_projector parameters."""
        for name, param in self.model.named_parameters():
            if "mm_projector" not in name:
                continue
            for key in ["0.weight", "0.bias", "2.weight", "2.bias"]:
                if name.endswith(key):
                    yield key, param
                    break

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overrides the standard compute_loss to inject WDGU logic:
        1. Extrapolate active weights backwards along poison path.
        2. Compute standard cross entropy loss at the extrapolated point.
        3. Add repulsive penalty based on distance to W_bad.
        4. Revert weights back to original state for backprop.
        """
        if self.delta_w is None or self.repulsion_lambda <= 0:
            # Fallback to standard LLaVA loss if WDGU is disabled
            labels = inputs.pop("labels")
            if "images" in inputs:
                inputs["pixel_values"] = inputs.pop("images")
            outputs = model(**inputs, labels=labels)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # --- Extrapolation Phase ---
        # Save original weights and mutate to W_explore
        original_weights = {}
        with torch.no_grad():
            for key, param in self._get_active_projector_tensors():
                if key in self.delta_w:
                    original_weights[key] = param.data.clone()
                    # W_explore = W - eta * (Delta_W / ||Delta_W||)
                    delta_tensor = self.delta_w[key]
                    norm = torch.norm(delta_tensor) + self.epsilon
                    normalized_delta = delta_tensor / norm
                    param.data.sub_(normalized_delta * self.extrapolation_eta)

        # --- Forward Pass (at W_explore) ---
        labels = inputs.pop("labels")
        if "images" in inputs:
            inputs["pixel_values"] = inputs.pop("images")
        
        # Compute outputs using the mutated model weights
        outputs = model(**inputs, labels=labels)
        base_loss = outputs.loss # This is L_clean(W_explore)

        # --- Revert Weights ---
        # We must revert the weights immediately so gradients are applied to true W(t), not W_explore
        with torch.no_grad():
            for key, param in self._get_active_projector_tensors():
                if key in original_weights:
                    param.data.copy_(original_weights[key])

        # --- Repulsion Phase ---
        # Calculate distance penalty to prevent returning to W_bad
        repulsive_loss = 0.0
        dist_sq_sum = 0.0
        
        # We compute this dynamically tracking gradients so it adds to the backward pass
        for key, param in self._get_active_projector_tensors():
            if key in self.w_bad_tensors:
                bad_tensor = self.w_bad_tensors[key]
                # L2 distance squared between current W(t) and W_bad
                dist_sq = torch.sum((param - bad_tensor) ** 2)
                dist_sq_sum += dist_sq
                
        # Loss_repel = lambda * (1 / (||W(t) - W_bad||^2 + epsilon))
        repulsive_loss = self.repulsion_lambda * (1.0 / (dist_sq_sum + self.epsilon))
        
        total_loss = base_loss + repulsive_loss

        return (total_loss, outputs) if return_outputs else total_loss
