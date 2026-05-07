"""ACT encoder attention rollout.

Composes self-attention across all encoder layers with the residual stream
(Abnar & Zuidema, 2020) so we can trace what the *proprio memory token*
effectively attends to in the original input — including vision tokens that
the decoder never reads directly.

Why this matters: standard decoder cross-attention can show the decoder
reading 100% from the proprio token, while inside the encoder that proprio
token has already absorbed vision content via self-attention. Rollout
recovers that hidden vision dependency.

Token layout in the ACT encoder input:
    0     : latent (VAE)
    1     : proprio (state)
    2..N-1: vision tokens (flattened backbone feature map)

Rollout method (per layer):
    A_eff = (A + I) / 2     # mix in residual stream
    A_eff = A_eff / sum(-1) # renormalise rows
Then R = A_L @ ... @ A_2 @ A_1.

The row R[1] tells us how much of the proprio output's effective input
attention falls on each input token; R[1, 2:] reshaped to (H_feat, W_feat)
is a spatial attribution map on the original image.

Reference: https://arxiv.org/abs/2005.00928
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


# Token offsets in the ACT encoder input sequence.
LATENT_TOKEN_IDX = 0
PROPRIO_TOKEN_IDX = 1
FIRST_VISION_TOKEN_IDX = 2


class ACTPolicyWithEncoderRollout:
    """Wrapper for ACTPolicy that computes encoder attention rollout.

    Usage:
        wrapper = ACTPolicyWithEncoderRollout(policy, preprocessor)
        result = wrapper.rollout_for_observation(obs)
        # result.spatial_heatmap : (H_feat, W_feat) numpy array
        # result.vision_share    : float in [0, 1]
        overlay = wrapper.overlay_on_image(result.spatial_heatmap, image_bgr)
    """

    def __init__(self, policy, preprocessor):
        """
        Args:
            policy: An instance of ACTPolicy.
            preprocessor: The preprocessor returned by make_pre_post_processors,
                applied to observations before policy.predict_action_chunk.
        """
        self.policy = policy
        self.preprocessor = preprocessor
        self.config = policy.config

        if not hasattr(policy, 'model') or \
           not hasattr(policy.model, 'encoder') or \
           not hasattr(policy.model.encoder, 'layers') or \
           not policy.model.encoder.layers:
            raise AttributeError(
                "Policy must expose policy.model.encoder.layers (ACT encoder)."
            )

        self.num_encoder_layers = len(policy.model.encoder.layers)

    # ----- core rollout -----

    @staticmethod
    def _compose_rollout(
        attn_per_layer: List[torch.Tensor],
        residual: bool = True,
    ) -> torch.Tensor:
        """Compose per-layer attention matrices into a single rollout matrix.

        Args:
            attn_per_layer: list of (B, N, N) attention tensors, one per layer,
                ordered from layer 0 (input-side) to last layer (output-side).
            residual: if True, mix in the residual stream as A_eff = (A+I)/2.

        Returns:
            (N, N) tensor for batch index 0.
        """
        R: Optional[torch.Tensor] = None
        for A in attn_per_layer:
            A = A[0]  # batch 0
            if residual:
                A = (A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)) / 2
                A = A / A.sum(-1, keepdim=True)
            R = A if R is None else (A @ R)
        assert R is not None, "no attention layers captured"
        return R

    def _install_hooks(self) -> Tuple[list, list]:
        """Hook every encoder layer's self_attn to capture (B, N, N) weights."""
        captured: List[Tuple[int, torch.Tensor]] = []
        handles = []
        for li, layer in enumerate(self.policy.model.encoder.layers):
            def make_hook(layer_idx: int):
                def hook(module, inputs, output):
                    # nn.MultiheadAttention returns (attn_output, attn_output_weights);
                    # attn_output_weights is None if need_weights=False at the call site.
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        captured.append((layer_idx, output[1].detach().cpu()))
                return hook
            handles.append(layer.self_attn.register_forward_hook(make_hook(li)))
        return captured, handles

    def get_feature_grid_shape(
        self,
        device: Optional[torch.device] = None,
        H_in: int = 360,
        W_in: int = 640,
    ) -> Tuple[int, int]:
        """Run the backbone once on a dummy input to recover (H_feat, W_feat)."""
        if device is None:
            device = next(self.policy.model.backbone.parameters()).device
        dummy = torch.zeros(1, 3, H_in, W_in, device=device)
        with torch.no_grad():
            feat = self.policy.model.backbone(dummy)["feature_map"]
        return feat.shape[-2], feat.shape[-1]

    # ----- main API -----

    def rollout_for_observation(
        self,
        observation: dict,
        residual: bool = True,
    ) -> "RolloutResult":
        """Compute the encoder rollout for a single observation.

        Args:
            observation: dict in the ACT input format, e.g.
                {"observation.images.front": (1, 3, H, W), "observation.state": (1, S)}.
            residual: pass-through to the rollout composition.

        Returns:
            RolloutResult with spatial heatmap and per-token shares.
        """
        captured, handles = self._install_hooks()
        try:
            obs_p = self.preprocessor(dict(observation))
            self.policy.reset()
            with torch.inference_mode():
                _ = self.policy.predict_action_chunk(obs_p)
        finally:
            for h in handles:
                h.remove()

        if not captured:
            raise RuntimeError(
                "No encoder attention captured. Confirm self_attn returns weights "
                "(need_weights=True is the default for nn.MultiheadAttention)."
            )

        captured.sort(key=lambda x: x[0])
        attns = [a for _, a in captured]
        R = self._compose_rollout(attns, residual=residual)
        N = R.shape[0]

        proprio_row = R[PROPRIO_TOKEN_IDX]
        latent_share = proprio_row[LATENT_TOKEN_IDX].item()
        proprio_share = proprio_row[PROPRIO_TOKEN_IDX].item()
        vision_share = proprio_row[FIRST_VISION_TOKEN_IDX:].sum().item()

        H_feat, W_feat = self.get_feature_grid_shape()
        n_vision = H_feat * W_feat
        spatial = None
        if N - FIRST_VISION_TOKEN_IDX == n_vision:
            spatial = proprio_row[FIRST_VISION_TOKEN_IDX:].reshape(H_feat, W_feat).numpy()

        return RolloutResult(
            rollout_matrix=R.numpy(),
            spatial_heatmap=spatial,
            feature_grid_shape=(H_feat, W_feat),
            latent_share=float(latent_share),
            proprio_share=float(proprio_share),
            vision_share=float(vision_share),
        )

    @staticmethod
    def overlay_on_image(
        spatial_heatmap: np.ndarray,
        image_bgr: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Upsample a (H_feat, W_feat) heatmap and blend it onto a BGR image."""
        if spatial_heatmap is None:
            return image_bgr.copy()
        H, W = image_bgr.shape[:2]
        heat = cv2.resize(spatial_heatmap.astype(np.float32), (W, H))
        if heat.max() > heat.min():
            heat = (heat - heat.min()) / (heat.max() - heat.min())
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), colormap)
        return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)


class RolloutResult:
    """Container for one rollout call."""

    def __init__(
        self,
        rollout_matrix: np.ndarray,
        spatial_heatmap: Optional[np.ndarray],
        feature_grid_shape: Tuple[int, int],
        latent_share: float,
        proprio_share: float,
        vision_share: float,
    ):
        self.rollout_matrix = rollout_matrix
        self.spatial_heatmap = spatial_heatmap
        self.feature_grid_shape = feature_grid_shape
        self.latent_share = latent_share
        self.proprio_share = proprio_share
        self.vision_share = vision_share

    def as_dict(self) -> dict:
        return {
            "feature_grid_shape": list(self.feature_grid_shape),
            "latent_share": self.latent_share,
            "proprio_share": self.proprio_share,
            "vision_share": self.vision_share,
        }
