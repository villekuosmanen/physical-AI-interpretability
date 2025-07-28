from typing import Optional

import cv2
import numpy as np
import torch

# Constants for tensor dimensions
TENSOR_DIM_3D = 3
TENSOR_DIM_4D = 4


class ACTPolicyWithAttention:
    """Wrapper for ACTPolicy that provides transformer attention visualizations.
    """

    def __init__(self, policy, image_shapes=None, specific_decoder_token_index: Optional[int] = None):
        """Initialize the wrapper with an ACTPolicy.

        Args:
            policy: An instance of ACTPolicy
            image_shapes: Optional list of image shapes [(H1, W1), (H2, W2), ...] if known in advance
            specific_decoder_token_index: experimental, allows visualising attention maps for a particular token rather than averaging all outputs.
        """
        self.policy = policy
        self.config = policy.config

        self.specific_decoder_token_index = specific_decoder_token_index
        if self.specific_decoder_token_index is not None:
            if not hasattr(self.config, 'chunk_size'):
                raise AttributeError("Policy's config object does not have 'chunk_size' attribute.")
            if not (0 <= self.specific_decoder_token_index < self.config.chunk_size):
                raise ValueError(
                    f"specific_decoder_token_index ({self.specific_decoder_token_index}) "
                    f"must be between 0 and chunk_size-1 ({self.config.chunk_size - 1})."
                )

        # Determine number of images from config
        if self.config.image_features:
            self.num_images = len(self.config.image_features)
        else:
            self.num_images = 0

        # Store image shapes if provided, otherwise will be detected at runtime
        self.image_shapes = image_shapes

        # For storing the last processed images and attention
        self.last_observation = None
        self.last_attention_maps = None

        if not hasattr(self.policy, 'model') or \
        not hasattr(self.policy.model, 'decoder') or \
        not hasattr(self.policy.model.decoder, 'layers') or \
        not self.policy.model.decoder.layers:
            raise AttributeError("Policy model structure does not match expected ACT architecture for target_layer.")
        self.target_layer = self.policy.model.decoder.layers[-1].multihead_attn

    def select_action(self, observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
        """Extends policy.select_action to also compute attention maps.

        Args:
            observation: Dictionary of observations

        Returns:
            action: The predicted action tensor
            attention_maps: List of attention maps, one for each image
        """
        # Store the observation for later use
        self.last_observation = observation.copy()

        # Process the images through the backbone first to understand spatial dimensions
        images = self._extract_images(observation)
        image_spatial_shapes = self._get_image_spatial_shapes(images)

        # Set up hook to capture attention weights
        attention_weights_capture = []

        def attention_hook(module, _input_args, output_tuple):
            # Capture the attention weights
            # In some MultiheadAttention implementations, the attention weights
            # might be returned with shape: [batch_size, tgt_len, src_len]
            # or [batch_size, num_heads, tgt_len, src_len]
            if isinstance(output_tuple, tuple) and len(output_tuple) > 1:
                # If output is a tuple with attention weights as second element
                attn_weights = output_tuple[1]
            else:
                # If output format is different, try to get weights from the module directly
                # Some implementations store attention weights in the module after forward pass
                attn_weights = getattr(module, 'attn_weights', None)

            if attn_weights is not None:
                # Store the weights regardless of shape - we'll handle reshape later
                attention_weights_capture.append(attn_weights.detach().cpu())

        # Register the hook
        handle = self.target_layer.register_forward_hook(attention_hook)

        # Call the original policy's select_action
        with torch.inference_mode():
            action = self.policy.select_action(observation, force_model_run=True)

        # Remove the hook
        handle.remove()

        # Process the attention weights
        if attention_weights_capture:
            attn = attention_weights_capture[0].to(action.device)
            attention_maps, proprio_attention = self._map_attention_to_images(attn, image_spatial_shapes)
            self.last_attention_maps = attention_maps
            self.last_proprio_attention = proprio_attention  # Store for visualization
        else:
            print("Warning: No attention weights were captured.")
            attention_maps = [None] * self.num_images
            self.last_attention_maps = attention_maps
            self.last_proprio_attention = 0.0  # Store for visualization

        return action, attention_maps

    def _extract_images(self, observation: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract image tensors from observation dictionary."""
        images = []
        for key in self.config.image_features:
            if key in observation:
                images.append(observation[key])
        return images

    def _get_image_spatial_shapes(self, images: list[torch.Tensor]) -> list[tuple[int, int]]:
        """Get the spatial shapes of the feature maps after ResNet processing.
        For ResNet, this is typically H/32 × W/32.
        """
        spatial_shapes = []
        for img_tensor in images:
            if img_tensor is None:
                spatial_shapes.append((0, 0))
                continue

            # Run image through backbone to get feature map shape
            with torch.no_grad():
                if img_tensor.dim() == TENSOR_DIM_3D:
                    img_tensor_batched = img_tensor.unsqueeze(0)
                else:
                    img_tensor_batched = img_tensor

                img_tensor_batched = img_tensor_batched.to(next(self.policy.model.backbone.parameters()).device)

                feature_map_dict = self.policy.model.backbone(img_tensor_batched) # Use batched tensor
                feature_map = feature_map_dict["feature_map"]
                h, w = feature_map.shape[2], feature_map.shape[3]
                spatial_shapes.append((h, w))

        return spatial_shapes

    def _map_attention_to_images(self,
                                attention: torch.Tensor,
                                image_spatial_shapes: list[tuple[int, int]]) -> tuple[list[np.ndarray], float]:
        """Map transformer attention weights back to the original images and extract proprioception attention.

        Normalizes attention maps globally across all images AND proprioception for this timestep.

        Args:
            attention: Tensor of shape [batch, heads, tgt_len, src_len]
                       (tgt_len is config.chunk_size)
            image_spatial_shapes: List of (height, width) tuples for feature maps

        Returns:
            Tuple of:

            - List of globally normalized attention maps as numpy arrays

            - Proprioception attention value (float, normalized to same scale as visual attention)
        """
        if attention.dim() == TENSOR_DIM_4D:
            attention = attention.mean(dim=1)  # -> [batch, tgt_len, src_len]
        elif attention.dim() != TENSOR_DIM_3D:
            raise ValueError(f"Unexpected attention dimension: {attention.shape}. Expected 3 or 4.")

        # Token structure: [latent, (robot_state), (env_state), (image_tokens)]
        n_prefix_tokens = 1  # latent token
        proprio_token_idx = None
        if self.config.robot_state_feature:
            proprio_token_idx = n_prefix_tokens  # proprioception is the next token
            n_prefix_tokens += 1
        if self.config.env_state_feature:
            n_prefix_tokens += 1

        # --- Step 1: Extract proprioception attention ---
        proprio_attention = 0.0
        if proprio_token_idx is not None:
            # Extract attention to proprioception token
            if self.specific_decoder_token_index is not None:
                if 0 <= self.specific_decoder_token_index < attention.shape[1]:
                    proprio_attention_tensor = attention[:, self.specific_decoder_token_index, proprio_token_idx]
                else:
                    proprio_attention_tensor = attention[:, :, proprio_token_idx].mean(dim=1)
            else:
                proprio_attention_tensor = attention[:, :, proprio_token_idx].mean(dim=1)

            # Take first batch element
            proprio_attention = proprio_attention_tensor[0].cpu().numpy().item()

        # --- Step 2: Collect all raw (unnormalized) 2D numpy attention maps ---
        raw_numpy_attention_maps = []
        # Store the per-image token counts for reshaping, needed later
        tokens_per_image = [h * w for h, w in image_spatial_shapes]


        current_src_token_idx = n_prefix_tokens
        for i, (h_feat, w_feat) in enumerate(image_spatial_shapes):
            if h_feat == 0 or w_feat == 0:
                raw_numpy_attention_maps.append(None)
                if tokens_per_image[i] > 0: # if shape was (0,0) but tokens_per_image[i] was not 0
                    current_src_token_idx += tokens_per_image[i]
                continue

            num_img_tokens = tokens_per_image[i]
            start_idx = current_src_token_idx
            end_idx = start_idx + num_img_tokens
            current_src_token_idx = end_idx

            attention_to_img_features = attention[:, :, start_idx:end_idx]

            if self.specific_decoder_token_index is not None:
                if not (0 <= self.specific_decoder_token_index < attention_to_img_features.shape[1]):
                    print(f"Warning (map_attention): specific_decoder_token_index {self.specific_decoder_token_index} "
                          f"is out of bounds for actual tgt_len {attention_to_img_features.shape[1]}. "
                          f"Falling back to averaging.")
                    img_attn_tensor_for_map = attention_to_img_features.mean(dim=1)
                else:
                    img_attn_tensor_for_map = attention_to_img_features[:, self.specific_decoder_token_index, :]
            else:
                img_attn_tensor_for_map = attention_to_img_features.mean(dim=1)

            if img_attn_tensor_for_map.shape[0] > 1 and i == 0: # Print once
                 print(f"Warning (map_attention): Batch size is {img_attn_tensor_for_map.shape[0]}. Processing first element for attention map.")

            if img_attn_tensor_for_map.shape[1] != num_img_tokens:
                print(f"Warning (map_attention): Mismatch in token count for image {i}. "
                      f"Expected {num_img_tokens}, got {img_attn_tensor_for_map.shape[1]}. "
                      f"Skipping map for this image.")
                raw_numpy_attention_maps.append(None)
                continue

            try:
                # Get the tensor for the first batch item, still on device
                img_attn_map_1d_tensor = img_attn_tensor_for_map[0] # [num_img_tokens]
                # Reshape to 2D tensor
                img_attn_map_2d_tensor = img_attn_map_1d_tensor.reshape(h_feat, w_feat)
                raw_numpy_attention_maps.append(img_attn_map_2d_tensor.cpu().numpy())
            except RuntimeError as e:
                print(f"Error (map_attention): Reshaping attention for image {i}: {e}. "
                      f"Shape was {img_attn_tensor_for_map[0].shape}, target HxW: {h_feat}x{w_feat}. "
                      f"Num tokens: {num_img_tokens}. Skipping.")
                raw_numpy_attention_maps.append(None)
                continue

        # --- Step 3: Find global min and max from all valid raw maps AND proprioception ---
        global_min = float('inf')
        global_max = float('-inf')
        found_any_valid_map = False

        # Include proprioception attention in global scaling
        if proprio_attention is not None:
            global_min = min(global_min, proprio_attention)
            global_max = max(global_max, proprio_attention)
            found_any_valid_map = True

        for raw_map_np in raw_numpy_attention_maps:
            if raw_map_np is not None:
                current_min = raw_map_np.min()
                current_max = raw_map_np.max()
                global_min = min(global_min, current_min)
                global_max = max(global_max, current_max)
                found_any_valid_map = True

        if not found_any_valid_map:
            # All maps were None, return the list of Nones
            return raw_numpy_attention_maps, 0.0

        # If global_min and global_max are still inf/-inf, it means all maps were empty or had issues
        # This case should be covered by found_any_valid_map, but as a safe guard:
        if global_min == float('inf') or global_max == float('-inf'):
            print("Warning (map_attention): Could not determine global min/max for attention. All maps might be invalid.")
            # Fallback: return unnormalized maps or list of Nones
            return [np.zeros_like(m, dtype=np.float32) if m is not None else None for m in raw_numpy_attention_maps], 0.0

        # --- Step 4: Normalize proprioception attention ---
        if global_max > global_min:
            normalized_proprio_attention = (proprio_attention - global_min) / (global_max - global_min)
        else:
            normalized_proprio_attention = 0.0

        # --- Step 5: Normalize all valid visual attention maps using global min/max ---
        final_normalized_attention_maps = []
        for raw_map_np in raw_numpy_attention_maps:
            if raw_map_np is None:
                final_normalized_attention_maps.append(None)
                continue

            if global_max > global_min:
                # Perform normalization
                normalized_map = (raw_map_np - global_min) / (global_max - global_min)
            else:
                # All values across all valid maps are the same (e.g., all are 0.001, or all are 0)
                # Create a uniform map (e.g., all zeros or all 0.5s)
                # If global_max == global_min, it implies all values are equal to global_min (or global_max).
                # If global_min is 0, then (raw_map_np - 0) / (0-0) is problematic.
                # A common practice is to make such a map uniform, often zeros.
                normalized_map = np.zeros_like(raw_map_np, dtype=np.float32)
                # If you prefer a mid-gray for perfectly flat attention:
                # normalized_map = np.full_like(raw_map_np, 0.5, dtype=np.float32)
            final_normalized_attention_maps.append(normalized_map)

        return final_normalized_attention_maps, normalized_proprio_attention

    def visualize_attention(self,
                        images: Optional[list[torch.Tensor]] = None,
                        attention_maps: Optional[list[np.ndarray]] = None,
                        observation: Optional[dict[str, torch.Tensor]] = None,
                        use_rgb: bool = False,
                        overlay_alpha: float = 0.5,
                        show_proprio_border: bool = True,
                        proprio_border_width: int = 15) -> list[np.ndarray]:
        """Create visualizations by overlaying attention maps on images.

        Args:
            images: List of image tensors (optional)
            attention_maps: List of attention maps (optional)
            observation: Observation dict (optional, used if images not provided)
            use_rgb: Whether to use RGB for visualization
            overlay_alpha: Alpha value for attention overlay
            show_proprio_border: Whether to show proprioception attention as border
            proprio_border_width: Width of the proprioception attention border

        Returns:
            List of visualization images as numpy arrays
        """
        # If no images provided, use from observation or last observation
        if images is None:
            if observation is not None:
                images = self._extract_images(observation)
            elif self.last_observation is not None:
                images = self._extract_images(self.last_observation)
            else:
                raise ValueError("No images provided and no stored observation available")

        # If no attention maps provided, use last computed ones
        if attention_maps is None:
            if self.last_attention_maps is not None:
                attention_maps = self.last_attention_maps
            else:
                raise ValueError("No attention maps provided and no stored attention maps available")

        # Get proprioception attention value
        proprio_attention = getattr(self, 'last_proprio_attention', 0.0)
        visualizations = []

        for _i, (img, attn_map) in enumerate(zip(images, attention_maps)):
            if img is None or attn_map is None:
                visualizations.append(None)
                continue

            # Convert tensor to numpy
            if isinstance(img, torch.Tensor):
                # Move channels to last dimension (H,W,C) for visualization
                if img.dim() == TENSOR_DIM_4D:  # (B,C,H,W)
                    img_tensor = img.squeeze(0)
                else:
                    img_tensor = img
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # Normalize if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            else:
                img_np = img

            # Get image dimensions
            h, w = img_np.shape[:2]

            # Resize attention map to match image size
            attn_map_resized = cv2.resize(attn_map, (w, h))

            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
            if use_rgb:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Create overlay with attention
            vis = cv2.addWeighted(
                np.uint8(255 * img_np), 1 - overlay_alpha,
                heatmap, overlay_alpha, 0
            )

            # Add proprioception attention border
            if show_proprio_border and proprio_attention > 0:
                # Convert normalized proprioception attention to color intensity
                border_intensity = int(255 * proprio_attention)
                # Create border color (use a different colormap for proprioception)
                # Using magenta/purple to distinguish from visual attention
                if use_rgb:
                    border_color = (border_intensity, 0, border_intensity)  # Magenta in RGB
                else:
                    border_color = (border_intensity, 0, border_intensity)  # Magenta in BGR

                # Draw border rectangles (outer and inner rectangles to create border effect)
                # Outer rectangle (full border)
                cv2.rectangle(vis, (0, 0), (w-1, h-1), border_color, proprio_border_width)

                # Optional: Add text label showing proprioception attention value
                text = f"Proprio: {proprio_attention:.3f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                # Draw background rectangle for text
                cv2.rectangle(vis, (5, 5), (5 + text_width + 10, 5 + text_height + 10), (0, 0, 0), -1)

                # Draw text
                cv2.putText(vis, text, (10, 5 + text_height), font, font_scale, (255, 255, 255), thickness)

            visualizations.append(vis)

        return visualizations

    # Forward other methods to the original policy
    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.policy, name)
        return self.__dict__[name]
