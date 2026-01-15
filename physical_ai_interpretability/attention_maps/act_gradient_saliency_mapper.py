import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


class ACTPolicyWithGradients:
    """
    Wrapper for ACTPolicy that provides gradient-based saliency visualizations using Integrated Gradients.
    """
    
    def __init__(self, policy, preprocessor, image_shapes=None, n_integration_steps: int = 50,
                 normalization_mode: str = 'percentile', normalization_percentile: float = 95.0):
        """
        Initialize the wrapper with an ACTPolicy.
        
        Args:
            policy: An instance of ACTPolicy or RewACTPolicy
            preprocessor: Preprocessor for observations
            image_shapes: Optional list of image shapes [(H1, W1), (H2, W2), ...] if known in advance
            n_integration_steps: Number of steps for Integrated Gradients interpolation
            normalization_mode: 'linear' for min-max, 'percentile' for percentile clipping (default)
            normalization_percentile: Percentile to clip at when using percentile mode (default: 95.0)
        """
        self.policy = policy
        self.preprocessor = preprocessor
        self.config = policy.config
        self.n_integration_steps = n_integration_steps
        self.normalization_mode = normalization_mode
        self.normalization_percentile = normalization_percentile
        
        # Determine number of images from config
        if self.config.image_features:
            self.num_images = len(self.config.image_features)
        else:
            self.num_images = 0
            
        # Store image shapes if provided, otherwise will be detected at runtime
        self.image_shapes = image_shapes
        
        # For storing the last processed data
        self.last_observation = None
        self.last_saliency_maps = None
        self.last_proprio_saliency = None
        
    def select_action(self, 
                     observation: Dict[str, torch.Tensor],
                     aggregation: str = 'sum',
                     timestep_idx: Optional[int] = None,
                     action_dim_idx: Optional[int] = None) -> Tuple[torch.Tensor, List[np.ndarray], float]:
        """
        Extends policy.select_action to also compute gradient-based saliency maps.
        
        Args:
            observation: Dictionary of observations
            aggregation: How to aggregate trajectory predictions for gradient computation
                        'sum' - sum all predictions (default)
                        'norm' - L2 norm of all predictions
                        'timestep' - specific timestep (requires timestep_idx)
                        'action_dim' - specific action dimension (requires action_dim_idx)
            timestep_idx: Which timestep to analyze (if aggregation='timestep')
            action_dim_idx: Which action dimension to analyze (if aggregation='action_dim')
            
        Returns:
            action: The predicted action tensor
            saliency_maps: List of saliency maps (one per camera), normalized to [0, 1]
            proprio_saliency: Scalar indicating proprioception importance
        """
        # Store the observation
        self.last_observation = observation.copy()
        
        # Extract images and get their spatial shapes
        images = self._extract_images(observation)
        
        # Compute Integrated Gradients
        image_saliencies, proprio_saliency = self._compute_integrated_gradients(
            observation, 
            aggregation=aggregation,
            timestep_idx=timestep_idx,
            action_dim_idx=action_dim_idx
        )
        
        # Store results
        self.last_saliency_maps = image_saliencies
        self.last_proprio_saliency = proprio_saliency
        
        # Get the actual action prediction
        observation_copy = observation.copy()
        observation_copy['observation.state'] = observation_copy['observation.state.pos']
        observation_copy = self.preprocessor(observation_copy)
        
        with torch.inference_mode():
            action = self.policy.select_action(observation_copy)
            if isinstance(action, tuple):
                action = action[0]
            self.policy.reset()
        
        return action, image_saliencies, proprio_saliency

    def _extract_images(self, observation: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Extract image tensors from observation dictionary"""
        images = []
        for key in self.config.image_features:
            if key in observation:
                images.append(observation[key])
        return images
    
    def _compute_integrated_gradients(self,
                                     observation: Dict[str, torch.Tensor],
                                     aggregation: str = 'sum',
                                     timestep_idx: Optional[int] = None,
                                     action_dim_idx: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """
        Compute Integrated Gradients for both visual and proprioceptive inputs.
        
        Args:
            observation: Dictionary of observations
            aggregation: How to reduce trajectory to scalar for gradient computation
            timestep_idx: Which timestep to analyze (if aggregation='timestep')
            action_dim_idx: Which action dimension to analyze (if aggregation='action_dim')
            
        Returns:
            Tuple of:
            - List of saliency maps (one per camera), each normalized to [0, 1]
            - Scalar proprioception saliency value
        """
        self.policy.eval()
        
        # Extract images and create baselines (black images)
        images = self._extract_images(observation)
        image_baselines = [torch.zeros_like(img) for img in images]
        
        # Extract proprioception and create baseline (zeros)
        proprio = observation['observation.state.pos']
        proprio_baseline = torch.zeros_like(proprio)
        
        # Storage for accumulated gradients
        image_integrated_grads = [torch.zeros_like(img) for img in images]
        proprio_integrated_grad = torch.zeros_like(proprio)
        
        # Interpolation coefficients
        alphas = torch.linspace(0, 1, self.n_integration_steps)
        
        # Compute gradients along the interpolation path
        for alpha in tqdm(alphas, desc="Computing Integrated Gradients", leave=False):
            # Create interpolated inputs
            interpolated_images = []
            for img, baseline in zip(images, image_baselines):
                interpolated = baseline + alpha * (img - baseline)
                interpolated.requires_grad = True
                interpolated_images.append(interpolated)
            
            interpolated_proprio = proprio_baseline + alpha * (proprio - proprio_baseline)
            interpolated_proprio.requires_grad = True
            
            # Create observation with interpolated inputs
            obs_interpolated = {}
            for i, key in enumerate(self.config.image_features):
                obs_interpolated[key] = interpolated_images[i]
            obs_interpolated['observation.state.pos'] = interpolated_proprio
            obs_interpolated['observation.state'] = interpolated_proprio
            
            # Preprocess
            obs_interpolated = self.preprocessor(obs_interpolated)
            
            # Forward pass through policy
            # Note: For RewACTPolicy, we need to handle the model structure
            if hasattr(self.policy, 'model'):
                # Construct batch format expected by model
                batch = dict(obs_interpolated)
                if self.config.image_features:
                    batch['observation.images'] = [batch[key] for key in self.config.image_features]
                
                # Forward through model
                actions, reward_output, _ = self.policy.model(batch)
            else:
                raise AttributeError("Policy doesn't have expected model structure")
            
            # Compute scalar loss based on aggregation method
            if aggregation == 'sum':
                loss = actions.sum()
            elif aggregation == 'norm':
                loss = actions.norm()
            elif aggregation == 'timestep':
                if timestep_idx is None:
                    raise ValueError("timestep_idx required for 'timestep' aggregation")
                loss = actions[:, timestep_idx, :].sum()
            elif aggregation == 'action_dim':
                if action_dim_idx is None:
                    raise ValueError("action_dim_idx required for 'action_dim' aggregation")
                loss = actions[:, :, action_dim_idx].sum()
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            for i, img in enumerate(interpolated_images):
                if img.grad is not None:
                    image_integrated_grads[i] += img.grad.detach()
            
            if interpolated_proprio.grad is not None:
                proprio_integrated_grad += interpolated_proprio.grad.detach()
            
            # Zero gradients for next iteration
            self.policy.zero_grad()
        
        # Average the accumulated gradients and multiply by (input - baseline)
        image_saliencies = []
        for i, (img, baseline, integrated_grad) in enumerate(zip(images, image_baselines, image_integrated_grads)):
            integrated_grad = integrated_grad / self.n_integration_steps
            saliency = (img - baseline) * integrated_grad
            
            # Convert to numpy and aggregate across channels
            saliency_np = saliency.squeeze(0).detach().cpu().numpy()  # (C, H, W)
            saliency_map = np.abs(saliency_np).mean(axis=0)  # (H, W) - average across channels
            
            image_saliencies.append(saliency_map)
        
        # Compute proprioception saliency
        proprio_integrated_grad = proprio_integrated_grad / self.n_integration_steps
        proprio_saliency_raw = (proprio - proprio_baseline) * proprio_integrated_grad
        # Take mean of absolute values across all joint dimensions
        proprio_saliency = torch.abs(proprio_saliency_raw).mean().item()
        
        # Normalize saliency maps globally (including proprio)
        image_saliencies, proprio_saliency = self._normalize_saliencies_globally(
            image_saliencies, proprio_saliency
        )
        
        return image_saliencies, proprio_saliency
    
    def _normalize_saliencies_globally(self, 
                                      image_saliencies: List[np.ndarray],
                                      proprio_saliency: float) -> Tuple[List[np.ndarray], float]:
        """
        Normalize all saliency values (visual + proprioception) to [0, 1] using global normalization.
        
        Supports two modes:
        - 'linear': Standard min-max normalization
        - 'percentile': Clip to specified percentile, then normalize (more robust to outliers)
        
        Args:
            image_saliencies: List of saliency maps
            proprio_saliency: Scalar proprioception saliency
            
        Returns:
            Normalized saliency maps and proprioception value
        """
        if self.normalization_mode == 'percentile':
            # Collect all values for percentile calculation
            all_values = [proprio_saliency]
            for saliency_map in image_saliencies:
                if saliency_map is not None:
                    all_values.extend(saliency_map.flatten().tolist())
            
            # Calculate percentile threshold
            percentile_value = np.percentile(all_values, self.normalization_percentile)
            
            # Clip and normalize
            normalized_images = []
            for saliency_map in image_saliencies:
                if saliency_map is not None:
                    clipped = np.clip(saliency_map, 0, percentile_value)
                    normalized = clipped / (percentile_value + 1e-8)  # Add epsilon to avoid div by zero
                    normalized_images.append(normalized)
                else:
                    normalized_images.append(None)
            
            # Normalize proprio
            normalized_proprio = np.clip(proprio_saliency, 0, percentile_value) / (percentile_value + 1e-8)
            
            return normalized_images, float(normalized_proprio)
        
        else:  # 'linear' mode (original min-max normalization)
            # Find global min and max across all saliencies
            global_min = float('inf')
            global_max = float('-inf')
            
            # Check proprioception
            if proprio_saliency < global_min:
                global_min = proprio_saliency
            if proprio_saliency > global_max:
                global_max = proprio_saliency
            
            # Check all image saliency maps
            for saliency_map in image_saliencies:
                if saliency_map is not None:
                    map_min = saliency_map.min()
                    map_max = saliency_map.max()
                    if map_min < global_min:
                        global_min = map_min
                    if map_max > global_max:
                        global_max = map_max
            
            # Normalize
            if global_max > global_min:
                normalized_images = []
                for saliency_map in image_saliencies:
                    if saliency_map is not None:
                        normalized = (saliency_map - global_min) / (global_max - global_min)
                        normalized_images.append(normalized)
                    else:
                        normalized_images.append(None)
                
                normalized_proprio = (proprio_saliency - global_min) / (global_max - global_min)
            else:
                # All values are the same
                normalized_images = [np.zeros_like(m) if m is not None else None 
                                   for m in image_saliencies]
                normalized_proprio = 0.0
            
            return normalized_images, normalized_proprio
    
    def visualize_saliency(self,
                          images: Optional[List[torch.Tensor]] = None,
                          saliency_maps: Optional[List[np.ndarray]] = None,
                          observation: Optional[Dict[str, torch.Tensor]] = None,
                          use_rgb: bool = False,
                          overlay_alpha: float = 0.5,
                          show_proprio_border: bool = True,
                          proprio_border_width: int = 15) -> List[np.ndarray]:
        """
        Create visualizations by overlaying saliency maps on images.
        
        Args:
            images: List of image tensors (optional)
            saliency_maps: List of saliency maps (optional)
            observation: Observation dict (optional, used if images not provided)
            use_rgb: Whether to use RGB for visualization
            overlay_alpha: Alpha value for saliency overlay
            show_proprio_border: Whether to show proprioception importance as border
            proprio_border_width: Width of proprioception border in pixels
            
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
        
        # If no saliency maps provided, use last computed ones
        if saliency_maps is None:
            if self.last_saliency_maps is not None:
                saliency_maps = self.last_saliency_maps
            else:
                raise ValueError("No saliency maps provided and no stored saliency maps available")
        
        # Get proprioception saliency value
        proprio_saliency = getattr(self, 'last_proprio_saliency', 0.0)
        
        visualizations = []
        
        for i, (img, saliency_map) in enumerate(zip(images, saliency_maps)):
            if img is None or saliency_map is None:
                visualizations.append(None)
                continue
            
            # Convert tensor to numpy
            if isinstance(img, torch.Tensor):
                # Move channels to last dimension (H,W,C) for visualization
                if img.dim() == 4:  # (B,C,H,W)
                    img = img.squeeze(0)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                # Normalize if needed
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            else:
                img_np = img
            
            # Get image dimensions
            h, w = img_np.shape[:2]
            
            # Resize saliency map to match image size
            saliency_resized = cv2.resize(saliency_map, (w, h))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
            if use_rgb:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Create overlay with saliency
            vis = cv2.addWeighted(
                np.uint8(255 * img_np), 1 - overlay_alpha,
                heatmap, overlay_alpha, 0
            )
            
            # Add proprioception saliency border
            if show_proprio_border and proprio_saliency > 0:
                # Convert normalized proprioception saliency to color intensity
                border_intensity = int(255 * proprio_saliency)
                # Using magenta/purple to distinguish from visual saliency
                if use_rgb:
                    border_color = (border_intensity, 0, border_intensity)  # Magenta in RGB
                else:
                    border_color = (border_intensity, 0, border_intensity)  # Magenta in BGR
                
                # Draw border
                cv2.rectangle(vis, (0, 0), (w-1, h-1), border_color, proprio_border_width)
                
                # Add text label
                text = f"Proprio: {proprio_saliency:.3f}"
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