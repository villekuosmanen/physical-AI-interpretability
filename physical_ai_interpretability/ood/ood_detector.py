import json
import logging
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy import stats
from huggingface_hub import hf_hub_download, HfApi

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from physical_ai_interpretability.sae.sae import MultiModalSAE
from physical_ai_interpretability.sae.builder import SAEBuilder
from physical_ai_interpretability.sae.token_sampler import TokenSampler, TokenSamplerConfig


class OODDetector:
    """
    Out-of-distribution detector based on detecting unusual attention patterns.
    
    Builds on the SAE model trained to extract features from a given ACT policy.
    It uses the SAE's reconstruction error as a proxy for scenarios that are deemed out of distribution.
    """
    
    def __init__(
        self, 
        policy: ACTPolicy, 
        sae_experiment_path: Optional[str] = None,
        sae_hub_repo_id: Optional[str] = None,
        ood_params_path: Optional[Path] = None,
        force_ood_refresh: bool = False,
        device: str = 'cuda',
    ):
        self.policy = policy
        self.device = device
        
        # Validate input - need either experiment path or hub repo_id
        if not sae_experiment_path and not sae_hub_repo_id:
            raise ValueError("Must provide either sae_experiment_path or sae_hub_repo_id")
        if sae_experiment_path and sae_hub_repo_id:
            raise ValueError("Cannot provide both sae_experiment_path and sae_hub_repo_id")
        
        # Load SAE model and config
        self.sae_config = None
        self.sae_source = None  # 'local' or 'hub'
        self.sae_hub_repo_id = sae_hub_repo_id
        
        builder = SAEBuilder(device=device)
        
        if sae_hub_repo_id:
            # Load from Hugging Face Hub
            logging.info(f"Loading SAE model from Hub: {sae_hub_repo_id}")
            self.sae_model = builder.load_from_hub(
                repo_id=sae_hub_repo_id,
            )
            self.sae_source = 'hub'
            
            # Try to download config from Hub
            try:
                config_file = hf_hub_download(
                    repo_id=sae_hub_repo_id,
                    filename="config.json",
                )
                with open(config_file, 'r') as f:
                    self.sae_config = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load config from Hub: {e}")
                
        else:
            # Load from local experiment path
            logging.info(f"Loading SAE model from experiment: {sae_experiment_path}")
            self.sae_model = builder.load_from_experiment(sae_experiment_path)
            self.sae_source = 'local'
            
            # Load config if exists
            config_path = Path(sae_experiment_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.sae_config = json.load(f)
        
        self.layer_name = self._infer_layer_name_from_policy()
        
        # Set SAE to eval mode
        self.sae_model.eval()
        
        # Initialize token sampler from SAE config
        self.token_sampler = None
        if self.sae_config is not None:
            # Create token sampler config from SAE config
            if self.sae_config.get('use_token_sampling', False):
                sampler_config = TokenSamplerConfig(
                    fixed_tokens=self.sae_config.get('fixed_tokens', [0, 1]),
                    sampling_strategy=self.sae_config.get('sampling_strategy', 'block_average'),
                    sampling_stride=self.sae_config.get('sampling_stride', 8),
                    max_sampled_tokens=self.sae_config.get('max_sampled_tokens', 100),
                    block_size=self.sae_config.get('block_size', 8)
                )
                # Infer total_tokens from the policy model (same as SAE training)
                from physical_ai_interpretability.sae.config import SAETrainingConfig
                temp_config = SAETrainingConfig()
                total_tokens = temp_config._infer_original_num_tokens(self.policy)
                if total_tokens is None:
                    total_tokens = 602  # Fallback default
                    logging.warning("Could not infer token count from model, using default 602")
                else:
                    logging.info(f"Inferred {total_tokens} tokens from policy model for OOD detection")
                
                self.token_sampler = TokenSampler(sampler_config, total_tokens=total_tokens)
                logging.info(f"Initialized token sampler for OOD detection: {sampler_config.sampling_strategy}")
            else:
                logging.info("Token sampling disabled for OOD detection")
        else:
            logging.warning("No SAE config available, token sampling disabled")
        
        # OOD distribution parameters
        self.ood_params = None
        self.ood_params_path = ood_params_path
        self.force_ood_refresh = force_ood_refresh
        
        # Handle existing OOD parameters based on force_ood_refresh flag
        if force_ood_refresh:
            logging.info("Force refresh requested - will not load existing OOD params")
        else:
            # Try to load OOD parameters from various sources
            loaded = False
            
            # 1. Try local path if provided
            if ood_params_path is not None and Path(ood_params_path).exists():
                logging.info(f"Loading existing OOD parameters from {ood_params_path}")
                self._load_ood_params()
                loaded = True
            
            # 2. Try downloading from Hub if SAE is from Hub and no local params found
            elif self.sae_source == 'hub' and not loaded:
                try:
                    ood_params_file = hf_hub_download(
                        repo_id=sae_hub_repo_id,
                        filename="ood_params.json",
                    )
                    with open(ood_params_file, 'r') as f:
                        self.ood_params = json.load(f)
                    logging.info(f"Loaded OOD parameters from Hub: {sae_hub_repo_id}")
                    loaded = True
                except Exception as e:
                    logging.info(f"Could not load OOD params from Hub: {e}")
            
            if not loaded:
                if ood_params_path is not None:
                    logging.info(f"OOD params path specified but file doesn't exist: {ood_params_path}")
                else:
                    logging.info("No OOD params found - will need to fit threshold")
        
        # Hook for activation extraction
        self._hook = None
        self._register_activation_hook()
    
    def _register_activation_hook(self):
        """Register forward hook to capture activations from the specified layer"""
        
        def hook_fn(module, input, output):
            # Store the activations for later use
            if isinstance(output, tuple):
                self._captured_activations = output[0].clone()  # Clone to avoid memory issues
            else:
                self._captured_activations = output.clone()
            
            logging.debug(f"Captured activations shape: {self._captured_activations.shape}")
        
        # Get layer by name
        layer = self.policy
        for attr in self.layer_name.split('.'):
            layer = getattr(layer, attr)
        
        # Verify the layer exists and is the right type
        logging.info(f"Target layer: {layer} (type: {type(layer)})")
        
        self._hook = layer.register_forward_hook(hook_fn)
        logging.info(f"Registered activation hook on layer: {self.layer_name}")
        
        # Verify hook was registered
        if hasattr(layer, '_forward_hooks') and len(layer._forward_hooks) > 0:
            logging.info(f"Hook successfully registered. Layer has {len(layer._forward_hooks)} forward hooks.")
        else:
            logging.warning("Hook registration may have failed - no forward hooks found on layer")
    
    def _remove_hook(self):
        """Remove the forward hook"""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
    
    def __del__(self):
        """Cleanup hook when object is destroyed"""
        self._remove_hook()
    
    def _infer_layer_name_from_policy(self) -> str:
        """
        Infer the layer name from the policy structure, using the same logic as SAE trainer.
        Returns the last encoder layer's norm2 by default.
        """
        # Default layer name
        default_layer = "model.encoder.layers.3.norm2"
        
        if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'encoder'):
            if hasattr(self.policy.model.encoder, 'layers') and len(self.policy.model.encoder.layers) > 0:
                # Use the last layer's norm2 by default
                layer_idx = len(self.policy.model.encoder.layers) - 1
                inferred_layer = f"model.encoder.layers.{layer_idx}.norm2"
                logging.info(f"Inferred layer name from policy structure: {inferred_layer}")
                return inferred_layer
        
        logging.info(f"Could not infer layer from policy structure, using default: {default_layer}")
        return default_layer

    def fit_ood_threshold_to_validation_dataset(
        self,
        dataset: LeRobotDataset,
        std_threshold: float = 2.5,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Calibrate the out-of-duistribution detector on an unseen validation dataset.
        
        This method runs the OOD Detector for each frame in the dataset, and fits the results on a Gaussian distribution.
        Anything above the specified standard deviation threshold (defaults to σ=2.5) is deemed out of distribution.
        While the default value will work for many datasets we recommend tuning it with a value that works best for your own datasets.

        Fit the OOD threshold to the validation dataset using Gaussian distribution fitting.
        
        Args:
            dataset: Validation dataset to fit on
            std_threshold: Number of standard deviations from mean to use as threshold
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None for all)
            save_path: Path to save OOD parameters (if None, uses self.ood_params_path)
            
        Returns:
            Dictionary with fitted parameters
        """
        logging.info("Fitting OOD threshold using validation dataset...")
        
        # Collect reconstruction errors from validation dataset
        reconstruction_errors = []
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        num_processed = 0
        max_batches = (max_samples // batch_size) if max_samples else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing validation data")):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Prepare batch for policy - convert to observation format
                batch_observations = self._prepare_batch_for_policy(batch)
                
                for obs in batch_observations:
                    # Get reconstruction error for this observation
                    recon_error = self.get_reconstruction_error(obs)
                    reconstruction_errors.append(recon_error)
                    num_processed += 1
                    
                    if max_samples and num_processed >= max_samples:
                        break
                
                if max_samples and num_processed >= max_samples:
                    break
        
        if len(reconstruction_errors) == 0:
            raise RuntimeError("No reconstruction errors were collected. Check dataset and model compatibility.")
        
        # Convert to numpy array for analysis
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Fit Gaussian distribution
        mean = float(np.mean(reconstruction_errors))
        std = float(np.std(reconstruction_errors))
        
        # Calculate threshold
        threshold = mean + std_threshold * std
        
        # Additional statistics
        percentiles = np.percentile(reconstruction_errors, [50, 90, 95, 99, 99.5, 99.9])
        
        # Store parameters
        self.ood_params = {
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'std_threshold': std_threshold,
            'num_samples': len(reconstruction_errors),
            'percentiles': {
                '50': float(percentiles[0]),
                '90': float(percentiles[1]),
                '95': float(percentiles[2]),
                '99': float(percentiles[3]),
                '99.5': float(percentiles[4]),
                '99.9': float(percentiles[5]),
            },
            'min': float(np.min(reconstruction_errors)),
            'max': float(np.max(reconstruction_errors)),
        }
        
        # Save parameters locally
        save_path = save_path or self.ood_params_path
        if save_path:
            self._save_ood_params(save_path)
            logging.info(f"OOD parameters {'refreshed and ' if self.force_ood_refresh else ''}saved to {save_path}")
        
        # Upload to Hub if the SAE model came from Hub
        if self.sae_source == 'hub':
            try:
                self._upload_ood_params_to_hub()
                logging.info(f"OOD parameters uploaded to Hub: {self.sae_hub_repo_id}")
            except Exception as e:
                logging.warning(f"Failed to upload OOD parameters to Hub: {e}")
        
        logging.info(f"OOD threshold fitted:")
        logging.info(f"  Mean: {mean:.6f}")
        logging.info(f"  Std: {std:.6f}")
        logging.info(f"  Threshold ({std_threshold}σ): {threshold:.6f}")
        logging.info(f"  Samples processed: {len(reconstruction_errors)}")
        
        return self.ood_params

    def is_out_of_distribution(self, observation: dict) -> Tuple[bool, float]:
        """
        Detect if the observation is OOD using the SAE model.
        
        Args:
            observation: Input observation dictionary
            
        Returns:
            Tuple of (is_ood, reconstruction_error)
        """
        if self.ood_params is None:
            raise RuntimeError("OOD parameters not fitted. Call fit_ood_threshold_to_validation_dataset first.")
        
        reconstruction_error = self.get_reconstruction_error(observation)
        threshold = self.ood_params['threshold']
        
        is_ood = reconstruction_error > threshold
        
        return is_ood, reconstruction_error

    def get_reconstruction_error(self, observation: dict) -> float:
        """
        Get the reconstruction error of the observation using the SAE model.
        
        Args:
            observation: Input observation dictionary (same format as policy input)
            
        Returns:
            Reconstruction error (MSE loss)
        """
        with torch.inference_mode():
            # Reset captured activations
            self._captured_activations = None
            
            # Run policy forward pass to capture activations
            # We don't need the actual output, just the activations
            _ = self.policy.select_action(observation)
            self.policy.reset()
                        
            # Get activations and prepare for SAE
            self._captured_activations = self._captured_activations.detach()
            activations = self._captured_activations.permute(1, 0, 2).contiguous()  # flip batch size and tokens_length dims
            
            # Apply token sampling if configured (same as SAE training)
            if self.token_sampler is not None:
                original_shape = activations.shape
                activations = self.token_sampler.sample_tokens(activations)
                logging.debug(f"Token sampling: {original_shape} -> {activations.shape}")
            else:
                logging.debug(f"No token sampling applied, activations shape: {activations.shape}")
            
            # Handle batch dimension - we expect single sample
            if activations.dim() == 3 and activations.shape[0] == 1:
                activations = activations.squeeze(0)  # Remove batch dim
            elif activations.dim() == 3:
                # Multiple samples in batch - take first one
                activations = activations[0]
            
            # Ensure activations are in the right format for SAE
            # SAE expects (num_tokens, token_dim) 
            if activations.dim() != 2:
                raise RuntimeError(f"Expected 2D activations, got shape {activations.shape}")
            
            # Add batch dimension for SAE
            activations_batch = activations.unsqueeze(0).to(self.device)
            
            # Get reconstruction from SAE
            reconstruction, features = self.sae_model(activations_batch)
            
            # Calculate reconstruction error (MSE)
            mse_loss = torch.nn.functional.mse_loss(
                reconstruction.squeeze(0), 
                activations_batch.squeeze(0), 
                reduction='mean'
            )
            
            return float(mse_loss.item())
    
    def _prepare_batch_for_policy(self, batch: dict) -> List[dict]:
        """
        Convert dataset batch to list of policy observation dictionaries.
        
        Args:
            batch: Batch from dataset
            
        Returns:
            List of observation dictionaries
        """
        batch_size = None
        observations = []
        
        # Determine batch size from first tensor
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch_size = value.shape[0]
                break
        
        if batch_size is None:
            raise ValueError("Could not determine batch size from batch data")
        
        # Convert batch to list of individual observations
        for i in range(batch_size):
            obs = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    # Extract single sample and add batch dimension
                    obs[key] = value[i:i+1].to(self.device)
                else:
                    # Handle non-tensor data
                    if hasattr(value, '__getitem__'):
                        obs[key] = value[i]
                    else:
                        obs[key] = value
            
            observations.append(obs)
        
        return observations
    
    def _load_ood_params(self):
        """Load OOD parameters from file"""
        if self.ood_params_path and Path(self.ood_params_path).exists():
            with open(self.ood_params_path, 'r') as f:
                self.ood_params = json.load(f)
            logging.info(f"Loaded OOD parameters from {self.ood_params_path}")
            logging.info(f"  Threshold: {self.ood_params.get('threshold', 'N/A')}")
    
    def _save_ood_params(self, save_path: str):
        """Save OOD parameters to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.ood_params, f, indent=2)
        
        logging.info(f"Saved OOD parameters to {save_path}")
    
    def _upload_ood_params_to_hub(self):
        """Upload OOD parameters to Hugging Face Hub"""
        if not self.sae_hub_repo_id:
            raise ValueError("No Hub repo ID available for upload")
        
        if not self.ood_params:
            raise ValueError("No OOD parameters to upload")
        
        # Create temporary file with OOD parameters
        from tempfile import NamedTemporaryFile
        import os
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.ood_params, f, indent=2)
            temp_path = f.name
        
        try:
            # Upload to Hub
            api = HfApi()
            api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="ood_params.json",
                repo_id=self.sae_hub_repo_id,
                commit_message="Update OOD parameters"
            )
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def get_ood_stats(self) -> Optional[Dict[str, float]]:
        """Get current OOD parameters and statistics"""
        return self.ood_params.copy() if self.ood_params else None
    
    def needs_ood_fitting(self) -> bool:
        """Check if OOD threshold needs to be fitted"""
        return self.ood_params is None or self.force_ood_refresh


def create_default_ood_params_path(experiment_name: str, base_dir: str = "output") -> str:
    """
    Create standard path for OOD parameters based on experiment name.
    
    Args:
        experiment_name: SAE experiment name
        base_dir: Base output directory
        
    Returns:
        Path string for OOD parameters file
    """
    return str(Path(base_dir) / experiment_name / "ood_params.json")