#!/usr/bin/env python

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from .sae import MultiModalSAE, create_multimodal_sae


class SAEBuilder:
    """
    Builder class for loading SAE models with their configurations.
    Provides convenient methods to load trained SAE models from standard paths.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def load_from_experiment(
        self, 
        experiment_path: str,
        checkpoint: str = 'latest',
        config_filename: str = 'config.json'
    ) -> MultiModalSAE:
        """
        Load SAE model from experiment directory.
        
        Args:
            experiment_path: Path to experiment directory (e.g., "output/sae_drop_footbag_into_di_838a8c8b")
            checkpoint: Which checkpoint to load - 'best', 'latest', or specific epoch number
            config_filename: Name of config file
            
        Returns:
            Loaded SAE model
        """
        experiment_path = Path(experiment_path)
        config_path = experiment_path / config_filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Determine model path based on checkpoint specification
        if checkpoint == 'best':
            model_path = experiment_path / "best_model.safetensors"
        elif checkpoint == 'latest':
            # Find latest epoch checkpoint
            model_files = list(experiment_path.glob("model_epoch_*.safetensors"))
            if not model_files:
                raise FileNotFoundError(f"No model checkpoints found in {experiment_path}")
            latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
            model_path = latest_model
        else:
            # Specific epoch or filename
            try:
                epoch_num = int(checkpoint)
                model_path = experiment_path / f"model_epoch_{epoch_num}.safetensors"
            except ValueError:
                # Treat as filename
                model_path = experiment_path / checkpoint
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        return self.load_from_files(str(model_path), str(config_path))
    
    def load_from_files(
        self, 
        model_path: str, 
        config_path: str
    ) -> MultiModalSAE:
        """
        Load SAE model from specific model and config files.
        
        Args:
            model_path: Path to safetensors model file
            config_path: Path to config.json file
            
        Returns:
            Loaded SAE model
        """
        model_path = Path(model_path)
        config_path = Path(config_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Extract model parameters
        num_tokens = config_dict.get('num_tokens')
        token_dim = config_dict.get('token_dim')
        feature_dim = config_dict.get('feature_dim')
        activation_fn = config_dict.get('activation_fn', 'relu')
        use_bfloat16 = config_dict.get('use_bfloat16', False)
        
        if any(param is None for param in [num_tokens, token_dim]):
            raise ValueError("Config file missing required parameters: num_tokens, token_dim")
            
        # Calculate feature_dim if not present
        if feature_dim is None:
            expansion_factor = config_dict.get('expansion_factor', 1.25)
            feature_dim = int(num_tokens * token_dim * expansion_factor)
        
        # Create model
        model = create_multimodal_sae(
            num_tokens=num_tokens,
            token_dim=token_dim,
            feature_dim=feature_dim,
            device=self.device,
            use_bfloat16=use_bfloat16
        )
        
        # Load weights from safetensors
        model_state = load_file(model_path)
        model.load_state_dict(model_state)
        model.eval()
        
        logging.info(f"Loaded SAE model from {model_path}")
        logging.info(f"Model config: {num_tokens} tokens, {token_dim} dim, {feature_dim} features")
        
        return model
    
    def load_with_auto_config(
        self, 
        model_path: str, 
        config_path: Optional[str] = None
    ) -> MultiModalSAE:
        """
        Load SAE model with automatic config discovery.
        
        Args:
            model_path: Path to safetensors model file
            config_path: Optional path to config.json. If None, searches automatically
            
        Returns:
            Loaded SAE model
        """
        model_path = Path(model_path)
        
        # Auto-discover config if not provided
        if config_path is None:
            potential_configs = [
                model_path.parent / "config.json",
                model_path.parent.parent / "config.json",
            ]
            
            for potential_config in potential_configs:
                if potential_config.exists():
                    config_path = potential_config
                    break
            
            if config_path is None:
                raise FileNotFoundError("Could not find config.json file. Please provide config_path explicitly.")
        
        return self.load_from_files(str(model_path), str(config_path))

    def load_from_hub(
        self, 
        repo_id: str,
        filename: str = "model.safetensors",
        config_filename: str = "config.json",
        revision: str = "main",
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        token: Optional[str] = None
    ) -> MultiModalSAE:
        """
        Load SAE model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            filename: Model filename to download
            config_filename: Config filename to download  
            revision: Git revision (branch/tag/commit)
            cache_dir: Local cache directory
            force_download: Force re-download even if cached
            token: Hugging Face token for private repos
            
        Returns:
            Loaded SAE model
        """
        # Download model file
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        
        # Download config file
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        
        return self.load_from_files(model_file, config_file)

    @classmethod 
    def from_default_path(
        cls,
        experiment_name: str,
        base_output_dir: str = "output",
        device: str = 'cuda'
    ) -> 'SAEBuilder':
        """
        Create SAEBuilder and load model from default output structure.
        
        Args:
            experiment_name: Name of experiment (e.g., "sae_drop_footbag_into_di_838a8c8b")
            base_output_dir: Base output directory
            device: Device to load model on
            
        Returns:
            SAEBuilder instance with loaded model
        """
        builder = cls(device=device)
        experiment_path = Path(base_output_dir) / experiment_name
        return builder.load_from_experiment(str(experiment_path))


def load_sae_model_simple(
    experiment_path: str,
    checkpoint: str = 'best',
    device: str = 'cuda'
) -> MultiModalSAE:
    """
    Simple convenience function to load SAE model from experiment directory.
    
    Args:
        experiment_path: Path to experiment directory
        checkpoint: Which checkpoint to load
        device: Device to load on
        
    Returns:
        Loaded SAE model
    """
    builder = SAEBuilder(device=device)
    return builder.load_from_experiment(experiment_path, checkpoint)
