from dataclasses import dataclass, field
from typing import Optional
import torch
import logging


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training"""
    # Model config - these will be auto-inferred
    num_tokens: Optional[int] = None  # Auto-inferred from token sampling config
    token_dim: Optional[int] = None   # Auto-inferred from ACT model
    expansion_factor: float = 1    # Feature expansion factor (feature_dim = num_tokens * token_dim * expansion_factor)
    activation_fn: str = 'relu'       # 'tanh', 'relu', 'leaky_relu'
    
    # Token sampling config - affects num_tokens calculation
    use_token_sampling: bool = True
    fixed_tokens: list = field(default_factory=lambda: [0, 1])  # VAE latent + proprioception tokens
    sampling_strategy: str = "block_average"  # "uniform", "stride", "random_fixed", "block_average"
    sampling_stride: int = 8
    max_sampled_tokens: int = 200
    block_size: int = 8
    
    # Training config
    batch_size: int = 128
    learning_rate: float = 1e-4
    num_epochs: int = 20
    validation_split: float = 0.1
    
    # Loss config
    l1_penalty: float = 0.3
    
    # Optimization config
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-5
    lr_schedule: str = 'constant'  # 'cosine', 'linear', 'constant'
    warmup_epochs: int = 2
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-5
    
    # Logging and saving
    log_every: int = 5
    save_every: int = 1000
    validate_every: int = 500
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def feature_dim(self) -> Optional[int]:
        """Calculate feature dimension based on expansion factor"""
        if self.num_tokens is not None and self.token_dim is not None:
            return int(self.num_tokens * self.token_dim * self.expansion_factor)
        return None
    
    def _infer_original_num_tokens(self, policy) -> int:
        """
        Infer the original number of tokens from the ACT policy model.
        
        The total number of tokens in ACT models is calculated as:
        - 2 fixed tokens (VAE latent + proprioception)
        - Plus tokens from each camera image (width/32 × height/32 for each camera)
        
        Args:
            policy: The ACT policy model
            
        Returns:
            Original number of tokens in the model
        """
        original_num_tokens = None
        
        # Method 1: Try to infer from model configuration
        if hasattr(policy, 'config'):
            config = policy.config
            
            # Check if config has image features information
            if hasattr(config, 'image_features') and config.image_features:
                # Calculate tokens from image dimensions
                # ACT models typically have 2 fixed tokens (VAE + proprioception)
                fixed_tokens = 2
                image_tokens = len(config.image_features)
                original_num_tokens = fixed_tokens + (image_tokens * 300)
        
        # Method 3: Use established default for ACT models if nothing else worked
        if original_num_tokens is None:
            # For common ACT setups with 2 cameras at 480x640 resolution
            original_num_tokens = 602  # 2 + 2*((480/32) * (640/32)) = 2 + 2*(15*20) = 602
            logging.warning(f"Using default ACT token count: {original_num_tokens} (configure your model for automatic detection)")
        
        return original_num_tokens

    def infer_model_params_from_cache(self, cache_path: str, token_sampler_config=None):
        """
        Infer model parameters from cached activation data.
        
        Args:
            cache_path: Path to cached activation data
            token_sampler_config: TokenSamplerConfig object for token sampling.
        
        Returns:
            Self for method chaining
        """
        # Load original_num_tokens from cache metadata
        from .activation_collector import load_original_num_tokens_from_cache
        original_num_tokens = load_original_num_tokens_from_cache(cache_path)
        
        if original_num_tokens is None:
            raise ValueError(f"Could not load original_num_tokens from cache at {cache_path}")
        
        # Infer token_dim from activation shape in cache metadata
        import json
        from pathlib import Path
        metadata_file = Path(cache_path) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            activation_shape = metadata.get('activation_shape')
            if activation_shape and len(activation_shape) == 2:
                # activation_shape is [num_tokens, token_dim] after excluding batch dimension
                self.token_dim = activation_shape[1]
                logging.info(f"Inferred token_dim from cache metadata: {self.token_dim}")
        
        if self.token_dim is None:
            raise ValueError("Could not infer token_dim from cache metadata")
        
        # Calculate num_tokens based on token sampling
        if token_sampler_config is not None:
            from physical_ai_interpretability.sae import TokenSampler
            sampler = TokenSampler(token_sampler_config, total_tokens=original_num_tokens)
            sampling_info = sampler.get_sampling_info()
            
            if sampling_info['use_token_sampling']:
                self.num_tokens = sampling_info['num_sampled_tokens']
            else:
                self.num_tokens = original_num_tokens
        else:
            # No token sampling, use the fixed tokens only
            self.num_tokens = len(self.fixed_tokens)
        
        logging.info(f"Inferred model parameters from cache:")
        logging.info(f"  original_num_tokens: {original_num_tokens}")
        logging.info(f"  token_dim: {self.token_dim}")
        logging.info(f"  num_tokens (after sampling): {self.num_tokens}")
        logging.info(f"  feature_dim: {self.feature_dim}")
        logging.info(f"  expansion_factor: {self.expansion_factor}")
        
        return self

    def infer_model_params(self, policy, token_sampler_config=None):
        """
        Infer num_tokens and token_dim from the policy model and token sampling configuration.
        
        Args:
            policy: The ACT policy model
            token_sampler_config: TokenSamplerConfig object for token sampling.

        :meta private:
        """
        # Infer token_dim from the model
        # Typically this is the hidden dimension of the encoder
        if hasattr(policy, 'model') and hasattr(policy.model, 'encoder'):
            if hasattr(policy.model.encoder, 'layers') and len(policy.model.encoder.layers) > 0:
                # Get from the first layer's dimension
                first_layer = policy.model.encoder.layers[0]
                if hasattr(first_layer, 'norm1') and hasattr(first_layer.norm1, 'normalized_shape'):
                    self.token_dim = first_layer.norm1.normalized_shape[0]
                elif hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'embed_dim'):
                    self.token_dim = first_layer.self_attn.embed_dim
                elif hasattr(first_layer, 'norm2') and hasattr(first_layer.norm2, 'normalized_shape'):
                    self.token_dim = first_layer.norm2.normalized_shape[0]
        
        # If we couldn't infer from the model structure, try a different approach
        if self.token_dim is None and hasattr(policy, 'config'):
            if hasattr(policy.config, 'dim_model'):
                self.token_dim = policy.config.dim_model
            elif hasattr(policy.config, 'hidden_size'):
                self.token_dim = policy.config.hidden_size
        
        # Infer original_num_tokens from the model itself
        original_num_tokens = self._infer_original_num_tokens(policy)
                
        # Final fallback to default
        if original_num_tokens is None:
            original_num_tokens = 602
            logging.warning("Using hardcoded default of 602 tokens. Consider configuring your model properly.")
        
        # Infer num_tokens based on token sampling
        if token_sampler_config is not None:
            from physical_ai_interpretability.sae import TokenSampler
            
            sampler = TokenSampler(token_sampler_config, total_tokens=original_num_tokens)
            sampling_info = sampler.get_sampling_info()
            
            if sampling_info['use_token_sampling']:
                self.num_tokens = sampling_info['num_sampled_tokens']
            else:
                self.num_tokens = original_num_tokens
        else:
            # No token sampling, use default
            self.num_tokens = len(self.fixed_tokens)  # Just the fixed tokens
        
        # Validate that we have the required values
        if self.token_dim is None:
            raise ValueError("Could not infer token_dim from the model. Please set it manually.")
        if self.num_tokens is None:
            raise ValueError("Could not infer num_tokens. Please check token sampling configuration.")
            
        logging.info(f"Auto-inferred model parameters:")
        logging.info(f"  original_num_tokens: {original_num_tokens}")
        logging.info(f"  token_dim: {self.token_dim}")
        logging.info(f"  num_tokens (after sampling): {self.num_tokens}")
        logging.info(f"  feature_dim: {self.feature_dim}")
        logging.info(f"  expansion_factor: {self.expansion_factor}")
        
        return self
