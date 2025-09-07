from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training"""
    # Model config - these will be auto-inferred
    num_tokens: Optional[int] = None  # Auto-inferred from token sampling config
    token_dim: Optional[int] = None   # Auto-inferred from ACT model
    expansion_factor: float = 1.25    # Feature expansion factor (feature_dim = num_tokens * token_dim * expansion_factor)
    activation_fn: str = 'relu'       # 'tanh', 'relu', 'leaky_relu'
    
    # Token sampling config - affects num_tokens calculation
    use_token_sampling: bool = True
    fixed_tokens: list = field(default_factory=lambda: [0, 1])  # VAE latent + proprioception tokens
    sampling_strategy: str = "block_average"  # "uniform", "stride", "random_fixed", "block_average"
    sampling_stride: int = 8
    max_sampled_tokens: int = 100
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
    
    def infer_model_params(self, policy, token_sampler_config=None):
        """
        Infer num_tokens and token_dim from the policy model and token sampling configuration
        
        Args:
            policy: The ACT policy model
            token_sampler_config: TokenSamplerConfig object for token sampling
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
        
        # Infer num_tokens based on token sampling
        if token_sampler_config is not None:
            from src.sae import TokenSampler
            
            # Create a dummy sampler to get the number of output tokens
            # We need to know the original number of tokens in the model
            original_num_tokens = 602  # Default for ACT models - TODO: make this configurable
            
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
            
        print(f"Auto-inferred model parameters:")
        print(f"  token_dim: {self.token_dim}")
        print(f"  num_tokens: {self.num_tokens}")
        print(f"  feature_dim: {self.feature_dim}")
        print(f"  expansion_factor: {self.expansion_factor}")
        
        return self
