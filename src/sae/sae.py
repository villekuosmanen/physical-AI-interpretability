#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiModalSAE(nn.Module):
    """
    Sparse Autoencoder that processes all tokens from ACT model simultaneously.

    num_tokens (int): should match the number of tokens of your ACT model, or the number of sampled tokens when using sampling
    token_dim (int): should match the dim_model hyperparam in your ACT model
    feature_dim (int): the number of features the SAE should learn to represent. Usually this would be (num_tokens * token_dim * expansion_factor)
    
    """
    
    def __init__(
        self,
        num_tokens: int,
        token_dim: int = 512, 
        feature_dim: int = 4096,
        use_bias: bool = True,
        activation_fn: str = 'leaky_relu',
        dropout_rate: float = 0.0,
        device: str = 'cuda',
        use_bfloat16: bool = False,
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.feature_dim = feature_dim
        self.input_dim = num_tokens * token_dim
        self.device = device
        self.use_bfloat16 = use_bfloat16
        
        # Set the default dtype for the model
        self.model_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        
        # Encoder: compress all tokens to feature space
        self.encoder = nn.Linear(self.input_dim, feature_dim, bias=use_bias, dtype=self.model_dtype)
        
        # Decoder: reconstruct all tokens from features
        self.decoder = nn.Linear(feature_dim, self.input_dim, bias=use_bias, dtype=self.model_dtype)
        
        # Activation function - these work naturally with bfloat16
        if activation_fn == 'tanh':
            self.activation = nn.Tanh()
        elif activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self._init_weights()
        
        # Move model to device and convert to bfloat16 if requested
        self.to(device)
        if use_bfloat16:
            self.to(dtype=torch.bfloat16)
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform - works with any dtype"""
        with torch.no_grad():
            # Initialize in float32 first for numerical stability
            nn.init.xavier_uniform_(self.encoder.weight.float())
            nn.init.xavier_uniform_(self.decoder.weight.float())
            
            if self.encoder.bias is not None:
                nn.init.zeros_(self.encoder.bias.float())
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias.float())
            
            # Convert back to target dtype if needed
            if self.use_bfloat16:
                self.encoder.weight.data = self.encoder.weight.data.to(torch.bfloat16)
                self.decoder.weight.data = self.decoder.weight.data.to(torch.bfloat16)
                if self.encoder.bias is not None:
                    self.encoder.bias.data = self.encoder.bias.data.to(torch.bfloat16)
                if self.decoder.bias is not None:
                    self.decoder.bias.data = self.decoder.bias.data.to(torch.bfloat16)
    
    def _ensure_correct_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Helper function to ensure input tensors match model dtype"""
        if self.use_bfloat16 and x.dtype != torch.bfloat16:
            return x.to(dtype=torch.bfloat16)
        elif not self.use_bfloat16 and x.dtype != torch.float32:
            return x.to(dtype=torch.float32)
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode flattened token representation to feature space
        Now handles bfloat16 conversion automatically
        """
        x = self._ensure_correct_dtype(x)
        
        features = self.encoder(x)
        features = self.activation(features)
        features = self.dropout(features)
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to token space
        """
        features = self._ensure_correct_dtype(features)
        reconstruction = self.decoder(features)
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode
        """
        # Handle both shaped and flattened input
        if len(x.shape) == 3:  # (batch_size, num_tokens, token_dim)
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
        else:  # Already flattened
            batch_size = x.shape[0]
            x_flat = x
        
        # Ensure correct dtype
        x_flat = self._ensure_correct_dtype(x_flat)
        
        # Encode to features
        features = self.encode(x_flat)
        
        # Decode back to token space
        reconstruction_flat = self.decode(features)
        
        # Reshape reconstruction to match input
        if len(x.shape) == 3:
            reconstruction = reconstruction_flat.view(batch_size, self.num_tokens, self.token_dim)
        else:
            reconstruction = reconstruction_flat
        
        return reconstruction, features
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        l1_penalty: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with optional regularization
        Note: Loss computation often benefits from float32 precision for stability
        """
        reconstruction, features = self.forward(x)
        
        # Handle input shape for loss computation
        if len(x.shape) == 3:
            x_flat = x.view(x.shape[0], -1)
            reconstruction_flat = reconstruction.view(reconstruction.shape[0], -1)
        else:
            x_flat = x
            reconstruction_flat = reconstruction
        
        x_flat_loss = x_flat
        reconstruction_flat_loss = reconstruction_flat
        features_loss = features
        
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(reconstruction_flat_loss, x_flat_loss, reduction='mean')
        
        # L1 penalty on features (sparsity)
        l1_loss = torch.mean(torch.abs(features_loss)) if l1_penalty > 0 else torch.tensor(0.0, device=x.device)

        # Total loss
        total_loss = (mse_loss + l1_penalty * l1_loss)
        
        # Compute metrics for monitoring (in float32 for accuracy)
        with torch.no_grad():
            feature_mean = features_loss.mean()
            feature_std = features_loss.std()
            feature_sparsity = (torch.abs(features_loss) < 0.1).float().mean()
            
            # Reconstruction quality (R²)
            ss_res = torch.sum((x_flat_loss - reconstruction_flat_loss) ** 2)
            ss_tot = torch.sum((x_flat_loss - x_flat_loss.mean()) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'l1_loss': l1_loss * l1_penalty,
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'feature_sparsity': feature_sparsity,
            'r_squared': r_squared
        }

def create_multimodal_sae(
    num_tokens: int = None,
    token_dim: int = 512,
    feature_dim: int = 1024,
    device: str = 'cuda',
    use_bfloat16: bool = False
) -> nn.Module:
    """
    Factory function to SAE models with bfloat16 support.
    
    Args:
        num_tokens: Number of tokens in the model. If None, will raise an error.
        token_dim: Dimension of each token
        feature_dim: Dimension of the feature space
        device: Device to place the model on
        use_bfloat16: Whether to use bfloat16 precision

    :meta private:
    """
    if num_tokens is None:
        raise ValueError(
            "num_tokens must be specified. Use SAETrainingConfig.infer_model_params() "
            "to automatically infer this value from your ACT policy model."
        )
    
    return MultiModalSAE(
        num_tokens=num_tokens,
        token_dim=token_dim,
        feature_dim=feature_dim,
        device=device,
        use_bfloat16=use_bfloat16
    )

# Utility function for converting input data to bfloat16
def prepare_batch_for_bfloat16(batch: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """
    Convert a batch of activations to bfloat16 and move to device
    
    Args:
        batch: Input tensor (usually float32)
        device: Target device
        
    Returns:
        Tensor converted to bfloat16 on target device

    :meta private:
    """
    return batch.to(device=device, dtype=torch.bfloat16)
