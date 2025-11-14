from dataclasses import dataclass
import logging
from typing import Dict, List, Any

import torch

@dataclass
class TokenSamplerConfig:
    fixed_tokens: List[int] = None  # Always include these token indices (e.g., [0, 1] for VAE + proprio)
    sampling_strategy: str = "block_average"  # "uniform", "stride", "random_fixed", "block_average"
    sampling_stride: int = 8  # Take every 8th token when using stride
    max_sampled_tokens: int = 100  # Maximum number of sampled tokens
    random_seed: int = 42  # For reproducible random sampling
    block_size: int = 8  # Size of blocks for block_average strategy

    def __post_init__(self):
        """Initialize fixed_tokens if not provided"""
        if self.fixed_tokens is None:
            self.fixed_tokens = [0, 1]  # Default: VAE latent + proprioception

class TokenSampler:
    """
    Handles consistent token sampling strategies
    """
    
    def __init__(self, config: TokenSamplerConfig, total_tokens: int):
        self.config = config
        self.total_tokens = total_tokens
        self.sampled_indices = None
        
        if config is not None:
            self.sampled_indices = self._generate_sampling_indices()
            logging.info(f"Token sampling enabled: {len(self.sampled_indices)} tokens selected from {total_tokens}")
            logging.info(f"Sampled token indices: {self.sampled_indices}")
    
    def _generate_sampling_indices(self) -> List[int]:
        """Generate consistent token sampling indices"""
        indices = set(self.config.fixed_tokens)  # Always include fixed tokens
        
        # Calculate remaining tokens to sample
        remaining_budget = self.config.max_sampled_tokens - len(indices)
        
        if self.config.sampling_strategy == "uniform":
            # Uniform sampling across the remaining token space
            available_tokens = [i for i in range(self.total_tokens) if i not in indices]
            if remaining_budget > 0 and len(available_tokens) > 0:
                step = max(1, len(available_tokens) // remaining_budget)
                sampled = available_tokens[::step][:remaining_budget]
                indices.update(sampled)
        
        elif self.config.sampling_strategy == "stride":
            # Stride-based sampling (every Nth token)
            stride = self.config.sampling_stride
            for i in range(0, self.total_tokens, stride):
                if i not in indices and len(indices) < self.config.max_sampled_tokens:
                    indices.add(i)
        
        elif self.config.sampling_strategy == "random_fixed":
            # Random sampling with fixed seed for reproducibility
            import random
            random.seed(self.config.random_seed)
            available_tokens = [i for i in range(self.total_tokens) if i not in indices]
            if remaining_budget > 0:
                sampled = random.sample(available_tokens, min(remaining_budget, len(available_tokens)))
                indices.update(sampled)
        
        elif self.config.sampling_strategy == "block_average":
            # Block-average sampling: return fixed tokens + block indices for averaging
            # We'll handle the actual averaging in sample_tokens method
            # For now, just return fixed tokens - the averaging logic is handled differently
            pass  # indices already contains fixed_tokens
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")
        
        # Convert to sorted list for consistent ordering
        return sorted(list(indices))
    
    def sample_tokens(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Sample tokens from activation tensor
        
        Args:
            activations: Tensor of shape (batch_size, num_tokens, token_dim)
            
        Returns:
            Sampled activations of shape (batch_size, num_sampled_tokens, token_dim)
        """
        if self.config is None:
            return activations
        
        if self.config.sampling_strategy == "block_average":
            # Block averaging strategy: preserve fixed tokens, average remaining tokens in blocks
            batch_size, num_tokens, token_dim = activations.shape
            
            # Start with fixed tokens
            fixed_tokens_tensor = activations[:, self.config.fixed_tokens, :]  # (batch_size, num_fixed, token_dim)
            
            # Find the maximum fixed token index to know where to start averaging
            max_fixed_idx = max(self.config.fixed_tokens)
            
            # Collect averaged blocks from remaining tokens
            averaged_blocks = []
            start_idx = max_fixed_idx + 1  # Start after the last fixed token
            
            while start_idx < num_tokens:
                end_idx = min(start_idx + self.config.block_size, num_tokens)
                
                # Extract block and average across the token dimension
                block = activations[:, start_idx:end_idx, :]  # (batch_size, block_size, token_dim)
                averaged_block = torch.mean(block, dim=1, keepdim=True)  # (batch_size, 1, token_dim)
                averaged_blocks.append(averaged_block)
                
                start_idx = end_idx
            
            # Concatenate fixed tokens with averaged blocks
            if averaged_blocks:
                averaged_tensor = torch.cat(averaged_blocks, dim=1)  # (batch_size, num_blocks, token_dim)
                result = torch.cat([fixed_tokens_tensor, averaged_tensor], dim=1)  # (batch_size, total_sampled, token_dim)
            else:
                result = fixed_tokens_tensor
            
            return result
        
        else:
            # Original sampling strategies (uniform, stride, random_fixed)
            if self.sampled_indices is None:
                return activations
            
            # Sample the specified token indices
            sampled_activations = activations[:, self.sampled_indices, :]
            return sampled_activations
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get information about the sampling configuration"""
        if self.config is None:
            return {
                'use_token_sampling': False,
            }

        if self.config.sampling_strategy == "block_average":
            # Calculate how many tokens we'll have after block averaging
            max_fixed_idx = max(self.config.fixed_tokens) if self.config.fixed_tokens else -1
            remaining_tokens = self.total_tokens - (max_fixed_idx + 1)
            num_blocks = (remaining_tokens + self.config.block_size - 1) // self.config.block_size  # Ceiling division
            total_output_tokens = len(self.config.fixed_tokens) + num_blocks

            return {
                'use_token_sampling': True,
                'sampling_strategy': self.config.sampling_strategy,
                'fixed_tokens': self.config.fixed_tokens,
                'block_size': self.config.block_size,
                'original_tokens': self.total_tokens,
                'num_fixed_tokens': len(self.config.fixed_tokens),
                'num_blocks': num_blocks,
                'num_sampled_tokens': total_output_tokens,
                'compression_ratio': self.total_tokens / total_output_tokens if total_output_tokens > 0 else 1.0,
                'sampled_indices': None  # Not applicable for block averaging
            }
        else:
            # Original sampling strategies
            return {
                'use_token_sampling': True,
                'sampled_indices': self.sampled_indices,
                'num_sampled_tokens': len(self.sampled_indices) if self.sampled_indices else None,
                'original_tokens': self.total_tokens,
                'compression_ratio': self.total_tokens / len(self.sampled_indices) if self.sampled_indices else 1.0,
                'sampling_strategy': self.config.sampling_strategy,
                'fixed_tokens': self.config.fixed_tokens
            }
