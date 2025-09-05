"""
Physical AI Attention Mapper

A package for interpretability analysis of physical AI models, including:
- Sparse Autoencoders (SAE) for feature extraction
- Attention mapping for transformer-based policies
- Token sampling and activation collection utilities
"""

__version__ = "0.1.0"

# Main exports
from .sae import SAETrainer, SAETrainingConfig, load_sae_model
from .feature_extraction import (
    MultiModalSAE,
    create_multimodal_sae,
    TokenSampler,
    TokenSamplerConfig,
    ActivationCollector,
    collect_and_cache_activations,
    create_cached_dataloader,
    is_cache_valid,
    cleanup_invalid_cache,
    get_cache_status,
)
from .attention_maps import ACTPolicyWithAttention
from .utils import make_dataset_without_config, get_repo_hash

__all__ = [
    # SAE components
    "SAETrainer",
    "SAETrainingConfig",
    "load_sae_model",
    # Feature extraction
    "MultiModalSAE", 
    "create_multimodal_sae",
    "TokenSampler",
    "TokenSamplerConfig",
    "ActivationCollector",
    "collect_and_cache_activations",
    "create_cached_dataloader",
    "is_cache_valid",
    "cleanup_invalid_cache",
    "get_cache_status",
    # Attention mapping
    "ACTPolicyWithAttention",
    # Utilities
    "make_dataset_without_config",
    "get_repo_hash",
]
