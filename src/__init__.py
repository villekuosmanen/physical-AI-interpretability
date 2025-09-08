"""
Physical AI Attention Mapper

A package for interpretability analysis of physical AI models, including:
- Sparse Autoencoders (SAE) for feature extraction
- Attention mapping for transformer-based policies
- Token sampling and activation collection utilities
"""

__version__ = "0.1.0"

# Main exports - Public API
from .sae import (
    SAETrainer,
    SAETrainingConfig,
    load_sae_model,
    TokenSampler,
    TokenSamplerConfig,
    MultiModalSAE,
    SAEBuilder,
    load_sae_model_simple,
)
from .attention_maps import ACTPolicyWithAttention
from .ood import OODDetector, create_default_ood_params_path

# Utility functions (consider if these should be public)
from .utils import make_dataset_without_config

__all__ = [
    # SAE Training and Models
    "SAETrainer",
    "SAETrainingConfig", 
    "load_sae_model",
    "SAEBuilder",
    "load_sae_model_simple",
    # SAE Components
    "MultiModalSAE", 
    "TokenSampler",
    "TokenSamplerConfig",
    # OOD Detection
    "OODDetector",
    "create_default_ood_params_path", 
    # Attention Mapping
    "ACTPolicyWithAttention",
    # Dataset Utilities
    "make_dataset_without_config",
]
