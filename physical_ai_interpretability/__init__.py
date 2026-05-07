"""
Physical AI Attention Mapper

A package for interpretability analysis of physical AI models, including:
- Sparse Autoencoders (SAE) for feature extraction
- Attention mapping for transformer-based policies
- Token sampling and activation collection utilities
"""

__version__ = "0.1.4"

from .attention_maps import ACTPolicyWithAttention

try:
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
    _SAE_AVAILABLE = True
except ImportError:
    _SAE_AVAILABLE = False

from .ood import OODDetector, create_default_ood_params_path

# Utility functions (consider if these should be public)
from .utils import make_dataset_without_config

__all__ = [
    # OOD Detection
    "OODDetector",
    "create_default_ood_params_path",
    # Attention Mapping
    "ACTPolicyWithAttention",
    # Dataset Utilities
    "make_dataset_without_config",
]

if _SAE_AVAILABLE:
    __all__ += [
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
    ]
