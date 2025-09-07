"""
Physical AI Attention Mapper

A package for interpretability analysis of physical AI models, including:
- Sparse Autoencoders (SAE) for feature extraction
- Attention mapping for transformer-based policies
- Token sampling and activation collection utilities
"""

__version__ = "0.1.0"

# Main exports
from .sae import SAETrainer, SAETrainingConfig, load_sae_model, TokenSampler, TokenSamplerConfig, MultiModalSAE, create_multimodal_sae, prepare_batch_for_bfloat16, SAEBuilder, load_sae_model_simple
from .attention_maps import ACTPolicyWithAttention
from .utils import make_dataset_without_config, get_repo_hash
from .ood import OODDetector, create_default_ood_params_path

__all__ = [
    # SAE components
    "SAETrainer",
    "SAETrainingConfig",
    "load_sae_model",
    "SAEBuilder",
    "load_sae_model_simple",
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
    # OOD Detection
    "OODDetector",
    "create_default_ood_params_path", 
    # Attention mapping
    "ACTPolicyWithAttention",
    # Utilities
    "make_dataset_without_config",
    "get_repo_hash",
]
