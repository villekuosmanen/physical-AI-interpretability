"""
Utilities for generating consistent names and hashes for experiments and caching
"""

import hashlib
from typing import Optional


def get_repo_hash(repo_id: str, length: int = 8) -> str:
    """
    Generate a consistent hash digest from a repo_id for use in file/folder names.
    
    This provides several benefits:
    - Handles arbitrarily long repo IDs
    - Avoids filesystem issues with special characters
    - Provides consistent naming across different uses
    - Short enough for human readability
    - Changes when repo_id changes (cache invalidation)
    
    Args:
        repo_id: The repository identifier (e.g., "huggingface/dataset-name")
        length: Length of the hash digest to return (default: 8)
        
    Returns:
        Hex digest of the repo_id of specified length
        
    Examples:
        >>> get_repo_hash("huggingface/my-dataset")
        'a1b2c3d4'
        >>> get_repo_hash("very/long/repository/name/that/might/cause/issues")
        'e5f6a7b8'

    :meta private:
    """
    if length <= 0:
        raise ValueError("Length must be positive")
    if length > 64:  # SHA256 produces 64 hex characters
        raise ValueError("Length cannot exceed 64 characters")
    
    # Use SHA256 for consistency and good distribution
    hash_object = hashlib.sha256(repo_id.encode('utf-8'))
    hex_digest = hash_object.hexdigest()
    
    return hex_digest[:length]


def get_experiment_name(repo_id: str, prefix: str = "sae", include_repo_hint: bool = True) -> str:
    """
    Generate a consistent experiment name from a repo_id.
    
    Args:
        repo_id: The repository identifier
        prefix: Prefix for the experiment name (default: "sae")
        include_repo_hint: Whether to include a hint about the original repo name
        
    Returns:
        Experiment name suitable for wandb, directory names, etc.
        
    Examples:
        >>> get_experiment_name("huggingface/coffee-task")
        'sae_coffee-task_a1b2c3d4'
        >>> get_experiment_name("very/long/name", include_repo_hint=False)
        'sae_e5f6a7b8'

    :meta private:
    """
    repo_hash = get_repo_hash(repo_id)
    
    if include_repo_hint and repo_id:
        # Extract a clean hint from the repo_id (last part after /)
        repo_hint = repo_id.split('/')[-1]
        # Clean the hint (remove special characters, limit length)
        repo_hint = ''.join(c for c in repo_hint if c.isalnum() or c in '-_')[:20]
        
        if repo_hint:
            return f"{prefix}_{repo_hint}_{repo_hash}"
    
    return f"{prefix}_{repo_hash}"


def get_cache_name(repo_id: str) -> str:
    """
    Generate a consistent cache directory name from a repo_id.
    
    Args:
        repo_id: The repository identifier
        
    Returns:
        Cache directory name
        
    Examples:
        >>> get_cache_name("huggingface/coffee-task")
        'coffee-task_a1b2c3d4'

    :meta private:
    """
    return get_experiment_name(repo_id, prefix="", include_repo_hint=True).lstrip('_')
