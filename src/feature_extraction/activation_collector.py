from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any

from safetensors.torch import save_file, load_file
import torch
from tqdm import tqdm

from src.feature_extraction import TokenSamplerConfig, TokenSampler


@dataclass
class ActivationCacheConfig:
    """Configuration for activation caching"""
    # Cache settings
    cache_dir: str = "./output/activation_cache"
    buffer_size: int = 128  # Number of samples per cache file
    
    # Data organization
    layer_name: str = "encoder.layers.4"
    experiment_name: str = "act_activations"
    use_token_sampling: bool = True  # Enable token sampling
    
    # Memory management
    max_memory_gb: float = 4.0  # Maximum memory to use for caching
    cleanup_on_start: bool = True  # Clean old cache files
    
    # Validation
    validate_cache: bool = True  # Validate cached files on load
    cache_metadata: bool = True  # Save metadata with cache

class ActivationCache:
    """
    Manages caching of activations to disk for memory-efficient SAE training
    """
    
    def __init__(
            self,
            config: ActivationCacheConfig,
            sampler_config: TokenSamplerConfig,
        ):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / config.experiment_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sampler = TokenSampler(sampler_config)
        
        # Initialize cache state
        self.current_buffer = []
        self.buffer_count = 0
        self.total_samples = 0
        self.cache_files = []
        
        # Metadata
        self.metadata = {
            'layer_name': config.layer_name,
            'experiment_name': config.experiment_name,
            'created_at': time.time(),
            'buffer_size': config.buffer_size,
            'cache_files': [],
            'total_samples': 0,
            'activation_shape': None
        }
        
        # Cleanup old cache if requested
        if config.cleanup_on_start:
            self._cleanup_cache()
        
        logging.info(f"Initialized activation cache at {self.cache_dir}")
    
    def _cleanup_cache(self):
        """Remove old cache files"""
        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                if file.suffix in ['.safetensors', '.pt', '.json']:
                    file.unlink()
            logging.info("Cleaned up old cache files")
    
    def add_activations(self, activations: torch.Tensor, batch_metadata: Optional[Dict[str, Any]] = None):
        """
        Add activations to cache buffer
        
        Args:
            activations: Tensor of shape (batch_size, num_tokens, token_dim)
            batch_metadata: Optional metadata for this batch
        """
        # Ensure activations are on CPU and detached
        if activations.requires_grad:
            activations = activations.detach()
        if activations.device != torch.device('cpu'):
            activations = activations.cpu()
        
        # Store activation shape for metadata
        if self.metadata['activation_shape'] is None:
            self.metadata['activation_shape'] = list(activations.shape[1:])  # Exclude batch dimension
        
        # Add to buffer
        activations = activations.permute(1, 0, 2).contiguous() # flip batch size and tokens_length dims
        activations = self.sampler.sample_tokens(activations)
        batch_size = activations.shape[0]
        for i in range(batch_size):
            sample = {
                'activation': activations[i],  # Shape: (num_tokens, token_dim)
                'sample_idx': self.total_samples,
                'buffer_idx': len(self.current_buffer)
            }
            
            # Add metadata if provided
            if batch_metadata:
                sample['metadata'] = self._extract_sample_metadata(batch_metadata, i)
            
            self.current_buffer.append(sample)
            self.total_samples += 1
            
            # Flush buffer if full
            if len(self.current_buffer) >= self.config.buffer_size:
                self._flush_buffer()
    
    def _extract_sample_metadata(self, batch_metadata: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """Extract metadata for a single sample from batch metadata"""
        sample_metadata = {}
        
        for key, value in batch_metadata.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) > 0 and value.shape[0] > sample_idx:
                    # Store only essential metadata to save space
                    if key in ['episode_idx', 'frame_idx', 'action']:
                        sample_metadata[key] = value[sample_idx].tolist() if hasattr(value[sample_idx], 'tolist') else value[sample_idx]
            elif isinstance(value, (list, tuple)) and len(value) > sample_idx:
                sample_metadata[key] = value[sample_idx]
        
        return sample_metadata
    
    def _flush_buffer(self):
        """Save current buffer to disk"""
        if not self.current_buffer:
            return
        
        # Create filename
        filename = f"activations_buffer_{self.buffer_count:06d}"
        
        filepath = self.cache_dir / f"{filename}.safetensors"
        self._save_buffer_safetensors(filepath)
        
        # Update metadata
        self.cache_files.append(str(filepath))
        self.metadata['cache_files'].append({
            'filename': filepath.name,
            'buffer_idx': self.buffer_count,
            'num_samples': len(self.current_buffer),
            'sample_range': (
                self.current_buffer[0]['sample_idx'],
                self.current_buffer[-1]['sample_idx']
            )
        })
        
        logging.info(f"Saved buffer {self.buffer_count} with {len(self.current_buffer)} samples to {filepath.name}")
        
        # Clear buffer
        self.current_buffer = []
        self.buffer_count += 1
    
    def _save_buffer_safetensors(self, filepath: Path):
        """Save buffer using safetensors format"""
        # Prepare tensors for safetensors (flat structure required)
        tensors = {}
        metadata_list = []
        
        for i, sample in enumerate(self.current_buffer):
            # Store activation tensor
            tensors[f"activation_{i}"] = sample['activation']
            
            # Store metadata separately (safetensors doesn't support nested structures)
            sample_metadata = {
                'sample_idx': sample['sample_idx'],
                'buffer_idx': sample['buffer_idx']
            }
            if 'metadata' in sample:
                sample_metadata.update(sample['metadata'])
            
            metadata_list.append(sample_metadata)
        
        # Save tensors
        save_file(tensors, filepath)
        
        # Save metadata separately
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f)
    
    def _save_buffer_torch(self, filepath: Path):
        """Save buffer using PyTorch format"""
        # Stack activations into single tensor
        activations = torch.stack([sample['activation'] for sample in self.current_buffer])
        
        # Prepare metadata
        metadata_list = []
        for sample in self.current_buffer:
            sample_metadata = {
                'sample_idx': sample['sample_idx'],
                'buffer_idx': sample['buffer_idx']
            }
            if 'metadata' in sample:
                sample_metadata.update(sample['metadata'])
            metadata_list.append(sample_metadata)
        
        # Save everything
        torch.save({
            'activations': activations,
            'metadata': metadata_list,
            'buffer_info': {
                'buffer_idx': self.buffer_count,
                'num_samples': len(self.current_buffer)
            }
        }, filepath)
    
    def finalize(self):
        """Flush any remaining activations and save metadata"""
        # Flush remaining buffer
        if self.current_buffer:
            self._flush_buffer()
        
        # Update final metadata
        self.metadata['total_samples'] = self.total_samples
        self.metadata['num_buffers'] = self.buffer_count
        self.metadata['finalized_at'] = time.time()
        
        # Save metadata
        metadata_path = self.cache_dir / "cache_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logging.info(f"Finalized cache with {self.total_samples} samples in {self.buffer_count} buffers")
        
        return {
            'total_samples': self.total_samples,
            'num_buffers': self.buffer_count,
            'cache_dir': str(self.cache_dir),
            'cache_files': self.cache_files
        }


class CachedActivationDataset(torch.utils.data.Dataset):
    """
    Dataset that loads activations from cached files
    """
    
    def __init__(self, cache_dir: str, shuffle: bool = True, preload_buffers: int = 2):
        self.cache_dir = Path(cache_dir)
        self.shuffle = shuffle
        self.preload_buffers = preload_buffers
        
        # Load metadata
        metadata_path = self.cache_dir / "cache_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cache metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_samples = self.metadata['total_samples']
        self.activation_shape = tuple(self.metadata['activation_shape'])
        self.cache_files_info = self.metadata['cache_files']
        
        # Create sample index mapping
        self._create_sample_mapping()
        
        # Buffer management for preloading
        self._buffer_cache = {}
        self._buffer_access_order = []
        
        logging.info(f"Loaded cached dataset with {self.total_samples} samples")
    
    def _create_sample_mapping(self):
        """Create mapping from global sample index to (buffer_file, local_index)"""
        self.sample_to_buffer = {}
        
        for buffer_info in self.cache_files_info:
            start_idx, end_idx = buffer_info['sample_range']
            buffer_filename = buffer_info['filename']
            
            for global_idx in range(start_idx, end_idx + 1):
                local_idx = global_idx - start_idx
                self.sample_to_buffer[global_idx] = (buffer_filename, local_idx)
    
    def _load_buffer(self, filename: str) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Load a specific buffer file"""
        filepath = self.cache_dir / filename
        
        if filename.endswith('.safetensors'):
            return self._load_buffer_safetensors(filepath)
        else:
            return self._load_buffer_torch(filepath)
    
    def _load_buffer_safetensors(self, filepath: Path) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Load buffer from safetensors format"""
        # Load tensors
        tensors = load_file(filepath)
        
        # Reconstruct activations
        activations = []
        i = 0
        while f"activation_{i}" in tensors:
            activations.append(tensors[f"activation_{i}"])
            i += 1
        
        activations_tensor = torch.stack(activations)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = [{}] * len(activations)
        
        return activations_tensor, metadata
    
    def _load_buffer_torch(self, filepath: Path) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Load buffer from PyTorch format"""
        data = torch.load(filepath, map_location='cpu')
        return data['activations'], data['metadata']
    
    def _get_buffer_with_cache(self, filename: str) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Get buffer with LRU caching"""
        if filename in self._buffer_cache:
            # Move to end (most recently used)
            self._buffer_access_order.remove(filename)
            self._buffer_access_order.append(filename)
            return self._buffer_cache[filename]
        
        # Load buffer
        activations, metadata = self._load_buffer(filename)
        
        # Add to cache
        self._buffer_cache[filename] = (activations, metadata)
        self._buffer_access_order.append(filename)
        
        # Evict old buffers if cache is full
        while len(self._buffer_cache) > self.preload_buffers:
            oldest_file = self._buffer_access_order.pop(0)
            del self._buffer_cache[oldest_file]
        
        return activations, metadata
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get activation by global index"""
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")
        
        # Get buffer info
        buffer_filename, local_idx = self.sample_to_buffer[idx]
        
        # Load buffer (with caching)
        activations, metadata = self._get_buffer_with_cache(buffer_filename)
        
        # Return specific activation
        return activations[local_idx]


class ActivationCollector:
    """
    Memory-efficient activation collector that caches to disk
    """
    
    def __init__(
            self,
            act_model,
            config: ActivationCacheConfig,
            sampler_config: TokenSamplerConfig,
        ):
        self.act_model = act_model
        self.config = config
        self.cache = ActivationCache(config, sampler_config)
        self.hook = None
        
        # Setup hook
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook to capture activations"""
        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Store activations to cache
            if len(activation.shape) == 3:  # (batch_size, seq_len, hidden_dim)
                self.cache.add_activations(activation)
            else:
                logging.warning(f"Unexpected activation shape: {activation.shape}")
        
        # Get layer by name
        layer = self.act_model
        for attr in self.config.layer_name.split('.'):
            layer = getattr(layer, attr)
        
        self.hook = layer.register_forward_hook(hook_fn)
        logging.info(f"Registered hook on layer: {self.config.layer_name}")
    
    def collect_activations(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        max_samples: Optional[int] = None,
        device: str = 'cuda'
    ) -> str:
        """
        Collect activations and cache to disk
        
        Args:
            dataloader: DataLoader with input data
            max_samples: Maximum number of samples to collect
            device: Device to run model on
            
        Returns:
            Path to cache directory
        """
        self.act_model.eval()
        self.act_model = self.act_model.to(device)
        
        samples_collected = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass (triggers hook and caching)
                _ = self.act_model(batch)
                
                samples_collected = self.cache.total_samples
                
                # Check memory usage periodically
                if batch_idx % 100 == 0:
                    self._check_memory_usage()
                
                if max_samples and samples_collected >= max_samples:
                    logging.info(f"Reached maximum samples limit: {max_samples}")
                    break
        
        # Finalize cache
        cache_info = self.cache.finalize()
        logging.info(f"Collected {cache_info['total_samples']} activations in {cache_info['num_buffers']} buffers")
        
        return str(self.cache.cache_dir)
    
    def _check_memory_usage(self):
        """Check and log memory usage"""
        try:
            import psutil
            memory_gb = psutil.virtual_memory().used / (1024**3)
            if memory_gb > self.config.max_memory_gb:
                logging.warning(f"Memory usage ({memory_gb:.1f}GB) exceeds limit ({self.config.max_memory_gb}GB)")
        except ImportError:
            pass  # psutil not available
    
    def remove_hook(self):
        """Remove the registered hook"""
        if self.hook:
            self.hook.remove()
            self.hook = None


def create_cached_dataloader(
    cache_dir: str,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    preload_buffers: int = 2
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader from cached activations
    
    Args:
        cache_dir: Directory containing cached activations
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        preload_buffers: Number of buffers to keep in memory
        
    Returns:
        DataLoader for cached activations
    """
    dataset = CachedActivationDataset(
        cache_dir=cache_dir,
        shuffle=shuffle,
        preload_buffers=preload_buffers
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def collect_and_cache_activations(
    act_model,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str = "model.encoder.layers.3.norm2",
    cache_dir: str = "./output/activation_cache",
    experiment_name: str = "act_activations",
    buffer_size: int = 128,
    max_samples: Optional[int] = None,
    device: str = 'cuda',
    # Token sampling parameters
    use_token_sampling: bool = True,
    fixed_tokens: List[int] = None,
    sampling_strategy: str = "block_average",
    sampling_stride: int = 8,
    max_sampled_tokens: int = 100,
    block_size: int = 8
) -> str:
    """
    Convenience function to collect and cache activations with optional token sampling
    
    Args:
        act_model: ACT model to collect activations from
        dataloader: DataLoader with input data
        layer_name: Name of layer to hook
        cache_dir: Directory to store cache
        experiment_name: Name for this experiment
        buffer_size: Number of samples per cache file
        max_samples: Maximum samples to collect
        device: Device to run model on
        use_token_sampling: Whether to use token sampling
        fixed_tokens: Token indices to always include (default: [0, 601])
        sampling_strategy: "uniform", "stride", "random_fixed", or "block_average"
        sampling_stride: Take every Nth token when using stride strategy
        max_sampled_tokens: Maximum number of tokens to sample
        block_size: Size of blocks for block_average strategy
        
    Returns:
        Path to cache directory
    """
    config = ActivationCacheConfig(
        cache_dir=cache_dir,
        layer_name=layer_name,
        experiment_name=experiment_name,
        buffer_size=buffer_size,
        use_token_sampling=use_token_sampling,
    )

    sampler_config = None
    if use_token_sampling:
        sampler_config = TokenSamplerConfig(
            fixed_tokens=fixed_tokens,
            sampling_strategy=sampling_strategy,
            sampling_stride=sampling_stride,
            max_sampled_tokens=max_sampled_tokens,
            block_size=block_size
        )
    
    collector = ActivationCollector(act_model, config, sampler_config=sampler_config)
    
    try:
        cache_path = collector.collect_activations(dataloader, max_samples, device)
        return cache_path
    finally:
        collector.remove_hook()
