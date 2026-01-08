from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any

from safetensors.torch import save_file, load_file
import torch
from tqdm import tqdm

from .token_sampler import TokenSamplerConfig, TokenSampler


@dataclass
class ActivationCacheConfig:
    """
    Configuration for activation caching.
    
    :meta private:
    """
    # Cache settings
    cache_dir: str = "./output/activation_cache"
    buffer_size: int = 128  # Number of samples per cache file
    
    # Data organization
    layer_name: str = "encoder.layers.4"
    experiment_name: str = "act_activations"
    use_token_sampling: bool = True  # Enable token sampling
    
    # Memory management
    max_memory_gb: float = 4.0  # Maximum memory to use for caching
    cleanup_on_start: bool = False  # Clean old cache files (set True to force fresh start)
    
    # Validation
    validate_cache: bool = True  # Validate cached files on load
    cache_metadata: bool = True  # Save metadata with cache

class ActivationCache:
    """
    Manages caching of activations to disk for memory-efficient SAE training.

    :meta private:
    """
    
    def __init__(
            self,
            config: ActivationCacheConfig,
            sampler_config: TokenSamplerConfig,
            total_tokens: int = None,
            policy_model = None,
        ):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / config.experiment_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # For now, create sampler with a placeholder value - we'll update it when we see real data
        if total_tokens is None:
            total_tokens = 602  # Temporary placeholder, will be updated from actual data
            logging.info("Using placeholder token count 602, will be updated from actual activation data")
        
        self.sampler = TokenSampler(sampler_config, total_tokens)
        self._sampler_config = sampler_config  # Store config for potential recreation
        
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
            'activation_shape': None,
            # Progress tracking for resumability
            'collection_status': 'in_progress',  # 'in_progress', 'completed', 'failed'
            'last_batch_idx': -1,  # Last successfully processed batch
            'last_updated': time.time(),
            'resume_info': {
                'dataloader_position': 0,
                'can_resume': True,
                'interruption_count': 0
            }
        }
        
        # Try to load existing cache state for resumption first
        cache_loaded = self._try_load_existing_cache()
        
        # Only cleanup if we couldn't load a valid resumable cache
        if config.cleanup_on_start and not cache_loaded:
            logging.info("No resumable cache found, cleaning up any invalid cache files")
            self._cleanup_cache()
        
        logging.info(f"Initialized activation cache at {self.cache_dir}")
    
    def update_batch_idx(self, batch_idx):
        self.metadata['last_batch_idx'] = batch_idx

    def _try_load_existing_cache(self):
        """Try to load existing cache metadata for resumption"""
        metadata_path = self.cache_dir / "cache_metadata.json"
        
        if metadata_path.exists():
            try:
                # Load existing metadata
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                                
                # Check if cache is resumable
                if (
                    existing_metadata.get('collection_status') == 'in_progress' and 
                    self._validate_existing_cache_files(existing_metadata)
                ):
                    
                    # Restore state for resumption
                    self.metadata.update(existing_metadata)
                    self.total_samples = existing_metadata.get('total_samples', 0)
                    
                    # If metadata doesn't have cache_files info, reconstruct it from directory
                    cache_files_info = existing_metadata.get('cache_files', [])
                    if not cache_files_info:
                        # Scan directory for existing cache files
                        cache_files_info = self._reconstruct_cache_files_info()
                        self.metadata['cache_files'] = cache_files_info
                    
                    self.buffer_count = len(cache_files_info)
                    self.cache_files = [
                        self.cache_dir / info['filename'] 
                        for info in cache_files_info
                    ]
                    
                    # Increment interruption count
                    self.metadata['resume_info']['interruption_count'] += 1
                    
                    logging.info(f"Found resumable cache with {self.total_samples} samples")
                    logging.info(f"Last batch: {self.metadata['last_batch_idx']}, "
                               f"Interruptions: {self.metadata['resume_info']['interruption_count']}")
                    return True
                else:
                    logging.info("Existing cache found but not resumable - starting fresh")
                    
            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                logging.warning(f"Could not load existing cache state: {e}")
        
        return False
    
    def _validate_existing_cache_files(self, metadata: dict) -> bool:
        """Validate that all referenced cache files actually exist and are valid"""
        try:
            cache_files_info = metadata.get('cache_files', [])
            
            # If no cache_files info in metadata, check if any safetensors files exist
            if not cache_files_info:
                safetensors_files = list(self.cache_dir.glob("activations_buffer_*.safetensors"))
                if not safetensors_files:
                    logging.warning("No cache files found in directory")
                    return False
                logging.info(f"Found {len(safetensors_files)} cache files without metadata")
                return True  # We can reconstruct the metadata
            
            # Validate referenced files
            for file_info in cache_files_info:
                file_path = self.cache_dir / file_info['filename']
                if not file_path.exists():
                    logging.warning(f"Missing cache file: {file_path}")
                    return False
                    
                # Basic size check - file should have reasonable size
                if file_path.stat().st_size < 100:  # Very small files are likely corrupted
                    logging.warning(f"Cache file too small (likely corrupted): {file_path}")
                    return False
            
            return True
        except Exception as e:
            logging.warning(f"Cache validation failed: {e}")
            return False
    
    def _reconstruct_cache_files_info(self) -> List[Dict[str, Any]]:
        """Reconstruct cache files info by scanning the directory"""
        cache_files_info = []
        
        # Find all safetensors files in the cache directory
        safetensors_files = sorted(self.cache_dir.glob("activations_buffer_*.safetensors"))
        
        for filepath in safetensors_files:
            # Extract buffer index from filename
            filename = filepath.name
            try:
                buffer_idx = int(filename.split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                continue
            
            # Try to determine number of samples by loading metadata
            metadata_path = filepath.with_suffix('.json')
            num_samples = 0
            sample_range = (0, 0)
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        buffer_metadata = json.load(f)
                    num_samples = len(buffer_metadata)
                    if buffer_metadata:
                        # Calculate sample range
                        sample_indices = [item.get('sample_idx', 0) for item in buffer_metadata]
                        sample_range = (min(sample_indices), max(sample_indices))
                except (json.JSONDecodeError, KeyError):
                    pass
            
            cache_files_info.append({
                'filename': filename,
                'buffer_idx': buffer_idx,
                'num_samples': num_samples,
                'sample_range': sample_range
            })
        
        logging.info(f"Reconstructed cache info for {len(cache_files_info)} buffers")
        return cache_files_info
    
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

        activations = activations.permute(1, 0, 2).contiguous() # flip batch size and tokens_length dims
        
        # Store activation shape for metadata
        if self.metadata['activation_shape'] is None:
            self.metadata['activation_shape'] = list(activations.shape[1:])  # Exclude batch dimension
        
        # Record the original number of tokens (before sampling) - this is the key insight!
        original_num_tokens = activations.shape[1]  # num_tokens from (batch_size, num_tokens, token_dim)
        if 'original_num_tokens' not in self.metadata:
            self.metadata['original_num_tokens'] = original_num_tokens
            logging.info(f"Recorded original_num_tokens from actual data: {original_num_tokens}")
            
            # Update the sampler with the correct token count if it was using placeholder
            if self.sampler.total_tokens != original_num_tokens:
                logging.info(f"Updating TokenSampler with correct token count: {self.sampler.total_tokens} -> {original_num_tokens}")
                self.sampler = TokenSampler(self._sampler_config, original_num_tokens)
                
        elif self.metadata['original_num_tokens'] != original_num_tokens:
            logging.warning(f"Token count mismatch! Expected {self.metadata['original_num_tokens']}, got {original_num_tokens}")
        
        # Add to buffer
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
        
        # Update and save incremental metadata after each buffer
        self.metadata['total_samples'] = self.total_samples
        self.metadata['num_buffers'] = self.buffer_count + 1  # +1 because we're about to increment
        self.metadata['last_updated'] = time.time()
        
        # Save incremental metadata so the cache can be loaded even during collection
        metadata_path = self.cache_dir / "cache_metadata.json" 
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
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
        
        # Only mark as completed if collection was actually complete
        if self.metadata.get('collection_complete', False):
            self.metadata['collection_status'] = 'completed'
            self.metadata['resume_info']['can_resume'] = False
        else:
            # Collection was interrupted - keep as in_progress and resumable
            self.metadata['collection_status'] = 'in_progress'
            self.metadata['resume_info']['can_resume'] = True
            logging.info("Cache finalized but collection incomplete - remains resumable")
        
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
    Dataset that loads activations from cached files.

    :meta private:
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
    Memory-efficient activation collector that caches to disk.

    :meta private:
    """
    
    def __init__(
            self,
            act_model,
            config: ActivationCacheConfig,
            sampler_config: TokenSamplerConfig,
        ):
        self.act_model = act_model
        self.config = config
        self.cache = ActivationCache(config, sampler_config, policy_model=act_model)
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
        Collect activations and cache to disk with resumption support
        
        Args:
            dataloader: DataLoader with input data
            max_samples: Maximum number of samples to collect
            device: Device to run model on
            
        Returns:
            Path to cache directory
        """
        self.act_model.eval()
        self.act_model = self.act_model.to(device)
        
        # Determine starting point for resumption
        start_batch_idx = self.cache.metadata.get('last_batch_idx', -1) + 1
        total_batches = len(dataloader)
        
        # Set completion tracking metadata on first run or update if needed
        if self.cache.metadata.get('total_batches_expected') is None:
            self.cache.metadata['total_batches_expected'] = total_batches
            self.cache.metadata['max_samples_target'] = max_samples
            logging.info(f"Set completion target: {total_batches} batches, max_samples: {max_samples}")
        
        if start_batch_idx > 0:
            logging.info(f"Resuming activation collection from batch {start_batch_idx}/{total_batches}")
            logging.info(f"Already collected {self.cache.total_samples} samples")
        
        samples_collected = self.cache.total_samples
        processed_batches = 0
        
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations", 
                                                     initial=start_batch_idx, total=total_batches)):
                    # Skip batches we've already processed (for resumption)
                    if batch_idx < start_batch_idx:
                        continue
                    
                    # Move batch to device
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device)
                    
                    # Extract dataset indices if available for better tracking
                    dataset_idx = None
                    if 'dataset_index' in batch:
                        dataset_idx = batch['dataset_index'][0].item() if torch.is_tensor(batch['dataset_index']) else batch['dataset_index'][0]
                    
                    # Forward pass (triggers hook and caching)
                    self.cache.update_batch_idx(batch_idx)
                    _ = self.act_model.select_action(batch)
                    self.act_model.reset()
                    samples_collected = self.cache.total_samples
                    processed_batches += 1
                    
                    # Check memory usage periodically
                    if batch_idx % 100 == 0:
                        self._check_memory_usage()
                    
                    if max_samples and samples_collected >= max_samples:
                        logging.info(f"Reached maximum samples limit: {max_samples}")
                        self.cache.metadata['collection_complete'] = True
                        break
                
                # Check if we completed all batches
                if batch_idx >= total_batches - 1:
                    logging.info(f"Completed all {total_batches} batches")
                    self.cache.metadata['collection_complete'] = True
                        
        except Exception as e:
            # Mark cache as failed but keep progress for potential resumption
            self.cache.metadata['collection_status'] = 'failed'
            self.cache.metadata['resume_info']['can_resume'] = True
            
            logging.error(f"Activation collection failed at batch {batch_idx if 'batch_idx' in locals() else start_batch_idx}: {e}")
            raise
        
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


def is_cache_valid(cache_dir: str) -> bool:
    """
    Check if a cache directory contains valid cached activations.
    Now supports checking resumable (incomplete) caches.
    
    Args:
        cache_dir: Directory to check
        
    Returns:
        True if cache is valid (complete or resumable), False otherwise

    :meta private:
    """
    cache_path = Path(cache_dir)
    
    # Check if directory exists
    if not cache_path.exists():
        return False
    
    # Check if metadata file exists
    metadata_path = cache_path / "cache_metadata.json"
    
    # For resumable caches, progress.json might exist without full metadata
    if not metadata_path.exists():
        return False
    
    try:
        # Load metadata if available
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Check collection status
        collection_status = metadata.get('collection_status')
        
        if collection_status == 'completed':
            # For completed caches, do full validation
            required_fields = ['total_samples', 'num_buffers', 'cache_files', 'activation_shape']
            if not all(field in metadata for field in required_fields):
                return False
            
            # Check if cache files actually exist
            cache_files_info = metadata.get('cache_files', [])
            for file_info in cache_files_info:
                file_path = cache_path / file_info['filename']
                if not file_path.exists():
                    return False
            
            # Check that we have a reasonable number of samples
            if metadata.get('total_samples', 0) <= 0:
                return False
            
        return True
        
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return False

def get_cache_status(cache_dir: str) -> dict:
    """
    Get detailed status information about a cache directory.
    
    Args:
        cache_dir: Directory to check
        
    Returns:
        Dictionary with cache status information

    :meta private:
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {'status': 'missing', 'exists': False}
    
    status_info = {
        'exists': True,
        'path': str(cache_path),
        'status': 'unknown',
        'total_samples': 0,
        'can_resume': False,
        'last_batch_idx': -1,
        'interruption_count': 0,
        'cache_files': 0
    }
    
    try:
        # Load metadata if available
        metadata_path = cache_path / "cache_metadata.json"
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        
        # Extract status information
        status_info['status'] = metadata.get('collection_status')
        status_info['total_samples'] = metadata.get('total_samples', 0)
        status_info['can_resume'] = metadata.get('resume_info', {}).get('can_resume', False)
        status_info['last_batch_idx'] = metadata.get('last_batch_idx', -1)
        status_info['interruption_count'] = metadata.get('resume_info', {}).get('interruption_count', 0)
        status_info['cache_files'] = len(metadata.get('cache_files', []))
        
        if metadata.get('created_at'):
            status_info['created_at'] = metadata['created_at']
        if metadata.get('last_updated'):
            status_info['last_updated'] = metadata.get('last_updated')
        if metadata.get('finalized_at'):
            status_info['finalized_at'] = metadata['finalized_at']
            
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        status_info['status'] = 'corrupted'
        status_info['error'] = str(e)
    
    return status_info


def cleanup_invalid_cache(cache_dir: str) -> None:
    """
    Remove an invalid cache directory and all its contents.
    
    Args:
        cache_dir: Directory to clean up

    :meta private:
    """
    cache_path = Path(cache_dir)
    
    if cache_path.exists():
        import shutil
        logging.info(f"Cleaning up invalid cache directory: {cache_path}")
        shutil.rmtree(cache_path)


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

    :meta private:
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


def load_original_num_tokens_from_cache(cache_path: str) -> Optional[int]:
    """
    Load the original number of tokens from cached activation metadata.
    
    Args:
        cache_path: Path to the activation cache directory
        
    Returns:
        Original number of tokens if found, None otherwise
    """
    cache_path = Path(cache_path)
    
    # Look for metadata.json file
    metadata_file = cache_path / "cache_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            original_num_tokens = metadata.get('original_num_tokens')
            if original_num_tokens is not None:
                logging.info(f"Loaded original_num_tokens from cache metadata: {original_num_tokens}")
                return int(original_num_tokens)
            else:
                logging.warning("original_num_tokens not found in cache metadata")
        except Exception as e:
            logging.warning(f"Could not load metadata from {metadata_file}: {e}")
    else:
        logging.warning(f"Metadata file not found at {metadata_file}")
    
    return None


def collect_and_cache_activations(
    act_model,
    dataloader: torch.utils.data.DataLoader,
    layer_name: str,
    cache_dir: str,
    experiment_name: str = "act_activations",
    buffer_size: int = 128,
    max_samples: Optional[int] = None,
    device: str = 'cuda',
    # Cache management
    cleanup_on_start: bool = False,  # Set True to force clean start instead of resuming
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
        cleanup_on_start: If True, force clean start instead of resuming from cache
        use_token_sampling: Whether to use token sampling
        fixed_tokens: Token indices to always include (default: [0, 601])
        sampling_strategy: "uniform", "stride", "random_fixed", or "block_average"
        sampling_stride: Take every Nth token when using stride strategy
        max_sampled_tokens: Maximum number of tokens to sample
        block_size: Size of blocks for block_average strategy
        
    Returns:
        Path to cache directory

    :meta private:
    """
    config = ActivationCacheConfig(
        cache_dir=cache_dir,
        layer_name=layer_name,
        experiment_name=experiment_name,
        buffer_size=buffer_size,
        use_token_sampling=use_token_sampling,
        cleanup_on_start=cleanup_on_start,
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
