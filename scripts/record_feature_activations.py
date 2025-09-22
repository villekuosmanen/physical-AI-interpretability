#!/usr/bin/env python

import argparse
from collections import Counter
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.common.datasets.factory import make_dataset_without_config
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.utils import dataloader_collate_fn
from lerobot.configs.policies import PreTrainedConfig

from src.sae import MultiModalSAE, TokenSamplerConfig, TokenSampler


@dataclass
class FrameExample(NamedTuple):
    """Single frame example with metadata"""
    frame_idx: int
    episode_idx: int
    dataset_idx: int  # Add dataset index
    activation_value: float
    batch_data: Dict[str, Any]  # Original batch data for this frame

@dataclass
class FeatureAnalysisConfig:
    """Configuration for feature analysis"""
    # Analysis parameters
    top_k_global: int = 50  # Top K frames globally for each feature
    top_s_per_episode: int = 3  # Top S frames per episode for each feature
    min_variance_threshold: float = 0.01  # Ignore low-variance features
    
    # Feature ranking criteria
    ranking_metric: str = 'variance'  # 'variance', 'range', 'sparsity', 'composite'
    sparsity_threshold: float = 0.1  # Threshold for considering a feature "active"
    use_token_sampling: bool = True  # Enable token sampling
    fixed_tokens: List[int] = None  # Always include these token indices (e.g., [0, 601])

    # Data sampling and storage
    skip_adjacent_frames: int = 5  # Skip frames within this distance in same episode
    max_frames_per_episode: int = 1000  # Limit frames per episode to prevent memory issues
    storage_format: str = "parquet"  # "parquet", "pickle", "feather"
    compression: str = "snappy"  # For parquet: "snappy", "gzip", "brotli"
    
    # Output settings
    output_dir: str = "./feature_analysis"
    episode_data_dir: str = "./output/episode_activations"  # Directory for per-episode files
    save_detailed_examples: bool = True
    save_statistics: bool = True
    create_plots: bool = True
    
    # Analysis scope
    analyze_all_features: bool = False  # If False, only analyze top features
    max_features_to_analyze: int = 100  # Max features to analyze in detail
    
    def __post_init__(self):
        """Initialize fixed_tokens if not provided"""
        if self.fixed_tokens is None:
            self.fixed_tokens = [0, 601]  # Default: VAE latent + proprioception


class StreamingFeatureAnalyzer:
    """
    Memory-efficient feature analyzer that processes episodes incrementally
    """

    def __init__(
        self,
        config: FeatureAnalysisConfig,
        sampler_config: TokenSamplerConfig,
        policy_model = None,
        total_tokens: int = None,
    ):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create episode data directory
        self.episode_data_dir = Path(config.episode_data_dir)
        self.episode_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Infer total_tokens if not provided
        if total_tokens is None and policy_model is not None:
            from src.sae.config import SAETrainingConfig
            temp_config = SAETrainingConfig()
            total_tokens = temp_config._infer_original_num_tokens(policy_model)
            if total_tokens is None:
                total_tokens = 602  # Fallback default
                logging.warning("Could not infer token count from model, using default 602")
            else:
                logging.info(f"Inferred {total_tokens} tokens from policy model for feature analysis")
        elif total_tokens is None:
            total_tokens = 602
            logging.warning("No token count or policy model provided, using default 602")
        
        # Initialize token sampler
        self.token_sampler = TokenSampler(sampler_config, total_tokens)
        
        # Streaming data storage
        self.current_episode_data = []  # List of activation records
        self.current_episode_id = None
        self.current_dataset_id = None
        self.processed_episodes = set()
        
        # Global statistics (kept in memory for efficiency)
        self.processed_frames = 0
        self.total_episodes = 0
        
        # Episode tracking
        self.episode_frame_counts = Counter()
        
        # Log sampling configuration
        sampling_info = self.token_sampler.get_sampling_info()
        logging.info(f"Streaming analyzer - Token sampling configuration: {sampling_info}")
    
    def _save_episode_data(self, episode_id: int):
        """Save current episode data to disk and clear from memory"""
        if not self.current_episode_data:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.current_episode_data)
        
        # Create filename with dataset ID
        filename = f"dataset_{self.current_dataset_id}_episode_{episode_id:06d}"
        
        # Save based on configured format
        if self.config.storage_format == "parquet":
            filepath = self.episode_data_dir / f"{filename}.parquet"
            df.to_parquet(filepath, compression=self.config.compression, index=False)
        elif self.config.storage_format == "pickle":
            filepath = self.episode_data_dir / f"{filename}.pkl"
            df.to_pickle(filepath)
        elif self.config.storage_format == "feather":
            filepath = self.episode_data_dir / f"{filename}.feather"
            df.to_feather(filepath)
        else:
            raise ValueError(f"Unknown storage format: {self.config.storage_format}")
        
        logging.info(f"Saved episode {episode_id} from dataset {self.current_dataset_id} with {len(self.current_episode_data)} activation records to {filepath}")
        
        # Clear episode data from memory
        self.current_episode_data.clear()
        self.processed_episodes.add((self.current_dataset_id, episode_id))  # Store as tuple
        
        # Force garbage collection
        gc.collect()
    
    def _process_episode_transition(self, new_episode_id: int, new_dataset_id: int):
        """Handle transition to a new episode"""
        if self.current_episode_id is not None:
            # Save previous episode data
            self._save_episode_data(self.current_episode_id)
        
        # Update to new episode and dataset
        self.current_episode_id = new_episode_id
        self.current_dataset_id = new_dataset_id
        self.total_episodes += 1
        
        logging.info(f"Processing episode {new_episode_id} from dataset {new_dataset_id} (total episodes processed: {self.total_episodes})")
    
    def _add_activation_records(self, feature_activations_np: np.ndarray, 
                               frame_indices: List[int], episode_indices: List[int],
                               dataset_indices: List[int]):
        """Add activation records to current episode data"""
        batch_size, num_features = feature_activations_np.shape
        
        # Create records for current batch
        for batch_idx in range(batch_size):
            frame_idx = frame_indices[batch_idx]
            episode_idx = episode_indices[batch_idx]
            dataset_idx = dataset_indices[batch_idx]
            
            # Only process if this is the current episode and dataset
            if episode_idx == self.current_episode_id and dataset_idx == self.current_dataset_id:
                for feature_idx in range(num_features):
                    activation_value = float(feature_activations_np[batch_idx, feature_idx])
                    
                    # Add record to current episode data
                    record = {
                        'episode_id': episode_idx,
                        'dataset_id': dataset_idx,
                        'frame_index': frame_idx,
                        'feature_index': feature_idx,
                        'activation': activation_value
                    }
                    self.current_episode_data.append(record)
    
    def save_feature_activations(
        self,
        sae_model: torch.nn.Module,
        act_model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        target_layer: str
    ):
        """
        Main analysis function with streaming processing
        """
        logging.info("Starting streaming feature analysis...")
        
        # Set models to eval mode
        sae_model.eval()
        act_model.eval()
        
        # Hook for collecting activations
        activations_buffer = []
        
        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            activations_buffer.append(activation.detach())
        
        # Register hook
        layer = self._get_layer_by_name(act_model, target_layer)
        hook = layer.register_forward_hook(activation_hook)
        
        try:
            self._collect_feature_activations_streaming(
                sae_model, act_model, dataloader, activations_buffer
            )
            
            # Save final episode if any
            if self.current_episode_id is not None:
                self._save_episode_data(self.current_episode_id)
            
            logging.info(f"Streaming analysis complete! Results saved to {self.output_dir}")
            
        finally:
            hook.remove()
    
    def _collect_feature_activations_streaming(
        self,
        sae_model: torch.nn.Module,
        act_model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        activations_buffer: List[torch.Tensor]
    ):
        """Collect feature activations with streaming episode processing"""
        device = next(sae_model.parameters()).device
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Streaming feature analysis")):
                # TODO: a hack to continue processing
                # mask = batch['dataset_index'] > 18
                # # Check if any elements remain after filtering
                # if not mask.any():
                #     continue
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass through ACT model (triggers hook)
                activations_buffer.clear()
                _ = act_model(batch)
                
                if not activations_buffer:
                    continue
                
                # Get layer activations
                layer_activations = activations_buffer[0]  # (batch_size, seq_len, hidden_dim)
                
                # Apply token sampling (CRITICAL: must match training sampling)
                layer_activations = layer_activations.permute(1, 0, 2).contiguous() # flip batch size and tokens_length dims
                sampled_activations = self.token_sampler.sample_tokens(layer_activations)
                _, feature_activations = sae_model(sampled_activations)
                
                # Convert to numpy for processing
                feature_activations_np = feature_activations.detach().cpu().numpy()
                batch_size, num_features = feature_activations_np.shape
                
                # Get frame, episode and dataset indices from batch
                frame_indices = batch['frame_index'].cpu().numpy().tolist()
                episode_indices = batch['episode_index'].cpu().numpy().tolist()
                dataset_indices = batch['dataset_index'].cpu().numpy().tolist()

                # Check for episode transitions
                unique_episodes = sorted(set(zip(episode_indices, dataset_indices)))  # Sort by both episode and dataset
                for episode_id, dataset_id in unique_episodes:
                    if self.current_episode_id is None:
                        # First episode
                        self.current_episode_id = episode_id
                        self.current_dataset_id = dataset_id
                        self.total_episodes = 1
                        logging.info(f"Starting analysis with episode {episode_id} from dataset {dataset_id}")
                    elif (episode_id > self.current_episode_id) or (dataset_id > self.current_dataset_id):
                        # New episode or dataset detected
                        self._process_episode_transition(episode_id, dataset_id)
                
                # Update episode frame counts
                for episode_idx, dataset_idx in zip(episode_indices, dataset_indices):
                    self.episode_frame_counts[(dataset_idx, episode_idx)] += 1
                
                # Add activation records to current episode
                self._add_activation_records(feature_activations_np, frame_indices, episode_indices, dataset_indices)
                
                self.processed_frames += batch_size
                
                # Log progress periodically
                if batch_idx % 100 == 0:
                    logging.info(f"Processed {self.processed_frames} frames, {self.total_episodes} episodes")

    def _get_layer_by_name(self, model, name: str):
        """Get layer by name from model"""
        layer = model
        for attr in name.split('.'):
            layer = getattr(layer, attr)
        return layer
        
    def _update_top_examples(
        self,
        feature_idx: int,
        feature_values: np.ndarray,
        frame_indices: List[int],
        episode_indices: List[int],
        batch_data: Dict[str, Any]
    ):
        """Update top examples for a feature"""
        for i, (activation_val, frame_idx, episode_idx) in enumerate(
            zip(feature_values, frame_indices, episode_indices)
        ):
            # Create example
            example = FrameExample(
                frame_idx=frame_idx,
                episode_idx=episode_idx,
                activation_value=float(activation_val),
                batch_data=self._extract_frame_data(batch_data, i)
            )
            
            # Update global top examples
            self._update_global_top_k(feature_idx, example)
            
            # Update per-episode top examples
            self._update_episode_top_s(feature_idx, example)
    
    def _extract_frame_data(self, batch_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Extract data for specific frame from batch"""
        frame_data = {}
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) > 0 and value.shape[0] > index:
                    frame_data[key] = value[index].detach().cpu()
            elif isinstance(value, list) and len(value) > index:
                frame_data[key] = value[index]
        
        return frame_data
    
    def _update_global_top_k(self, feature_idx: int, example: FrameExample):
        """Update global top-k examples for feature"""
        top_examples = self.top_global_examples[feature_idx]
        
        if len(top_examples) < self.config.top_k_global:
            top_examples.append(example)
        else:
            # Find minimum activation in current top-k
            min_example = min(top_examples, key=lambda x: x.activation_value)
            if example.activation_value > min_example.activation_value:
                top_examples.remove(min_example)
                top_examples.append(example)
        
        # Keep sorted (highest first)
        top_examples.sort(key=lambda x: x.activation_value, reverse=True)
    
    def _update_episode_top_s(self, feature_idx: int, example: FrameExample):
        """Update per-episode top-s examples for feature"""
        episode_examples = self.top_episode_examples[feature_idx][example.episode_idx]
        
        if len(episode_examples) < self.config.top_s_per_episode:
            episode_examples.append(example)
        else:
            # Find minimum activation in current top-s for this episode
            min_example = min(episode_examples, key=lambda x: x.activation_value)
            if example.activation_value > min_example.activation_value:
                episode_examples.remove(min_example)
                episode_examples.append(example)
        
        # Keep sorted (highest first)
        episode_examples.sort(key=lambda x: x.activation_value, reverse=True)

def record_sae_features(
    sae_model_path: str,
    act_model,
    dataloader: torch.utils.data.DataLoader,
    target_layer: str,
    config: Optional[FeatureAnalysisConfig] = None,
):
    """
    Main function to record SAE feature activations
    
    Args:
        sae_model_path: Path to trained SAE checkpoint
        act_model: Original ACT model
        dataloader: DataLoader for analysis dataset
        target_layer: Layer name being analyzed
        config: Analysis configuration
        
    """
    if config is None:
        config = FeatureAnalysisConfig()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load SAE model
    logging.info(f"Loading SAE model from {sae_model_path}")
    checkpoint = torch.load(sae_model_path, map_location='cpu')
    
    # Determine SAE type from checkpoint
    model_config = checkpoint.get('config', {})
    
    # Token sampling config
    sampler_config = TokenSamplerConfig()

    # Create SAE model
    sae_model = MultiModalSAE(
        num_tokens=model_config.get('num_tokens', 77),
        token_dim=model_config.get('token_dim', 128),
        feature_dim=model_config.get('feature_dim', 64)
    )
    
    # Load state dict
    sae_model.load_state_dict(checkpoint['model_state_dict'])
    sae_model.eval()
    
    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sae_model = sae_model.to(device)
    act_model = act_model.to(device)
    
    analyzer = StreamingFeatureAnalyzer(config, sampler_config, policy_model=act_model)
    logging.info("Using StreamingFeatureAnalyzer for memory-efficient processing")
    
    analyzer.save_feature_activations(sae_model, act_model, dataloader, target_layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train safety classifier")
    # held out test set: villekuosmanen/pick_coffee_prop_20Hz_3, villekuosmanen/dAgger_coffee_prop_2.4.0, villekuosmanen/eval_4Apr25
    parser.add_argument("--repo-id", type=str, default="[villekuosmanen/pick_coffee_prop_20Hz,villekuosmanen/pick_coffee_prop_20Hz_alt,villekuosmanen/dAgger_coffee_prop_2.1.0,villekuosmanen/dAgger_coffee_prop_2.2.0,villekuosmanen/dAgger_coffee_prop_2.3.0,villekuosmanen/dAgger_coffee_prop_3.0.0,villekuosmanen/dAgger_coffee_prop_3.1.1,villekuosmanen/dAgger_plus_coffee_prop,villekuosmanen/dAgger_coffee_prop,villekuosmanen/dAgger_plus_v2_coffee_prop,villekuosmanen/eval_26Jan25,villekuosmanen/eval_27Jan25,villekuosmanen/eval_29Jan25,villekuosmanen/eval_31Jan25,villekuosmanen/eval_31Jan25_v2,villekuosmanen/eval_2Feb25_v2,villekuosmanen/eval_3Feb25,villekuosmanen/eval_3Mar25,villekuosmanen/eval_6Mar25,villekuosmanen/eval_28May25,villekuosmanen/eval_30May25,villekuosmanen/eval_30May25_alt,villekuosmanen/eval_31May25,villekuosmanen/eval_8Jun25,villekuosmanen/eval_8Jun25_aug]", 
                        help="Dataset repo ID for training")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # Example usage
    config = FeatureAnalysisConfig(
        top_k_global=50,
        top_s_per_episode=1,
        ranking_metric='variance',
        min_variance_threshold=0.01,
        max_features_to_analyze=100,
        output_dir="./feature_analysis_results",
        create_plots=True,
        save_detailed_examples=True
    )
    
    sae_model_path = "output/sae_checkpoints/multimodal_sae/checkpoint_epoch_19.pt"
    
    # Load your pre-trained ACT model and dataloader
    chunk_size = 100
    dataset = make_dataset_without_config(
        repo_id=args.repo_id,
        action_delta_indices=list(range(chunk_size)),
        observation_delta_indices=None,
        force_cache_sync=True,
    )
    dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=128,
        shuffle=False,
        collate_fn=dataloader_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy = make_policy(policy_cfg, ds_meta=dataset._datasets[0].meta)
    
    # Run analysis
    results = record_sae_features(
        sae_model_path, 
        policy, 
        dataloader, 
        target_layer="model.encoder.layers.3.norm2",
        config=config
    )
