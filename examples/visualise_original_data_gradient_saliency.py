#!/usr/bin/env python

"""
Script to analyze policy behavior using gradient-based saliency maps.
Runs policy inference on episodes and visualizes which pixels and proprioceptive
features influence the policy's predictions using Integrated Gradients.
"""

import argparse
import os
import time
import subprocess
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.import_utils import register_third_party_plugins
from robocandywrapper.factory import make_dataset_without_config

from physical_ai_interpretability.attention_maps import ACTPolicyWithGradients


TRAIN_DATASET_REPO_IDS = [
    "villekuosmanen/build_block_tower",
    "villekuosmanen/dAgger_build_block_tower_1.0.0",
    "villekuosmanen/dAgger_build_block_tower_1.1.0",
    "villekuosmanen/dAgger_build_block_tower_1.2.0",
    "villekuosmanen/dAgger_build_block_tower_1.3.0",
    "villekuosmanen/dAgger_build_block_tower_1.4.0",
    "villekuosmanen/fail_build_block_tower_stationary",
    "villekuosmanen/fail_build_block_tower_autonomous_interaction",
]


def none_or_int(value):
    if value == "None":
        return None
    return int(value)

def encode_video_ffmpeg(frames, output_filename, fps, pix_fmt_in="bgr24"):
    """
    Encodes a list of numpy frames into a video using ffmpeg.
    """
    if not frames:
        print(f"No frames to encode for {output_filename}.")
        return

    height, width, channels = frames[0].shape
    if channels != 3:
        print(f"Error: Frames must be 3-channel (BGR). Got {channels} channels.")
        return

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',  # Frame size
        '-pix_fmt', pix_fmt_in,     # Input pixel format
        '-r', str(fps),             # Frames per second
        '-i', '-',                  # Input comes from stdin
        '-an',                      # No audio
        '-vcodec', 'libx264',       # Output video codec
        '-pix_fmt', 'yuv420p',      # Output pixel format for broad compatibility
        '-crf', '23',               # Constant Rate Factor (quality, 18-28 is good range)
        output_filename
    ]

    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            process.stdin.write(frame.tobytes())
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error encoding video {output_filename}:")
            print(f"FFmpeg stderr:\n{stderr.decode(errors='ignore')}")
        else:
            print(f"Successfully encoded video: {output_filename}")
            
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred during video encoding for {output_filename}: {e}")

def load_policy(policy_path: str, dataset_meta, policy_overrides: list = None,
                n_integration_steps: int = 50, ig_batch_size: int = 10,
                normalization_mode: str = 'percentile',
                normalization_percentile: float = 95.0) -> Tuple[torch.nn.Module, dict]:
    """Load and initialize a policy from checkpoint."""
    
    # Load regular LeRobot policy
    if policy_overrides:
        # Convert list of "key=value" strings to dict
        overrides = {}
        for override in policy_overrides:
            key, value = override.split('=', 1)
            overrides[key] = value
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, **overrides)
    else:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

    # NOTE: policy has to be an ACT policy for this to work
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    
    # Create processors
    processor_kwargs = {}
    postprocessor_kwargs = {}

    device = torch.device('cuda')
    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset_meta.stats,
            "features": {**policy.config.input_features, **policy.config.output_features},
            "norm_map": policy.config.normalization_mapping,
        },
    }
    postprocessor_kwargs["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "stats": dataset_meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy.config.pretrained_path,
        dataset_stats=dataset_meta.stats,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    
    # Wrap policy with gradient visualizer
    policy = ACTPolicyWithGradients(
        policy, 
        preprocessor,
        n_integration_steps=n_integration_steps,
        ig_batch_size=ig_batch_size,
        normalization_mode=normalization_mode,
        normalization_percentile=normalization_percentile
    )
    return policy, policy_cfg

def prepare_observation_for_policy(frame: dict, 
                                 device: torch.device, 
                                 model_dtype: torch.dtype = torch.float32,
                                 debug: bool = False) -> dict:
    """Convert dataset frame to policy observation format."""
    observation = {}
    
    for key, value in frame.items():
        if "image" in key:
            if debug:
                print(f"Processing {key}: original shape {value.shape}, dtype {value.dtype}")
            
            # Convert image to policy format: channel first, float32 in [0,1], with batch dimension
            if isinstance(value, torch.Tensor):
                # Remove any extra batch dimensions first
                while value.dim() > 3:
                    value = value.squeeze(0)
                
                # Now we should have 3D tensor in format (H, W, C) from camera
                if value.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for {key} after squeezing, got shape {value.shape}")
                
                # Camera images from your robot are in (H, W, C) format, so we need to permute to (C, H, W)                
                # Let's identify dimensions by size
                h, w, c = value.shape
                
                # Sanity check: channels should be 1 or 3
                if c not in [1, 3]:
                    # Maybe the format is actually (H, C, W) or (C, H, W)?
                    if h in [1, 3]:
                        # Format is (C, H, W) - already correct
                        pass
                    elif w in [1, 3]:
                        # Format is (H, W, C) but W is the channel dim - unusual
                        value = value.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                    else:
                        # Assume standard (H, W, C) and C is whatever it is
                        value = value.permute(2, 0, 1)
                else:
                    # Standard (H, W, C) format - convert to (C, H, W)
                    value = value.permute(2, 0, 1)
                
                if debug:
                    print(f"After permutation: {value.shape}")
                
                # Ensure float and normalize if needed
                if value.dtype != model_dtype:
                    value = value.type(model_dtype)
                
                # Normalize to [0, 1] if values are in [0, 255] range
                if value.max() > 1.0:
                    value = value / 255.0
                
                if debug:
                    print(f"Final shape for {key}: {value.shape}, range: [{value.min():.3f}, {value.max():.3f}]")
            
            observation[key] = value.unsqueeze(0).to(device)  # Add batch dimension
            
        elif key in ["observation.state", "robot_state", "state", "observation.state.pos"]:
            # Proprioceptive state
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).type(model_dtype)
            observation[key] = value.to(device)
    
    return observation

def analyze_episode(dataset: LeRobotDataset,
                   policy,
                   episode_id: int,
                   device: torch.device,
                   output_dir: str,
                   aggregation: str = 'sum',
                   timestep_idx: Optional[int] = None,
                   action_dim_idx: Optional[int] = None,
                   model_dtype: torch.dtype = torch.float32) -> Dict:
    """
    Run policy inference on an episode and analyze gradient-based importance.
    
    Args:
        dataset: LeRobot dataset
        policy: Policy wrapped with ACTPolicyWithGradients
        episode_id: Episode ID to analyze
        device: Device for computation
        output_dir: Directory to save results
        aggregation: How to aggregate trajectory for gradients ('sum', 'norm', 'timestep', 'action_dim')
        timestep_idx: Which timestep to analyze (if aggregation='timestep')
        action_dim_idx: Which action dimension to analyze (if aggregation='action_dim')
        model_dtype: Model data type
    
    Returns:
        Dictionary containing analysis results
    """
    
    # Filter dataset to only include the specified episode
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    episode_length = len(episode_frames)
    
    if episode_length == 0:
        raise ValueError(f"Episode {episode_id} not found or is empty")
    
    print(f"Analyzing episode {episode_id} with {episode_length} frames")
    print(f"Gradient aggregation method: {aggregation}")
    if timestep_idx is not None:
        print(f"  Timestep index: {timestep_idx}")
    if action_dim_idx is not None:
        print(f"  Action dimension index: {action_dim_idx}")
    
    # Initialize storage for results
    saliency_videos = None
    side_by_side_buffer = []
    actions_predicted = []
    actions_ground_truth = []
    timestamps = []
    proprio_saliencies = []
    
    # Debug policy configuration
    if hasattr(policy, 'config'):
        print("=== Policy Configuration ===")
        
        # Handle image features
        image_features = getattr(policy.config, 'image_features', None)
        if image_features:
            if hasattr(image_features, '__iter__') and not isinstance(image_features, str):
                image_feature_names = [getattr(f, 'name', str(f)) for f in image_features]
            else:
                image_feature_names = [str(image_features)]
            print(f"Image features: {image_feature_names}")
        else:
            print(f"Image features: None")
            
        # Handle robot state feature
        robot_state_feature = getattr(policy.config, 'robot_state_feature', None)
        if robot_state_feature:
            robot_state_name = getattr(robot_state_feature, 'name', str(robot_state_feature))
            print(f"Robot state feature: {robot_state_name}")
        else:
            print(f"Robot state feature: None")
            
        print(f"Env state feature: {getattr(policy.config, 'env_state_feature', 'None')}")
        print(f"Chunk size: {getattr(policy.config, 'chunk_size', 'None')}")
        print("=" * 30)
    
    # Process each frame
    for i in tqdm(range(episode_length), desc="Processing frames"):
        # if i < 100:
        #     continue
        # if i > 150:
        #     break
        frame = dataset[episode_frames[i]['index'].item()]
        timestamps.append(frame['timestamp'].item())
        
        # Prepare observation for policy (with debug on first frame)
        observation = prepare_observation_for_policy(frame, device, model_dtype, debug=(i==0))
        
        # Run policy inference with gradient computation
        action, saliency_maps, proprio_saliency = policy.select_action(
            observation,
            aggregation=aggregation,
            timestep_idx=timestep_idx,
            action_dim_idx=action_dim_idx
        )
        
        # Generate saliency visualizations
        visualizations = policy.visualize_saliency(
            saliency_maps=saliency_maps,
            observation=observation,
        )
        
        # Initialize video buffers on first frame
        if saliency_videos is None and visualizations:
            num_cameras = len(visualizations)
            saliency_videos = [[] for _ in range(num_cameras)]
            print(f"Detected {num_cameras} camera views for saliency visualization")
        
        # Store saliency frames
        if saliency_videos is not None:
            valid_frames_this_step = []
            for j, vis in enumerate(visualizations):
                if vis is not None and j < len(saliency_videos):
                    saliency_videos[j].append(vis.copy())
                    valid_frames_this_step.append(vis.copy())
                else:
                    valid_frames_this_step.append(None)
            
            # Create side-by-side frame
            if len(valid_frames_this_step) == num_cameras and all(f is not None for f in valid_frames_this_step):
                first_height = valid_frames_this_step[0].shape[0]
                if all(f.shape[0] == first_height for f in valid_frames_this_step):
                    side_by_side_frame = np.hstack(valid_frames_this_step)
                    side_by_side_buffer.append(side_by_side_frame)
        
        # Store predicted action
        actions_predicted.append(action.squeeze(0).cpu().numpy())
        
        # Store ground truth action
        if 'action' in frame:
            actions_ground_truth.append(frame['action'].numpy())
        
        # Store proprioception saliency
        proprio_saliencies.append(proprio_saliency)
    
    # Generate output files
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    
    # Save saliency videos
    if saliency_videos:
        for i, cam_buffer in enumerate(saliency_videos):
            if cam_buffer:
                output_filename = f"{output_dir}/gradient_saliency_ep{episode_id}_cam{i}_{aggregation}_{timestamp_str}.mp4"
                encode_video_ffmpeg(cam_buffer, output_filename, dataset.fps)
        
        if side_by_side_buffer:
            output_filename_sbs = f"{output_dir}/gradient_saliency_ep{episode_id}_combined_{aggregation}_{timestamp_str}.mp4"
            encode_video_ffmpeg(side_by_side_buffer, output_filename_sbs, dataset.fps)
    
    # Analyze and save importance results
    analysis_results = {
        'episode_id': episode_id,
        'episode_length': episode_length,
        'timestamps': timestamps,
        'actions_predicted': actions_predicted,
        'actions_ground_truth': actions_ground_truth,
        'proprio_saliencies': proprio_saliencies,
        'aggregation_method': aggregation,
    }
    
    # Print summary statistics
    print(f"\nProprioception saliency statistics:")
    print(f"  Mean: {np.mean(proprio_saliencies):.4f}")
    print(f"  Std:  {np.std(proprio_saliencies):.4f}")
    print(f"  Min:  {np.min(proprio_saliencies):.4f}")
    print(f"  Max:  {np.max(proprio_saliencies):.4f}")
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Analyze policy behavior using gradient-based saliency maps")
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Repository ID of the dataset to analyze")
    parser.add_argument("--episode-id", type=int, default=None,
                        help="Episode ID to analyze (if not specified, analyzes all episodes)")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--output-dir", type=str, default="./output/gradient_analysis_output",
                        help="Directory to save analysis results")
    parser.add_argument("--policy-overrides", type=str, nargs="*",
                        help="Policy config overrides in key=value format")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    parser.add_argument("--model-dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model data type")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "norm", "timestep", "action_dim"],
                        help="How to aggregate trajectory predictions for gradient computation")
    parser.add_argument("--timestep-idx", type=int, default=None,
                        help="Which timestep to analyze (required if aggregation='timestep')")
    parser.add_argument("--action-dim-idx", type=int, default=None,
                        help="Which action dimension to analyze (required if aggregation='action_dim')")
    parser.add_argument("--n-integration-steps", type=int, default=20,
                        help="Number of steps for Integrated Gradients interpolation")
    parser.add_argument("--ig-batch-size", type=int, default=10,
                        help="Batch size for Integrated Gradients computation. Higher values use more memory but are faster (default: 10)")
    parser.add_argument("--normalization-mode", type=str, default="percentile",
                        choices=["linear", "percentile"],
                        help="Normalization mode: 'linear' for min-max, 'percentile' for percentile clipping (default: percentile)")
    parser.add_argument("--normalization-percentile", type=float, default=99.99,
                        help="Percentile to clip at when using percentile normalization (default: 95.0)")
    
    args = parser.parse_args()
    
    # Validate aggregation arguments
    if args.aggregation == 'timestep' and args.timestep_idx is None:
        parser.error("--timestep-idx required when aggregation='timestep'")
    if args.aggregation == 'action_dim' and args.action_dim_idx is None:
        parser.error("--action-dim-idx required when aggregation='action_dim'")
    
    # Set up device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16, 
        "bfloat16": torch.bfloat16
    }
    model_dtype = dtype_map[args.model_dtype]
    
    print(f"Loading dataset: {args.dataset_repo_id}")
    print(f"Policy path: {args.policy_path}")
    print(f"Using device: {device}")
    print(f"Integration steps: {args.n_integration_steps}")
    print(f"IG batch size: {args.ig_batch_size}")
    print(f"Normalization mode: {args.normalization_mode}")
    if args.normalization_mode == 'percentile':
        print(f"Normalization percentile: {args.normalization_percentile}")
    
    # Load dataset
    try:
        dataset = LeRobotDataset(args.dataset_repo_id)
        print(f"Dataset loaded successfully. Total episodes: {dataset.num_episodes}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Determine which episodes to analyze
    if args.episode_id is not None:
        # Single episode analysis
        if args.episode_id >= dataset.num_episodes:
            raise ValueError(f"Episode {args.episode_id} not found. Dataset has {dataset.num_episodes} episodes.")
        episodes_to_analyze = [args.episode_id]
        print(f"Target episode: {args.episode_id}")
    else:
        # All episodes analysis
        episodes_to_analyze = list(range(dataset.num_episodes))
        print(f"Will analyze all {dataset.num_episodes} episodes")
    
    # Load policy
    try:
        print("Loading policy...")
        train_dataset = make_dataset_without_config(TRAIN_DATASET_REPO_IDS)
        policy, policy_cfg = load_policy(
            args.policy_path,
            train_dataset.meta,
            args.policy_overrides,
            n_integration_steps=args.n_integration_steps,
            ig_batch_size=args.ig_batch_size,
            normalization_mode=args.normalization_mode,
            normalization_percentile=args.normalization_percentile
        )
        
        if hasattr(policy, 'model'):
            policy.model.eval()
            policy.model.to(device)
        elif hasattr(policy, 'eval'):
            policy.eval()
            
        print("Policy loaded successfully")
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run analysis on all specified episodes
    all_results = []
    failed_episodes = []
    
    for episode_id in tqdm(episodes_to_analyze, desc="Analyzing episodes"):
        try:
            print(f"\nStarting gradient-based analysis of episode {episode_id}...")
            results = analyze_episode(
                dataset=dataset,
                policy=policy,
                episode_id=episode_id,
                device=device,
                output_dir=args.output_dir,
                aggregation=args.aggregation,
                timestep_idx=args.timestep_idx,
                action_dim_idx=args.action_dim_idx,
                model_dtype=model_dtype
            )
            all_results.append(results)
            print(f"Episode {episode_id} analysis completed successfully")
            
        except Exception as e:
            print(f"Error analyzing episode {episode_id}: {e}")
            failed_episodes.append(episode_id)
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GRADIENT-BASED SALIENCY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully analyzed: {len(all_results)} episodes")
    if failed_episodes:
        print(f"Failed episodes: {len(failed_episodes)} ({failed_episodes})")
    else:
        print("No failed episodes")
    print(f"Aggregation method: {args.aggregation}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    register_third_party_plugins()
    main()
