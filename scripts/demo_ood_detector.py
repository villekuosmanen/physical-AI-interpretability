#!/usr/bin/env python

"""
Demo script showing how to use the OOD detector with SAE models.
This script demonstrates:
1. Loading a trained SAE model using the builder
2. Creating an OOD detector
3. Fitting the OOD threshold on a validation dataset
4. Testing OOD detection on individual observations
"""

import argparse
import logging
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

from src.ood import OODDetector, create_default_ood_params_path
from src.utils import make_dataset_without_config


def main():
    parser = argparse.ArgumentParser(description="Demo OOD detector functionality")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--sae-experiment-path", type=str, required=True,
                        help="Path to SAE experiment directory (e.g., output/sae_drop_footbag_into_di_838a8c8b)")
    parser.add_argument("--validation-repo-id", type=str, required=True,
                        help="Repository ID of validation dataset for threshold fitting")
    parser.add_argument("--test-repo-id", type=str,
                        help="Repository ID of test dataset for OOD detection (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max-validation-samples", type=int, default=1000,
                        help="Maximum number of validation samples for threshold fitting")
    parser.add_argument("--std-threshold", type=float, default=3.0,
                        help="Number of standard deviations for OOD threshold")
    parser.add_argument("--force-ood-refresh", action="store_true",
                        help="Force refresh of OOD parameters (ignore existing cache)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    device = torch.device(args.device)
    
    print("=" * 60)
    print("OOD Detector Demo")
    print("=" * 60)
    print(f"Policy path: {args.policy_path}")
    print(f"SAE experiment path: {args.sae_experiment_path}")
    print(f"Validation dataset: {args.validation_repo_id}")
    print(f"Device: {device}")
    print()
    
    # Step 1: Load policy
    print("1. Loading policy...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    
    # Load validation dataset to get metadata
    validation_dataset = make_dataset_without_config(
        repo_id=args.validation_repo_id,
        action_delta_indices=list(range(policy_cfg.chunk_size)),
    )
    
    ds_meta = validation_dataset.meta if hasattr(validation_dataset, 'meta') else validation_dataset._datasets[0].meta
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()
    
    print(f"✓ Policy loaded successfully")
    
    # Step 2: Create OOD detector (will automatically load SAE)
    print("\n2. Creating OOD detector...")
    # Create path for OOD parameters
    experiment_name = Path(args.sae_experiment_path).name
    ood_params_path = create_default_ood_params_path(experiment_name)
    
    ood_detector = OODDetector(
        policy=policy,
        sae_experiment_path=args.sae_experiment_path,
        ood_params_path=ood_params_path,
        force_ood_refresh=True,
        device=args.device,
    )
        
    print(f"✓ OOD detector created successfully")
    print(f"  SAE model loaded from: {args.sae_experiment_path}")
    print(f"  OOD params path: {ood_params_path}")
    print(f"  Force OOD refresh: {args.force_ood_refresh}")
    print(f"  OOD params status: {'Will be refreshed' if args.force_ood_refresh else ('Loaded from cache' if ood_detector.ood_params else 'Not found, will be fitted')}")
        
    
    # Step 3: Fit OOD threshold on validation data (if needed)
    if ood_detector.needs_ood_fitting():
        print(f"\n3. Fitting OOD threshold on validation dataset...")
        ood_params = ood_detector.fit_ood_threshold_to_validation_dataset(
            dataset=validation_dataset,
            # std_threshold=args.std_threshold,
            max_samples=args.max_validation_samples,
        )
    else:
        print(f"\n3. Using existing OOD threshold from cache...")
        ood_params = ood_detector.get_ood_stats()
    
    print(f"✓ OOD threshold fitted successfully:")
    print(f"  Mean reconstruction error: {ood_params['mean']:.6f}")
    print(f"  Standard deviation: {ood_params['std']:.6f}")
    print(f"  OOD threshold ({args.std_threshold}σ): {ood_params['threshold']:.6f}")
    print(f"  Validation samples used: {ood_params['num_samples']}")
    print(f"  99th percentile: {ood_params['percentiles']['99']:.6f}")
    
    # Step 4: Test on some validation samples
    print(f"\n4. Testing OOD detection on validation samples...")
    test_samples = min(10, len(validation_dataset))
    ood_count = 0
    
    for i in range(test_samples):
        # Get sample from dataset
        sample = validation_dataset[i]
        
        # Convert to observation format
        observation = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                observation[key] = value.unsqueeze(0).to(device)
            else:
                observation[key] = value
        
        # Test OOD detection
        is_ood, recon_error = ood_detector.is_out_of_distribution(observation)
        if is_ood:
            ood_count += 1
        
        print(f"  Sample {i+1}: recon_error={recon_error:.6f}, OOD={is_ood}")
        
    print(f"\n  Summary: {ood_count}/{test_samples} validation samples flagged as OOD")
    print(f"  (Note: Low OOD rate on validation data is expected)")
    
    # Step 5: Test on different dataset if provided
    if args.test_repo_id:
        print(f"\n5. Testing on different dataset: {args.test_repo_id}")
        test_dataset = make_dataset_without_config(
            repo_id=args.test_repo_id,
            action_delta_indices=list(range(policy_cfg.chunk_size)),
        )
        
        test_samples = min(20, len(test_dataset))
        ood_count = 0
        
        for i in range(test_samples):
            sample = test_dataset[i]
            
            # Convert to observation format
            observation = {}
            for key, value in sample.items():
                if torch.is_tensor(value):
                    observation[key] = value.unsqueeze(0).to(device)
                else:
                    observation[key] = value
            
            # Test OOD detection
            is_ood, recon_error = ood_detector.is_out_of_distribution(observation)
            if is_ood:
                ood_count += 1
            
            if i < 5:  # Show first 5 samples in detail
                print(f"  Sample {i+1}: recon_error={recon_error:.6f}, OOD={is_ood}")
        
        print(f"\n  Summary: {ood_count}/{test_samples} test samples flagged as OOD")
        print(f"  OOD rate: {100*ood_count/test_samples:.1f}%")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"OOD parameters saved to: {ood_params_path}")
    print("You can now use this OOD detector in your applications.")
    print("="*60)


if __name__ == "__main__":
    main()
