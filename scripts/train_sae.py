import argparse
import logging
from pathlib import Path

from physical_ai_interpretability.sae import SAETrainer, SAETrainingConfig


def main():
    """Main function to train SAE using the new trainer class"""
    parser = argparse.ArgumentParser(description="Train SAE using the new trainer class")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Dataset repo ID for training")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="physical_ai_interpretability", help="W&B project name")
    parser.add_argument("--activation-cache-path", type=str, help="Path to cache activations")
    parser.add_argument("--force-cache-refresh", action="store_true", help="Force refresh of activation cache")
    parser.add_argument("--expansion-factor", type=float, default=1.25, help="Feature expansion factor for SAE")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--l1-penalty", type=float, default=0.3, help="L1 penalty for sparsity")
    
    # Hugging Face Hub arguments
    parser.add_argument("--upload-to-hub", action="store_true", help="Upload trained model to Hugging Face Hub")
    parser.add_argument("--hub-repo-id", type=str, help="Hugging Face Hub repository ID (required if --upload-to-hub)")
    parser.add_argument("--hub-private", action="store_true", help="Make Hub repository private")
    parser.add_argument("--hub-license", type=str, default="mit", help="License for Hub repository")
    parser.add_argument("--hub-tags", type=str, nargs="*", 
                        default=["physical-ai-interpretability-sae", "LeRobot", "Robotics"],
                        help="Tags for Hub repository")
    
    args = parser.parse_args()
    
    # Validate Hub arguments
    if args.upload_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required when --upload-to-hub is specified")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log configuration
    logging.info("Starting SAE training with configuration:")
    logging.info(f"  Dataset: {args.repo_id}")
    logging.info(f"  Policy: {args.policy_path}")
    logging.info(f"  Epochs: {args.num_epochs}")
    logging.info(f"  Learning rate: {args.learning_rate}")
    logging.info(f"  L1 penalty: {args.l1_penalty}")
    if args.upload_to_hub:
        logging.info(f"  Hub upload: ENABLED")
        logging.info(f"  Hub repo: {args.hub_repo_id}")
        logging.info(f"  Private repo: {args.hub_private}")
    else:
        logging.info(f"  Hub upload: DISABLED")

    # Create SAE config with command line overrides
    sae_config = SAETrainingConfig(
        expansion_factor=args.expansion_factor,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        l1_penalty=args.l1_penalty,
        batch_size=args.batch_size,
    )

    # Create trainer using the new API
    trainer = SAETrainer(
        repo_id=args.repo_id,
        policy_path=Path(args.policy_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_directory=Path(args.output_dir),
        resume_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        activation_cache_path=args.activation_cache_path or str(Path.home() / ".cache" / "physical_ai_interpretability" / "sae_activations"),
        force_cache_refresh=args.force_cache_refresh,
        use_wandb=args.use_wandb,
        wandb_project_name=args.wandb_project,
        sae_config=sae_config,
        # Hugging Face Hub parameters
        upload_to_hub=args.upload_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_private=args.hub_private,
        hub_license=args.hub_license,
        hub_tags=args.hub_tags,
    )
    
    # Train the model
    sae_model = trainer.train()
    
    logging.info("SAE training completed successfully!")
    if args.upload_to_hub:
        logging.info(f"Model uploaded to Hub: {args.hub_repo_id}")
        logging.info("You can now use this model by loading from Hub:")
        logging.info(f"  from physical_ai_interpretability.sae.trainer import load_sae_from_hub")
        logging.info(f"  model = load_sae_from_hub('{args.hub_repo_id}')")
    else:
        logging.info("Local training completed. Use --upload-to-hub to share your model!")
    
    return sae_model


if __name__ == "__main__":
    main()
