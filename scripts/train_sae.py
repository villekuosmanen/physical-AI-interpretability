import argparse
import logging
from pathlib import Path

from src.sae import SAETrainer, SAETrainingConfig


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
    
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
    )
    
    # Train the model
    sae_model = trainer.train()
    
    logging.info("SAE training completed successfully!")
    return sae_model


if __name__ == "__main__":
    main()
