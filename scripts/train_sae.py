import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from lerobot.common.datasets.factory import make_dataset_without_config
from lerobot.common.datasets.utils import dataloader_collate_fn
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

from src.feature_extraction import (
    create_multimodal_sae,
    prepare_batch_for_bfloat16,
    collect_and_cache_activations,
    create_cached_dataloader)


@dataclass
class SAETrainingConfig:
    """Configuration for SAE training"""
    # Model config
    num_tokens: int = 77
    token_dim: int = 128
    feature_dim: int = 12320 # 1.25x expansion factor by default
    activation_fn: str = 'relu'  # 'tanh', 'relu', 'leaky_relu'
    
    # Training config
    batch_size: int = 128
    learning_rate: float = 1e-4
    num_epochs: int = 20
    validation_split: float = 0.1
    
    # Loss config
    l1_penalty: float = 0.3
    
    # Optimization config
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    weight_decay: float = 1e-5
    lr_schedule: str = 'constant'  # 'cosine', 'linear', 'constant'
    warmup_epochs: int = 2
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-5
    
    # Logging and saving
    log_every: int = 5
    save_every: int = 1000
    validate_every: int = 500
    output_dir: str = "./output/sae_checkpoints"
    experiment_name: str = "multimodal_sae"
    use_wandb: bool = True
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class SAETrainer:
    """
    Trainer for Multi-Modal Sparse Autoencoders
    """
    
    def __init__(self, config: SAETrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project="physical-ai-interpretability",
                name=config.experiment_name,
                config=config.__dict__
            )
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
    
    def create_model(self) -> nn.Module:
        """Create SAE model based on config"""
        model = create_multimodal_sae(
            num_tokens=self.config.num_tokens,
            token_dim=self.config.token_dim,
            feature_dim=self.config.feature_dim,
            device=self.config.device
        )
        return model.to(self.device)
    
    def create_optimizer_and_scheduler(self, model: nn.Module, train_loader: DataLoader):
        """Create optimizer and learning rate scheduler"""
        # Optimizer
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs
        
        if self.config.lr_schedule == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        elif self.config.lr_schedule == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=warmup_steps
            )
        else:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer)
        
        return optimizer, scheduler
        
    def train_step(self, model: nn.Module, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step - now returns both scalars and tensors"""
        batch = batch.to(self.device)
        
        # Forward pass - keep the tensor version for backprop
        loss_dict_tensors = model.compute_loss(
            batch,
            l1_penalty=self.config.l1_penalty,
        )
        
        # Convert to scalars for logging, but keep tensor versions
        loss_dict_scalars = {k: v.item() if torch.is_tensor(v) else v 
                            for k, v in loss_dict_tensors.items()}
        
        return loss_dict_scalars, loss_dict_tensors
        
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        save_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best checkpoint at epoch {epoch}")
    
    def train(self, train_loader: DataLoader) -> nn.Module:
        """
        Main training loop
        
        Args:
            activations: Collected activations from ACT model
            
        Returns:
            Trained SAE model
        """
        logging.info("Starting SAE training")
                
        # Create model
        model = self.create_model()
        logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, train_loader)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            model.train()
            epoch_metrics = {}
            
            # Training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in progress_bar:
                if hasattr(model, 'use_bfloat16') and model.use_bfloat16:
                    batch = prepare_batch_for_bfloat16(batch, self.device)
                else:
                    batch = batch.to(self.device)
                
                # Single forward pass that returns both scalars and tensors
                loss_dict_scalars, loss_dict_tensors = self.train_step(model, batch)
                
                # Backward pass using the tensor version
                optimizer.zero_grad()
                loss_dict_tensors['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config.gradient_clip_norm
                )
                
                optimizer.step()
                scheduler.step()

                # Update metrics
                for key, value in loss_dict_scalars.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
                
                # Use scalar version for logging/progress bar updates
                progress_bar.set_postfix({
                    'loss': f"{loss_dict_scalars['total_loss']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    avg_metrics = {k: np.mean(v[-100:]) for k, v in epoch_metrics.items()}
                    
                    logging.info(
                        f"Step {self.global_step}, Epoch {epoch+1}, "
                        f"Loss: {avg_metrics['total_loss']:.4f}, "
                        f"MSE: {avg_metrics['mse_loss']:.4f}, "
                        f"R²: {avg_metrics.get('r_squared', 0):.4f}"
                    )
                    
                    if self.config.use_wandb:
                        wandb.log({
                            f"train/{k}": v for k, v in avg_metrics.items()
                        }, step=self.global_step)
                                
                # Periodic saving
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(model, optimizer, scheduler, epoch)
            
            # End of epoch
            avg_epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logging.info(f"End of epoch {epoch+1}: {avg_epoch_metrics}")
        
        # Final save
        self.save_checkpoint(model, optimizer, scheduler, self.config.num_epochs - 1)
        
        logging.info("Training completed!")
        return model

def train_sae_on_cached_activations(
    dataloader: DataLoader,
    config: Optional[SAETrainingConfig] = None
) -> nn.Module:
    """
    Train SAE using cached activations
    
    Args:
        cache_dir: Directory containing cached activations
        config: Training configuration
        
    Returns:
        Trained SAE model
    """
    if config is None:
        config = SAETrainingConfig()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
        
    # Train SAE
    trainer = SAETrainer(config)
    sae_model = trainer.train(dataloader)
    
    return sae_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train safety classifier")
    parser.add_argument("--repo-id", type=str, 
                        help="Dataset repo ID for training")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--use-activation-cache", action="store_true", help="use cached activations to skip data loading and go straight to training")
    args = parser.parse_args()

    # Example usage
    config = SAETrainingConfig()
    
    if args.use_activation_cache:
        cache_path = "./output/activation_cache/demo_activations"
    else:
        # load dataset and create data loader
        # TODO: this part currently only works on Ville's LeRobot fork - add support to current `main` branch
        chunk_size = 100    # defaults to 100 in LeRobot
        dataset = make_dataset_without_config(
            repo_id=args.repo_id,
            action_delta_indices=list(range(chunk_size)),
            observation_delta_indices=None,
        )
        dataloader = DataLoader(
            dataset,
            num_workers=4,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=dataloader_collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        # Load your pre-trained ACT model and dataloader
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        policy_cfg.pretrained_path = args.policy_path
        policy = make_policy(policy_cfg, ds_meta=dataset._datasets[0].meta)

        cache_path = collect_and_cache_activations(
            act_model=policy,
            dataloader=dataloader,
            layer_name="model.encoder.layers.3.norm2",
            experiment_name="demo_activations",
            buffer_size=4096,
        )

    cached_dataloader = create_cached_dataloader(
        cache_dir=cache_path,
        batch_size=128,
        shuffle=False,
        preload_buffers=2
    )
    
    sae_model = train_sae_on_cached_activations(cached_dataloader, config)
