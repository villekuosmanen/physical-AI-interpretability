import json
import logging
from pathlib import Path
from typing import Optional, Dict
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, hf_hub_download

from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robocandywrapper import make_dataset_without_config
from physical_ai_interpretability.utils import get_repo_hash
from physical_ai_interpretability.utils.naming import get_experiment_name, get_cache_name
from .config import SAETrainingConfig
from .token_sampler import TokenSamplerConfig
from .activation_collector import (
    collect_and_cache_activations,
    create_cached_dataloader,
    is_cache_valid,
    cleanup_invalid_cache,
    get_cache_status,
)
from .sae import create_multimodal_sae


def parse_repo_ids(repo_id: str) -> list[str]:
    """
    Parse repo_id string which can be either a single dataset or a bracketed list.
    
    Args:
        repo_id: Either "owner/dataset" or "[owner/ds1, owner/ds2, ...]"
        
    Returns:
        List of dataset repo IDs
    """
    if repo_id.startswith('['):
        datasets = repo_id.strip('[]').split(',')
        return [x.strip() for x in datasets if x.strip()]
    return [repo_id]


class SAETrainer():
    
    def __init__(
        self,
        repo_id: str,
        policy_path: Path,
        batch_size: int = 16,
        num_workers: int = 4,
        output_directory: Path = "output",
        resume_checkpoint: Optional[Path] = None, 
        activation_cache_path: str = str(Path.home() / ".cache" / "physical_ai_interpretability" / "sae_activations"),
        force_cache_refresh: bool = False,
        use_wandb: bool = False,
        wandb_project_name: str = "physical_ai_interpretability",
        sae_config: Optional[SAETrainingConfig] = None,
        # Hugging Face integration parameters
        upload_to_hub: bool = False,
        hub_repo_id: Optional[str] = None,
        hub_private: bool = True,
        hub_license: str = "mit",
        hub_tags: Optional[list] = None,
    ):
        # Load policy
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

        # Load dataset with proper delta indices
        # ACT models typically need current observation only
        dataset = make_dataset_without_config(
            repo_id=repo_id,
            action_delta_indices=list(range(policy_cfg.chunk_size)),
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

        ds_meta = dataset.meta if isinstance(dataset, LeRobotDataset) else dataset._datasets[0].meta
        self.policy = make_policy(policy_cfg, ds_meta=ds_meta)

        # Use provided config or create default
        self.config = sae_config if sae_config is not None else SAETrainingConfig()
        
        # Create token sampler config from SAE config
        token_sampler_config = TokenSamplerConfig(
            fixed_tokens=self.config.fixed_tokens,
            sampling_strategy=self.config.sampling_strategy,
            sampling_stride=self.config.sampling_stride,
            max_sampled_tokens=self.config.max_sampled_tokens,
            block_size=self.config.block_size
        ) if self.config.use_token_sampling else None
        
        # Auto-infer model parameters - will be updated later if using cached activations
        self.config.infer_model_params(self.policy, token_sampler_config)
        self._token_sampler_config = token_sampler_config  # Store for potential cache-based inference

        # Determine layer name from policy - pick last layer in encoder
        self.layer_name = "model.encoder.layers.3.norm2"  # Default layer
        if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'encoder'):
            if hasattr(self.policy.model.encoder, 'layers') and len(self.policy.model.encoder.layers) > 0:
                # Use the last layer's norm2 by default
                layer_idx = len(self.policy.model.encoder.layers) - 1
                self.layer_name = f"model.encoder.layers.{layer_idx}.norm2"

        # Store initialization parameters
        self.repo_id = repo_id
        self.output_directory = Path(output_directory)
        self.resume_checkpoint = resume_checkpoint
        self.force_cache_refresh = force_cache_refresh
        self.activation_cache_path = activation_cache_path
        self.use_wandb = use_wandb
        self.wandb_project_name = wandb_project_name

        # Store Hugging Face parameters
        self.upload_to_hub = upload_to_hub
        self.hub_repo_id = hub_repo_id
        self.hub_private = hub_private
        self.hub_license = hub_license
        self.hub_tags = hub_tags or ["sae", "sparse-autoencoder", "robotics", "out-of-distribution"]

        # Store token sampler config for activation collection
        self.token_sampler_config = token_sampler_config

        # Initialize wandb
        if use_wandb:
            experiment_name = get_experiment_name(repo_id, prefix="sae")
            self.wandb = wandb.init(
                project=wandb_project_name,
                name=experiment_name,
                config={
                    'repo_id': repo_id,
                    'repo_hash': get_repo_hash(repo_id),
                    'layer_name': self.layer_name,
                    'num_tokens': self.config.num_tokens,
                    'token_dim': self.config.token_dim,
                    'feature_dim': self.config.feature_dim,
                    'expansion_factor': self.config.expansion_factor,
                    **self.config.__dict__
                }
            )
        else:
            self.wandb = None

    def collect_activations(self):
        """Collect activations and return cached dataloader with resumption support"""
        cache_path = Path(self.activation_cache_path) / get_cache_name(self.repo_id)
        
        # Get detailed cache status
        cache_status = get_cache_status(str(cache_path))
        logging.info(f"Cache status: {cache_status['status']}")
        
        if cache_status['exists']:
            if cache_status['status'] == 'completed':
                logging.info(f"Found completed cache with {cache_status['total_samples']} samples")
            elif cache_status['status'] == 'in_progress' and cache_status['can_resume']:
                logging.info(f"Found resumable cache with {cache_status['total_samples']} samples")
                logging.info(f"Last batch: {cache_status['last_batch_idx']}, Interruptions: {cache_status['interruption_count']}")
            elif cache_status['status'] == 'in_progress':
                logging.warning("Found incomplete cache but it's not resumable")
        
        # Check if we should use existing cache
        use_existing_cache = (
            cache_status['status'] == 'completed' and not self.force_cache_refresh
        )
        
        if use_existing_cache:
            logging.info(f"Using existing valid activation cache at {cache_path}")
            try:
                # Update config with parameters from cache metadata
                logging.info("Updating model parameters from cached activation data...")
                self.config.infer_model_params_from_cache(str(cache_path), self._token_sampler_config)
                
                return create_cached_dataloader(
                    cache_dir=str(cache_path),
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    preload_buffers=2
                )
            except Exception as e:
                logging.warning(f"Failed to load existing cache: {e}")
                logging.info("Will recreate cache from scratch")
                # Fall through to cache recreation
        
        # Handle invalid or missing cache
        if cache_path.exists():
            if not is_cache_valid(str(cache_path)):
                logging.info(f"Cache at {cache_path} is invalid or incomplete, cleaning up")
                cleanup_invalid_cache(str(cache_path))
            elif self.force_cache_refresh:
                logging.info(f"Force refresh requested, cleaning existing cache at {cache_path}")
                cleanup_invalid_cache(str(cache_path))
        
        logging.info(f"Collecting new activations to {cache_path}")
        
        # Collect activations with token sampling
        try:
            cache_dir_str = collect_and_cache_activations(
                act_model=self.policy,
                dataloader=self.dataloader,
                layer_name=self.layer_name,
                cache_dir=str(cache_path.parent),
                experiment_name=cache_path.name,
                device=self.config.device,
                cleanup_on_start=self.force_cache_refresh,  # Force clean start if requested
                use_token_sampling=self.config.use_token_sampling,
                fixed_tokens=self.config.fixed_tokens if self.config.use_token_sampling else None,
                sampling_strategy=self.config.sampling_strategy if self.config.use_token_sampling else "uniform",
                sampling_stride=self.config.sampling_stride,
                max_sampled_tokens=self.config.max_sampled_tokens,
                block_size=self.config.block_size
            )
        except Exception as e:
            # Clean up any partial cache that might have been created
            if cache_path.exists():
                logging.warning(f"Activation collection failed, cleaning up partial cache: {e}")
                cleanup_invalid_cache(str(cache_path))
            raise
        
        # Verify the cache was created successfully
        if not is_cache_valid(cache_dir_str):
            logging.error(f"Cache creation completed but validation failed for {cache_dir_str}")
            cleanup_invalid_cache(cache_dir_str)
            raise RuntimeError("Failed to create valid activation cache")
        
        # Update config with actual parameters from the newly created cache
        logging.info("Updating model parameters from newly created cache data...")
        self.config.infer_model_params_from_cache(cache_dir_str, self._token_sampler_config)
        
        # Create dataloader from cached activations
        return create_cached_dataloader(
            cache_dir=cache_dir_str,
            batch_size=self.config.batch_size,
            shuffle=True,
            preload_buffers=2
        )

    def create_model(self) -> nn.Module:
        """Create SAE model based on config"""
        model = create_multimodal_sae(
            num_tokens=self.config.num_tokens,
            token_dim=self.config.token_dim,
            feature_dim=self.config.feature_dim,
            device=self.config.device
        )
        return model.to(self.config.device)
    
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
        """Single training step - returns both scalars and tensors"""
        batch = batch.to(self.config.device)
        
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
        """Save model checkpoint using safetensors for model weights"""
        
        # Save model weights with safetensors (secure and efficient)
        model_path = self.final_output_directory / f"model_epoch_{epoch}.safetensors"
        save_file(model.state_dict(), model_path)
        
        # Save training state with torch.save (optimizer/scheduler states need pickle)
        training_state = {
            'epoch': epoch,
            'global_step': getattr(self, 'global_step', 0),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': getattr(self, 'best_val_loss', float('inf')),
            'config': self.config.__dict__,
            'model_path': str(model_path)  # Reference to the safetensors model file
        }
        state_path = self.final_output_directory / f"training_state_epoch_{epoch}.pt"
        torch.save(training_state, state_path)
        
        logging.info(f"Saved checkpoint - Model: {model_path.name}, State: {state_path.name}")
        
        # Save best checkpoint
        if is_best:
            best_model_path = self.final_output_directory / "best_model.safetensors"
            best_state_path = self.final_output_directory / "best_training_state.pt"
            
            # Copy current best to dedicated best files
            save_file(model.state_dict(), best_model_path)
            training_state['model_path'] = str(best_model_path)
            torch.save(training_state, best_state_path)
            
            logging.info(f"Saved best checkpoint at epoch {epoch} - Model: {best_model_path.name}")

    def save_complete_model(self, model: nn.Module, epoch: int = None):
        """
        Save the complete model in a 'complete' folder ready for Hugging Face upload.
        This includes model.safetensors, config.json, and training_state.pt
        """
        # Create complete folder
        complete_dir = self.final_output_directory / "complete"
        complete_dir.mkdir(exist_ok=True)
        
        # Save model weights as model.safetensors (standard HF naming)
        model_path = complete_dir / "model.safetensors"
        save_file(model.state_dict(), model_path)
        
        # Copy or create config.json
        source_config = self.final_output_directory / "config.json"
        dest_config = complete_dir / "config.json"
        if source_config.exists():
            # Copy existing config
            import shutil
            shutil.copy2(source_config, dest_config)
        else:
            # Create minimal config
            config_dict = self.config.__dict__.copy()
            config_dict.update({
                'repo_id': self.repo_id,
                'repo_hash': get_repo_hash(self.repo_id),
                'layer_name': self.layer_name,
                'experiment_name': getattr(self, 'experiment_name', 'unknown')
            })
            with open(dest_config, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        # Save training state (if available)
        training_state_path = complete_dir / "training_state.pt"
        if epoch is not None:
            # Copy the training state from specific epoch
            source_state = self.final_output_directory / f"training_state_epoch_{epoch}.pt"
        else:
            # Use best training state if available
            source_state = self.final_output_directory / "best_training_state.pt"
            if not source_state.exists():
                # Find latest training state
                state_files = list(self.final_output_directory.glob("training_state_epoch_*.pt"))
                if state_files:
                    source_state = max(state_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        if source_state.exists():
            import shutil
            shutil.copy2(source_state, training_state_path)
        
        logging.info(f"Saved complete model to: {complete_dir}")
        return complete_dir

    def push_model_to_hub(self, complete_model_dir: Path):
        """
        Push the complete model to Hugging Face Hub
        """
        if not self.upload_to_hub:
            logging.info("Hub upload disabled, skipping...")
            return None
            
        if not self.hub_repo_id:
            raise ValueError("hub_repo_id must be specified to upload to Hub")
        
        api = HfApi()
        
        # Create repo
        repo_info = api.create_repo(
            repo_id=self.hub_repo_id, 
            private=self.hub_private, 
            exist_ok=True
        )
        
        logging.info(f"Created/accessed Hub repo: {repo_info.repo_id}")
        
        # Generate model card
        readme_content = self.generate_model_card()
        readme_path = complete_model_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Upload folder
        commit_info = api.upload_folder(
            repo_id=repo_info.repo_id,
            repo_type="model",
            folder_path=complete_model_dir,
            commit_message="Upload SAE model weights, config, and training state",
            allow_patterns=["*.safetensors", "*.json", "*.pt", "*.md"],
            ignore_patterns=["*.tmp", "*.log", "__pycache__/*"],
        )
        
        logging.info(f"Model pushed to Hub: {commit_info.repo_url.url}")
        return commit_info

    def generate_model_card(self) -> str:
        """Generate a model card for the SAE model"""
        # Generate YAML frontmatter
        yaml_tags = '\n'.join([f'- {tag}' for tag in self.hub_tags])
        
        # Parse repo_id to handle both single and multiple datasets
        dataset_list = parse_repo_ids(self.repo_id)
        yaml_datasets = '\n'.join([f'- {ds}' for ds in dataset_list])
        datasets_display = ', '.join([f'`{ds}`' for ds in dataset_list])
        
        card_content = f"""---
license: {self.hub_license}
tags:
{yaml_tags}
datasets:
{yaml_datasets}
library_name: physical-ai-interpretability
---

# Sparse Autoencoder (SAE) Model

This model is a Sparse Autoencoder trained for interpretability analysis of robotics policies using the LeRobot framework.

## Model Details

- **Architecture**: Multi-modal Sparse Autoencoder
- **Training Dataset**: {datasets_display}
- **Base Policy**: LeRobot ACT policy
- **Layer Target**: `{self.layer_name}`
- **Tokens**: {self.config.num_tokens}
- **Token Dimension**: {self.config.token_dim}
- **Feature Dimension**: {self.config.feature_dim}
- **Expansion Factor**: {self.config.expansion_factor}

## Training Configuration

- **Learning Rate**: {self.config.learning_rate}
- **Batch Size**: {self.config.batch_size}
- **L1 Penalty**: {self.config.l1_penalty}
- **Epochs**: {self.config.num_epochs}
- **Optimizer**: {self.config.optimizer}

## Usage

```python
from physical_ai_interpretability.sae.trainer import load_sae_from_hub

# Load model from Hub
model = load_sae_from_hub("{self.hub_repo_id}")

# Or load using builder
from physical_ai_interpretability.sae.builder import SAEBuilder
builder = SAEBuilder(device='cuda')
model = builder.load_from_hub("{self.hub_repo_id}")
```

## Out-of-Distribution Detection

This SAE model can be used for OOD detection with LeRobot policies:

```python
from physical_ai_interpretability.ood import OODDetector

# Create OOD detector with Hub-loaded SAE
ood_detector = OODDetector(
    policy=your_policy,
    sae_hub_repo_id="{self.hub_repo_id}"
)

# Fit threshold and use for detection
ood_detector.fit_ood_threshold_to_validation_dataset(validation_dataset)
is_ood, error = ood_detector.is_out_of_distribution(observation)
```

## Files

- `model.safetensors`: The trained SAE model weights
- `config.json`: Training and model configuration
- `training_state.pt`: Complete training state (optimizer, scheduler, metrics)
- `ood_params.json`: OOD detection parameters (if fitted)

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{sae_model,
  title={{Sparse Autoencoder for {dataset_list[0].split('/')[-1].replace('_', ' ').title()}}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/{self.hub_repo_id}}}
}}
```

## Framework

This model was trained using the [physical-ai-interpretability](https://github.com/your-repo/physical-ai-interpretability) framework with [LeRobot](https://github.com/huggingface/lerobot).
"""
        return card_content

    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer = None, 
                       scheduler = None, checkpoint_path: str = None, load_best: bool = False):
        """Load model checkpoint from safetensors format"""
        
        if checkpoint_path is None:
            if load_best:
                model_path = self.final_output_directory / "best_model.safetensors"
                state_path = self.final_output_directory / "best_training_state.pt"
            else:
                # Find the latest checkpoint
                model_files = list(self.final_output_directory.glob("model_epoch_*.safetensors"))
                if not model_files:
                    raise FileNotFoundError("No checkpoint files found")
                latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
                epoch_num = latest_model.stem.split('_')[-1]
                model_path = latest_model
                state_path = self.final_output_directory / f"training_state_epoch_{epoch_num}.pt"
        else:
            # Custom checkpoint path provided
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.suffix == '.safetensors':
                model_path = checkpoint_path
                # Try to find corresponding training state
                base_name = checkpoint_path.stem
                state_path = checkpoint_path.parent / f"training_state_{base_name.replace('model_', '')}.pt"
            else:
                # Legacy .pt format - load differently
                return self._load_legacy_checkpoint(model, optimizer, scheduler, checkpoint_path)
        
        # Load model weights from safetensors
        if model_path.exists():
            model_state = load_file(model_path)
            model.load_state_dict(model_state)
            logging.info(f"Loaded model weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load training state if available
        if state_path.exists() and (optimizer is not None or scheduler is not None):
            training_state = torch.load(state_path, map_location='cpu')
            
            if optimizer is not None and 'optimizer_state_dict' in training_state:
                optimizer.load_state_dict(training_state['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in training_state:
                scheduler.load_state_dict(training_state['scheduler_state_dict'])
            
            # Restore training state
            if hasattr(self, 'global_step'):
                self.global_step = training_state.get('global_step', 0)
            if hasattr(self, 'best_val_loss'):
                self.best_val_loss = training_state.get('best_val_loss', float('inf'))
            
            logging.info(f"Loaded training state from {state_path}")
            return training_state
        
        return None
    
    def _load_legacy_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer = None, 
                               scheduler = None, checkpoint_path: str = None):
        """Load legacy .pt checkpoint format"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        if hasattr(self, 'global_step'):
            self.global_step = checkpoint.get('global_step', 0)
        if hasattr(self, 'best_val_loss'):
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logging.info(f"Loaded legacy checkpoint from {checkpoint_path}")
        return checkpoint

    def train(self):
        """
        Main training method
        
        Returns:
            Trained SAE model
        """
        # Step 1: Collect activations
        cached_dataloader = self.collect_activations()
        if cached_dataloader is None:
            raise ValueError("No activations collected. Please check the cache path and try again.")
        
        logging.info(f"Activations collected and cached")

        # Step 2: Set up training
        logging.info("Starting SAE training")
        
        # Create output directory using trainer parameters
        experiment_name = get_experiment_name(self.repo_id, prefix="sae")
        output_dir = self.output_directory / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_directory = output_dir
        
        # Save config
        config_dict = self.config.__dict__.copy()
        # Add trainer-specific info
        config_dict.update({
            'repo_id': self.repo_id,
            'repo_hash': get_repo_hash(self.repo_id),
            'layer_name': self.layer_name,
            'activation_cache_path': self.activation_cache_path,
            'experiment_name': experiment_name
        })
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
                
        # Create model
        model = self.create_model()
        logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, cached_dataloader)
        
        # Handle checkpoint resuming
        start_epoch = 0
        if self.resume_checkpoint is not None:
            logging.info(f"Resuming training from checkpoint: {self.resume_checkpoint}")
            training_state = self.load_checkpoint(model, optimizer, scheduler, str(self.resume_checkpoint))
            if training_state:
                start_epoch = training_state.get('epoch', 0) + 1
                logging.info(f"Resuming from epoch {start_epoch}")
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        self.global_step = global_step
        self.best_val_loss = best_val_loss
        
        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            model.train()
            epoch_metrics = {}
            
            # Training
            progress_bar = tqdm(cached_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in progress_bar:
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
                global_step += 1
                self.global_step = global_step
                
                # Logging
                if global_step % self.config.log_every == 0:
                    avg_metrics = {k: np.mean(v[-100:]) for k, v in epoch_metrics.items()}
                    
                    logging.info(
                        f"Step {global_step}, Epoch {epoch+1}, "
                        f"Loss: {avg_metrics['total_loss']:.4f}, "
                        f"MSE: {avg_metrics['mse_loss']:.4f}, "
                        f"R²: {avg_metrics.get('r_squared', 0):.4f}"
                    )
                    
                    if self.use_wandb and self.wandb is not None:
                        self.wandb.log({
                            f"train/{k}": v for k, v in avg_metrics.items()
                        }, step=global_step)
                                
                # Periodic saving
                if global_step % self.config.save_every == 0:
                    self.save_checkpoint(model, optimizer, scheduler, epoch)
            
            # End of epoch
            avg_epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logging.info(f"End of epoch {epoch+1}: {avg_epoch_metrics}")
        
        # Final save
        final_epoch = self.config.num_epochs - 1
        self.save_checkpoint(model, optimizer, scheduler, final_epoch)
        
        # Save complete model for potential Hub upload
        complete_dir = self.save_complete_model(model, final_epoch)
        
        # Upload to Hub if requested
        if self.upload_to_hub:
            try:
                commit_info = self.push_model_to_hub(complete_dir)
                if commit_info:
                    logging.info(f"Successfully uploaded model to Hub: {commit_info.repo_url.url}")
            except Exception as e:
                logging.error(f"Failed to upload model to Hub: {e}")
                logging.info("Model training completed successfully, but Hub upload failed")
        
        logging.info("Training completed!")
        return model


def load_sae_from_hub(
    repo_id: str, 
    filename: str = "model.safetensors",
    config_filename: str = "config.json",
    revision: str = "main",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    token: Optional[str] = None,
    device: str = 'cuda'
):
    """
    Load SAE model from Hugging Face Hub
    
    Args:
        repo_id: Repository ID on Hugging Face Hub
        filename: Model filename to download
        config_filename: Config filename to download  
        revision: Git revision (branch/tag/commit)
        cache_dir: Local cache directory
        force_download: Force re-download even if cached
        token: Hugging Face token for private repos
        device: Device to load model on
        
    Returns:
        Loaded SAE model
    """
    from physical_ai_interpretability.sae import create_multimodal_sae
    
    # Download model file
    model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
    )
    
    # Download config file
    config_file = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
    )
    
    # Load config
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # Extract model parameters
    num_tokens = config_dict.get('num_tokens')
    token_dim = config_dict.get('token_dim') 
    feature_dim = config_dict.get('feature_dim')
    
    if any(param is None for param in [num_tokens, token_dim, feature_dim]):
        raise ValueError("Config file missing required parameters: num_tokens, token_dim, feature_dim")
    
    # Create model
    model = create_multimodal_sae(
        num_tokens=num_tokens,
        token_dim=token_dim,
        feature_dim=feature_dim,
        device=device
    )
    
    # Load weights from safetensors
    model_state = load_file(model_file)
    model.load_state_dict(model_state)
    model.eval()
    
    logging.info(f"Loaded SAE model from Hub: {repo_id}")
    logging.info(f"Model config: {num_tokens} tokens, {token_dim} dim, {feature_dim} features")
    
    return model


def load_sae_model(model_path: str, config_path: str = None, device: str = 'cuda'):
    """
    Standalone function to load a trained SAE model from safetensors checkpoint
    
    Args:
        model_path: Path to the safetensors model file
        config_path: Optional path to config.json file. If None, tries to find it automatically
        device: Device to load the model on
        
    Returns:
        Loaded SAE model
    """
    from physical_ai_interpretability.sae import create_multimodal_sae
    
    model_path = Path(model_path)
    
    # Try to find config automatically if not provided
    if config_path is None:
        # Look for config.json in the same directory or parent directory
        potential_configs = [
            model_path.parent / "config.json",
            model_path.parent.parent / "config.json"
        ]
        config_path = None
        for potential_config in potential_configs:
            if potential_config.exists():
                config_path = potential_config
                break
    
    if config_path is None:
        raise FileNotFoundError("Could not find config.json file. Please provide config_path explicitly.")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Extract model parameters
    num_tokens = config_dict.get('num_tokens')
    token_dim = config_dict.get('token_dim')
    feature_dim = config_dict.get('feature_dim')
    activation_fn = config_dict.get('activation_fn', 'relu')
    
    if any(param is None for param in [num_tokens, token_dim, feature_dim]):
        raise ValueError("Config file missing required parameters: num_tokens, token_dim, feature_dim")
    
    # Create model
    model = create_multimodal_sae(
        num_tokens=num_tokens,
        token_dim=token_dim,
        feature_dim=feature_dim,
        device=device
    )
    
    # Load weights from safetensors
    model_state = load_file(model_path)
    model.load_state_dict(model_state)
    model.eval()
    
    logging.info(f"Loaded SAE model from {model_path}")
    logging.info(f"Model config: {num_tokens} tokens, {token_dim} dim, {feature_dim} features")
    
    return model
