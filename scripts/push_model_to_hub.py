#!/usr/bin/env python3
"""
Script to upload existing SAE model checkpoints to Hugging Face Hub.

This script allows you to upload previously trained SAE models to Hugging Face Hub
without needing to retrain them. It handles both individual checkpoint files and
complete checkpoint directories.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from safetensors.torch import load_file

from physical_ai_interpretability.utils import get_repo_hash


def find_model_files(checkpoint_path: Path) -> Dict[str, Optional[Path]]:
    """
    Find model files in the checkpoint directory or file.
    
    Returns:
        Dictionary with paths to model, config, and training_state files
    """
    files = {
        'model': None,
        'config': None,
        'training_state': None
    }
    
    if checkpoint_path.is_file():
        # Single file provided - assume it's the model file
        if checkpoint_path.suffix == '.safetensors':
            files['model'] = checkpoint_path
            # Look for config in same directory
            config_path = checkpoint_path.parent / 'config.json'
            if config_path.exists():
                files['config'] = config_path
            # Look for training state
            if 'best_model' in checkpoint_path.name:
                state_path = checkpoint_path.parent / 'best_training_state.pt'
            else:
                # Extract epoch number if present
                stem = checkpoint_path.stem
                if 'epoch_' in stem:
                    epoch_num = stem.split('epoch_')[-1]
                    state_path = checkpoint_path.parent / f'training_state_epoch_{epoch_num}.pt'
                else:
                    # Look for any training state file
                    state_files = list(checkpoint_path.parent.glob('training_state*.pt'))
                    state_path = state_files[0] if state_files else None
            
            if state_path and state_path.exists():
                files['training_state'] = state_path
        else:
            raise ValueError(f"Unsupported model file format: {checkpoint_path.suffix}")
    
    elif checkpoint_path.is_dir():
        # Directory provided - look for files
        
        # Look for model files (prefer best, then latest epoch)
        model_candidates = [
            checkpoint_path / 'best_model.safetensors',
            checkpoint_path / 'model.safetensors',
        ]
        
        # Add epoch-specific models
        epoch_models = sorted(
            checkpoint_path.glob('model_epoch_*.safetensors'),
            key=lambda x: int(x.stem.split('_')[-1]),
            reverse=True
        )
        model_candidates.extend(epoch_models)
        
        for candidate in model_candidates:
            if candidate.exists():
                files['model'] = candidate
                break
        
        # Look for config
        config_path = checkpoint_path / 'config.json'
        if config_path.exists():
            files['config'] = config_path
        
        # Look for training state (prefer best, then match model epoch)
        if files['model'] and 'best_model' in files['model'].name:
            state_path = checkpoint_path / 'best_training_state.pt'
        elif files['model'] and 'epoch_' in files['model'].name:
            epoch_num = files['model'].stem.split('_')[-1]
            state_path = checkpoint_path / f'training_state_epoch_{epoch_num}.pt'
        else:
            # Look for any training state
            state_files = sorted(
                checkpoint_path.glob('training_state*.pt'),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            state_path = state_files[0] if state_files else None
        
        if state_path and state_path.exists():
            files['training_state'] = state_path
    
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    return files


def create_complete_model_directory(
    model_files: Dict[str, Optional[Path]], 
    output_dir: Path,
    repo_id: Optional[str] = None,
    layer_name: Optional[str] = None
) -> Path:
    """
    Create a complete model directory ready for Hub upload.
    
    Args:
        model_files: Dictionary with paths to model files
        output_dir: Directory to create the complete model in
        repo_id: Optional repo_id to include in config
        layer_name: Optional layer_name to include in config
        
    Returns:
        Path to the complete model directory
    """
    complete_dir = output_dir / "complete"
    complete_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model file as model.safetensors (standard HF naming)
    if not model_files['model']:
        raise ValueError("No model file found in checkpoint")
    
    model_dest = complete_dir / "model.safetensors"
    shutil.copy2(model_files['model'], model_dest)
    logging.info(f"Copied model: {model_files['model']} -> {model_dest}")
    
    # Handle config file
    config_dest = complete_dir / "config.json"
    if model_files['config']:
        # Copy existing config and potentially update it
        with open(model_files['config'], 'r') as f:
            config_dict = json.load(f)
        
        # Add missing fields if provided
        if repo_id and 'repo_id' not in config_dict:
            config_dict['repo_id'] = repo_id
            config_dict['repo_hash'] = get_repo_hash(repo_id)
        
        if layer_name and 'layer_name' not in config_dict:
            config_dict['layer_name'] = layer_name
        
        with open(config_dest, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logging.info(f"Updated and copied config: {model_files['config']} -> {config_dest}")
    else:
        # Try to infer config from model file
        logging.warning("No config.json found. Attempting to infer from model weights...")
        try:
            model_state = load_file(model_files['model'])
            
            # Infer dimensions from model weights
            # This is a basic inference - may need adjustment based on actual SAE structure
            encoder_weight = None
            decoder_weight = None
            
            for key, tensor in model_state.items():
                if 'encoder' in key.lower() and 'weight' in key:
                    encoder_weight = tensor
                elif 'decoder' in key.lower() and 'weight' in key:
                    decoder_weight = tensor
            
            if encoder_weight is not None and decoder_weight is not None:
                # Assuming encoder: [feature_dim, token_dim * num_tokens]
                # and decoder: [token_dim * num_tokens, feature_dim]
                feature_dim = encoder_weight.shape[0]
                token_dim_times_tokens = encoder_weight.shape[1]
                
                # This is a guess - you might need to adjust based on your SAE structure
                # For now, assume token_dim = 256 (common value)
                token_dim = 256
                num_tokens = token_dim_times_tokens // token_dim
                
                config_dict = {
                    'num_tokens': num_tokens,
                    'token_dim': token_dim,
                    'feature_dim': feature_dim,
                    'expansion_factor': feature_dim / token_dim_times_tokens,
                    'activation_fn': 'relu',  # Default assumption
                }
                
                if repo_id:
                    config_dict['repo_id'] = repo_id
                    config_dict['repo_hash'] = get_repo_hash(repo_id)
                
                if layer_name:
                    config_dict['layer_name'] = layer_name
                
                with open(config_dest, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                logging.info(f"Created inferred config: {config_dest}")
            else:
                raise ValueError("Could not infer model dimensions from weights")
                
        except Exception as e:
            logging.error(f"Failed to infer config: {e}")
            raise ValueError("No config.json found and could not infer from model. Please provide config manually.")
    
    # Copy training state if available
    if model_files['training_state']:
        state_dest = complete_dir / "training_state.pt"
        shutil.copy2(model_files['training_state'], state_dest)
        logging.info(f"Copied training state: {model_files['training_state']} -> {state_dest}")
    
    return complete_dir


def generate_model_card(
    config_path: Path,
    hub_repo_id: str,
    hub_license: str = "mit",
    hub_tags: list = None
) -> str:
    """Generate a model card for the SAE model"""
    
    # Load config to get model details
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    if hub_tags is None:
        hub_tags = ["sae", "sparse-autoencoder", "robotics", "physical-ai-interpretability"]
    
    # Generate YAML frontmatter
    yaml_tags = '\n'.join([f'- {tag}' for tag in hub_tags])
    
    repo_id = config_dict.get('repo_id', 'unknown')
    layer_name = config_dict.get('layer_name', 'unknown')
    num_tokens = config_dict.get('num_tokens', 'unknown')
    token_dim = config_dict.get('token_dim', 'unknown')
    feature_dim = config_dict.get('feature_dim', 'unknown')
    expansion_factor = config_dict.get('expansion_factor', 'unknown')
    
    # Format datasets properly for YAML frontmatter
    if isinstance(repo_id, str) and repo_id.startswith('[') and repo_id.endswith(']'):
        # Handle multiple datasets: "[dataset1, dataset2]" -> ["dataset1", "dataset2"]
        datasets_str = repo_id.strip('[]')
        datasets = [ds.strip() for ds in datasets_str.split(',')]
        yaml_datasets = '\n'.join([f'- {ds}' for ds in datasets])
    else:
        # Handle single dataset
        yaml_datasets = f'- {repo_id}'
    
    card_content = f"""---
license: {hub_license}
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
- **Training Dataset**: `{repo_id}`
- **Base Policy**: LeRobot ACT policy
- **Layer Target**: `{layer_name}`
- **Tokens**: {num_tokens}
- **Token Dimension**: {token_dim}
- **Feature Dimension**: {feature_dim}
- **Expansion Factor**: {expansion_factor}

## Training Configuration

- **Learning Rate**: {config_dict.get('learning_rate', 'unknown')}
- **Batch Size**: {config_dict.get('batch_size', 'unknown')}
- **L1 Penalty**: {config_dict.get('l1_penalty', 'unknown')}
- **Epochs**: {config_dict.get('num_epochs', 'unknown')}
- **Optimizer**: {config_dict.get('optimizer', 'unknown')}

## Usage

```python
from physical_ai_interpretability.sae import load_sae_from_hub

# Load model from Hub
model = load_sae_from_hub("{hub_repo_id}")

# Or load using builder
from physical_ai_interpretability.sae import SAEBuilder
builder = SAEBuilder(device='cuda')
model = builder.load_from_hub("{hub_repo_id}")
```

## Out-of-Distribution Detection

This SAE model can be used for OOD detection with LeRobot policies:

```python
from physical_ai_interpretability.ood import OODDetector

# Create OOD detector with Hub-loaded SAE
ood_detector = OODDetector(
    policy=your_policy,
    sae_hub_repo_id="{hub_repo_id}"
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
  title={{Sparse Autoencoder for {repo_id.split('/')[-1].replace('_', ' ').title() if '/' in repo_id else repo_id.replace('_', ' ').title()}}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/{hub_repo_id}}}
}}
```

## Framework

This model was trained using the [physical-ai-interpretability](https://github.com/your-repo/physical-ai-interpretability) framework with [LeRobot](https://github.com/huggingface/lerobot).
"""
    return card_content


def push_model_to_hub(
    complete_model_dir: Path,
    hub_repo_id: str,
    hub_private: bool = True,
    hub_license: str = "mit",
    hub_tags: list = None,
    commit_message: str = "Upload SAE model weights, config, and training state"
):
    """
    Push the complete model to Hugging Face Hub
    """
    api = HfApi()
    
    # Create repo
    repo_info = api.create_repo(
        repo_id=hub_repo_id, 
        private=hub_private, 
        exist_ok=True
    )
    
    logging.info(f"Created/accessed Hub repo: {repo_info.repo_id}")
    
    # Generate model card
    config_path = complete_model_dir / "config.json"
    readme_content = generate_model_card(
        config_path=config_path,
        hub_repo_id=hub_repo_id,
        hub_license=hub_license,
        hub_tags=hub_tags
    )
    readme_path = complete_model_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Upload folder
    commit_info = api.upload_folder(
        repo_id=repo_info.repo_id,
        repo_type="model",
        folder_path=complete_model_dir,
        commit_message=commit_message,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.md"],
        ignore_patterns=["*.tmp", "*.log", "__pycache__/*"],
    )
    
    logging.info(f"Model pushed to Hub: {commit_info.repo_url.url}")
    return commit_info


def main():
    """Main function to upload existing SAE checkpoint to Hugging Face Hub"""
    parser = argparse.ArgumentParser(
        description="Upload existing SAE model checkpoint to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a specific model file
  python push_model_to_hub.py --checkpoint /path/to/best_model.safetensors --hub-repo-id username/my-sae-model

  # Upload from a checkpoint directory
  python push_model_to_hub.py --checkpoint /path/to/checkpoint_dir --hub-repo-id username/my-sae-model

  # Upload with additional metadata
  python push_model_to_hub.py --checkpoint /path/to/model.safetensors \\
    --hub-repo-id username/my-sae-model \\
    --repo-id lerobot/aloha_sim_insertion_human \\
    --layer-name model.encoder.layers.3.norm2 \\
    --hub-tags sae robotics aloha
        """
    )
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.safetensors) or directory containing checkpoints")
    parser.add_argument("--hub-repo-id", type=str, required=True,
                        help="Hugging Face Hub repository ID (e.g., username/model-name)")
    
    # Optional metadata arguments
    parser.add_argument("--repo-id", type=str,
                        help="Original dataset repo ID used for training (will be added to config if missing)")
    parser.add_argument("--layer-name", type=str,
                        help="Layer name that was targeted during training (will be added to config if missing)")
    
    # Hub configuration arguments
    parser.add_argument("--hub-private", action="store_true", 
                        help="Make Hub repository private")
    parser.add_argument("--hub-license", type=str, default="mit", 
                        help="License for Hub repository (default: mit)")
    parser.add_argument("--hub-tags", type=str, nargs="*", 
                        default=["sae", "sparse-autoencoder", "robotics", "physical-ai-interpretability"],
                        help="Tags for Hub repository")
    parser.add_argument("--commit-message", type=str,
                        default="Upload SAE model weights, config, and training state",
                        help="Commit message for the upload")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./hub_upload_temp",
                        help="Temporary directory for preparing upload (default: ./hub_upload_temp)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files after upload")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log configuration
    logging.info("Starting SAE model upload to Hub with configuration:")
    logging.info(f"  Checkpoint: {args.checkpoint}")
    logging.info(f"  Hub repo: {args.hub_repo_id}")
    logging.info(f"  Private repo: {args.hub_private}")
    logging.info(f"  License: {args.hub_license}")
    logging.info(f"  Tags: {args.hub_tags}")
    
    try:
        # Find model files
        checkpoint_path = Path(args.checkpoint)
        logging.info(f"Scanning checkpoint path: {checkpoint_path}")
        
        model_files = find_model_files(checkpoint_path)
        logging.info(f"Found files:")
        for file_type, file_path in model_files.items():
            if file_path:
                logging.info(f"  {file_type}: {file_path}")
            else:
                logging.info(f"  {file_type}: Not found")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create complete model directory
        logging.info("Creating complete model directory...")
        complete_dir = create_complete_model_directory(
            model_files=model_files,
            output_dir=output_dir,
            repo_id=args.repo_id,
            layer_name=args.layer_name
        )
        
        # Upload to Hub
        logging.info("Uploading to Hugging Face Hub...")
        commit_info = push_model_to_hub(
            complete_model_dir=complete_dir,
            hub_repo_id=args.hub_repo_id,
            hub_private=args.hub_private,
            hub_license=args.hub_license,
            hub_tags=args.hub_tags,
            commit_message=args.commit_message
        )
        
        logging.info("✅ Upload completed successfully!")
        logging.info(f"🔗 Model URL: {commit_info.repo_url.url}")
        logging.info(f"📝 You can now load this model using:")
        logging.info(f"     from physical_ai_interpretability.sae import load_sae_from_hub")
        logging.info(f"     model = load_sae_from_hub('{args.hub_repo_id}')")
        
        # Clean up temporary files unless requested to keep them
        if not args.keep_temp:
            shutil.rmtree(output_dir)
            logging.info(f"🧹 Cleaned up temporary directory: {output_dir}")
        else:
            logging.info(f"📁 Temporary files kept at: {output_dir}")
            
    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()