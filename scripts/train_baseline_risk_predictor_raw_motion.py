#!/usr/bin/env python3
"""
Training script for LSTM-based policy ID predictor using motion reference history.
Uses the same motion data format and windowing as VQVAE training.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import argparse
import json
from pathlib import Path
from itertools import cycle
from typing import Optional

# Add the parent directory to the path so we can import motion_vqvae
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from motion_vqvae.data.vq_dataset import MotionVQDataset
from motion_vqvae.models.policy_lstm import PolicyLSTM

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def setup_device(device_str: str) -> torch.device:
    """Setup device based on string input."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def parse_motion_ids(motion_ids_str: str, motion_id: int, motion_ids_file: str = None) -> list:
    """
    Parse motion IDs from various formats:
    - JSON list: "[0,1,2,5,10]"
    - Comma-separated: "0,1,2,5,10"
    - Range: "0-10" or "0-10,20,30-35"
    - File path: path to file containing IDs (one per line or JSON list)
    - Single ID: via motion_id parameter (backward compatibility)
    
    Examples:
        --motion_ids "[0,1,2,5,10]"           # JSON list
        --motion_ids "0,1,2,5,10"              # Comma-separated
        --motion_ids "0-10"                    # Range
        --motion_ids "0-10,20,30-35"           # Mixed
        --motion_ids_file motion_ids.txt       # From file
    """
    # First check if a file is provided
    if motion_ids_file is not None:
        file_path = Path(motion_ids_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Motion IDs file not found: {motion_ids_file}")
        
        try:
            # Try to parse as JSON first
            with open(file_path, 'r') as f:
                content = f.read().strip()
                try:
                    motion_ids = json.loads(content)
                    if isinstance(motion_ids, list):
                        return sorted(list(set(int(id) for id in motion_ids)))
                except json.JSONDecodeError:
                    # If not JSON, read as one ID per line
                    motion_ids = []
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip empty lines and comments
                            motion_ids.append(int(line))
                    return sorted(list(set(motion_ids)))
        except Exception as e:
            raise ValueError(f"Failed to parse motion IDs file {motion_ids_file}: {e}")
    
    # Parse from string
    if motion_ids_str is not None:
        motion_ids_str = motion_ids_str.strip()
        
        # Try to parse as JSON list first
        if motion_ids_str.startswith('[') and motion_ids_str.endswith(']'):
            try:
                motion_ids = json.loads(motion_ids_str)
                if isinstance(motion_ids, list):
                    return sorted(list(set(int(id) for id in motion_ids)))
            except json.JSONDecodeError:
                pass  # Fall through to comma-separated parsing
        
        # Parse comma-separated IDs or ranges
        motion_ids = []
        for part in motion_ids_str.split(','):
            part = part.strip()
            if '-' in part:
                # Handle range (e.g., "0-10")
                start, end = map(int, part.split('-'))
                motion_ids.extend(range(start, end + 1))
            else:
                # Handle single ID
                motion_ids.append(int(part))
        return sorted(list(set(motion_ids)))  # Remove duplicates and sort
    
    elif motion_id is not None:
        # Backward compatibility with single motion_id
        return [motion_id]
    else:
        # No motion IDs specified, use all
        return None


def train_lstm_policy_predictor(
    config: dict,
    motion_file: str,
    motion_ids: Optional[list],
    device: torch.device,
    output_dir: Path,
    checkpoint_dir: Path,
    checkpoint_path: Optional[Path] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
):
    """Train LSTM model for policy ID prediction."""
    
    # Create motion data adapter
    motion_adapter = MotionDataAdapter(config)
    
    # Load motion data
    log.info(f"Loading motion data from: {motion_file}")
    mocap_data, end_indices, frame_size = motion_adapter.load_motion_data(motion_file, motion_ids)
    policy_ids = motion_adapter.get_policy_ids()
    
    log.info(f"Loaded {mocap_data.shape[0]} frames, frame_size={frame_size}, {len(end_indices)} motions")
    
    # Check if policy IDs are available
    if policy_ids is None:
        raise ValueError("Policy IDs are required for training. Make sure the motion file contains policy_id data.")
    
    # Count unique policy IDs
    unique_policy_ids = np.unique(policy_ids)
    num_policies = len(unique_policy_ids)
    log.info(f"Found {num_policies} unique policy IDs: {unique_policy_ids.tolist()}")
    
    # Compute normalization stats
    mean = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)
    std[std == 0] = 1.0
    
    # Create dataset
    window_size = config.get('window_size', 32)
    dataset = MotionVQDataset(
        mocap_data=mocap_data,
        end_indices=end_indices,
        window_size=window_size,
        mean=mean,
        std=std,
        policy_ids=policy_ids
    )
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    log.info(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
    
    # Create model
    model = PolicyLSTM(
        input_dim=frame_size,
        hidden_dim=config.get('lstm_hidden_dim', 256),
        num_layers=config.get('lstm_num_layers', 2),
        num_policies=num_policies,
        dropout=config.get('lstm_dropout', 0.1),
        bidirectional=config.get('lstm_bidirectional', False),
    ).to(device)
    
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    if checkpoint_path and checkpoint_path.exists():
        log.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        log.info(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.5f}")
    
    # Training loop
    model.train()
    train_loader_iter = cycle(train_loader)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Train for one epoch (iterate through dataset once)
        for batch_idx in range(len(train_loader)):
            motion_batch, policy_batch = next(train_loader_iter)
            motion_batch = motion_batch.to(device)  # (batch_size, window_size, feature_dim)
            policy_batch = policy_batch.to(device)  # (batch_size, window_size)
            
            # Use the last frame's policy ID as target (or majority vote)
            # For simplicity, use the last frame's policy ID
            target_policy = policy_batch[:, -1]  # (batch_size,)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(motion_batch)  # (batch_size, num_policies)
            loss = criterion(logits, target_policy)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            pred_policy = logits.argmax(dim=1)
            epoch_correct += (pred_policy == target_policy).sum().item()
            epoch_total += target_policy.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        
        log.info(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.5f}, Accuracy: {accuracy:.2%}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': min(best_loss, avg_loss),
            'config': config,
            'num_policies': num_policies,
            'frame_size': frame_size,
            'window_size': window_size,
        }
        
        # Save regular checkpoint
        checkpoint_path_epoch = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.ckpt"
        torch.save(checkpoint, checkpoint_path_epoch)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = checkpoint_dir / "best_model.ckpt"
            torch.save(checkpoint, best_checkpoint)
            log.info(f"Saved best model (loss: {best_loss:.5f})")
    
    log.info(f"Training completed! Best loss: {best_loss:.5f}")
    return model, best_loss


def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LSTM policy ID predictor with config file')
    parser.add_argument('--config', type=str, default='configs/agent_codebook_64.yaml', help='Path to YAML config file')
    parser.add_argument('--motion_file', type=str, required=True, help='Path to motion file (PKL or NPY)')
    parser.add_argument('--motion_ids', type=str, default=None, 
                       help='Motion IDs to load. Supports multiple formats:\n'
                            '  - JSON list: "[0,1,2,5,10]"\n'
                            '  - Comma-separated: "0,1,2,5,10"\n'
                            '  - Range: "0-10" or mixed "0-10,20,30-35"')
    parser.add_argument('--motion_ids_file', type=str, default=None,
                       help='Path to file containing motion IDs (one per line or JSON list)')
    parser.add_argument('--motion_id', type=int, default=None, help='Single motion ID to load (for backward compatibility)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint file to load')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for final models')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save checkpoints during training (defaults to output_dir if not specified)')
    parser.add_argument('--resume', action='store_true', help='Automatically resume from most recent checkpoint in output_dir or ./checkpoints')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Parse motion IDs
    motion_ids = parse_motion_ids(args.motion_ids, args.motion_id, args.motion_ids_file)
    
    log.info("Starting LSTM Policy ID Predictor training")
    log.info(f"Config file: {args.config}")
    log.info(f"Motion file: {args.motion_file}")
    if motion_ids is not None:
        if args.motion_ids_file:
            log.info(f"Motion IDs loaded from file '{args.motion_ids_file}': {motion_ids} (total: {len(motion_ids)})")
        else:
            log.info(f"Motion IDs: {motion_ids} (total: {len(motion_ids)})")
    else:
        log.info("Motion IDs: All motions")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log.info(f"Set random seed to: {args.seed}")
    
    # Setup device
    device = setup_device(args.device)
    log.info(f"Using device: {device}")
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.to_dict()
    
    # Override seed in config if provided
    config['seed'] = args.seed
    config['device'] = device
    config['motion_file'] = args.motion_file
    config['motion_ids'] = motion_ids
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set checkpoint directory (defaults to output_dir if not specified)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    log.info(f"Final models will be saved to: {output_dir}")
    
    # Load checkpoint if provided or resume from latest
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            log.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        log.info(f"Loading checkpoint from: {checkpoint_path}")
    elif args.resume:
        # Find the most recent checkpoint
        checkpoint_dirs = []
        if args.checkpoint_dir:
            checkpoint_dirs.append(checkpoint_dir)
        checkpoint_dirs.append(output_dir)
        default_checkpoint_dir = Path("./checkpoints")
        if default_checkpoint_dir.exists():
            checkpoint_dirs.append(default_checkpoint_dir)
        
        checkpoint_files = []
        for ckpt_dir in checkpoint_dirs:
            checkpoint_files.extend(list(ckpt_dir.glob("*.ckpt")))
        
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            checkpoint_path = checkpoint_files[0]
            log.info(f"Resuming from most recent checkpoint: {checkpoint_path}")
        else:
            log.warning("No checkpoint files found. Starting training from scratch.")
            checkpoint_path = None
    
    # Start training
    try:
        log.info("Starting training...")
        model, best_loss = train_lstm_policy_predictor(
            config=config,
            motion_file=args.motion_file,
            motion_ids=motion_ids,
            device=device,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=checkpoint_path,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        
        # Save final model
        final_checkpoint = output_dir / "final_model.ckpt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'best_loss': best_loss,
        }, final_checkpoint)
        log.info(f"Final model saved to: {final_checkpoint}")
        
        # Copy best model to output_dir
        best_checkpoint = checkpoint_dir / "best_model.ckpt"
        if best_checkpoint.exists():
            import shutil
            output_best = output_dir / "best_model.ckpt"
            shutil.copy2(best_checkpoint, output_best)
            log.info(f"Best model copied to: {output_best} (Loss: {best_loss:.5f})")
        
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


'''

# Example usage for LSTM policy ID predictor:

python scripts/train_baseline_risk_predictor.py \
  --config configs/agent_codebook_switching.yaml \
  --motion_file /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --motion_ids "0-1000" \
  --device cuda:0 \
  --output_dir ./outputs/lstm_policy_predictor \
  --checkpoint_dir ./checkpoints/lstm_policy_predictor \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --batch_size 32

# Resume training from checkpoint:
python scripts/train_baseline_risk_predictor.py \
  --config configs/agent_codebook_switching.yaml \
  --motion_file /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --motion_ids "0-1000" \
  --device cuda:0 \
  --output_dir ./outputs/lstm_policy_predictor \
  --checkpoint_dir ./checkpoints/lstm_policy_predictor \
  --resume








'''