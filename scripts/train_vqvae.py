#!/usr/bin/env python3
"""
Training script for MotionVQVAE using YAML configuration and motion data files
"""

import sys
import os
import torch
import numpy as np
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import motion_vqvae
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def setup_device(device_str: str) -> torch.device:
    """Setup device based on string input."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def parse_motion_ids(motion_ids_str: str, motion_id: int) -> list:
    """Parse motion IDs from string or single ID."""
    if motion_ids_str is not None:
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


def main():

    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MotionVQVAE with config file')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to YAML config file')
    parser.add_argument('--motion_file', type=str, required=True, help='Path to motion file (PKL or NPY)')
    parser.add_argument('--motion_ids', type=str, default=None, help='Motion IDs to load (comma-separated, e.g., "0,1,2" or "0-10" for range)')
    parser.add_argument('--motion_id', type=int, default=None, help='Single motion ID to load (for backward compatibility)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation on existing model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Parse motion IDs
    motion_ids = parse_motion_ids(args.motion_ids, args.motion_id)
    
    log.info("Starting MotionVQVAE training with config file")
    log.info(f"Config file: {args.config}")
    log.info(f"Motion file: {args.motion_file}")
    if motion_ids is not None:
        log.info(f"Motion IDs: {motion_ids}")
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
    
    # Create agent
    agent = MVQVAEAgent(config=config, device=device)
    
    # Setup agent with motion file
    try:
        agent.setup_from_file(args.motion_file, motion_ids)
        log.info("Agent setup completed successfully!")
    except Exception as e:
        log.error(f"Failed to setup agent: {e}")
        raise
    
    # Load checkpoint if provided
    if args.checkpoint:
        log.info(f"Loading checkpoint from: {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Start training
    try:
        log.info("Starting training...")
        agent.fit()
        log.info("Training completed successfully!")
        
        # Save final model
        final_checkpoint = output_dir / "final_model.ckpt"
        agent.save(str(output_dir), "final_model.ckpt")
        log.info(f"Final model saved to: {final_checkpoint}")
        
        # Also save best model if available
        if agent.best_model_path:
            best_checkpoint = output_dir / "best_model.ckpt"
            import shutil
            shutil.copy2(agent.best_model_path, best_checkpoint)
            log.info(f"Best model copied to: {best_checkpoint} (Loss: {agent.best_loss:.5f})")
        
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
        # Save checkpoint before exiting
        interrupted_checkpoint = output_dir / "interrupted_model.ckpt"
        agent.save(str(output_dir), "interrupted_model.ckpt")
        log.info(f"Model saved to: {interrupted_checkpoint}")
    except Exception as e:
        log.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()


'''

python scripts/train_vqvae.py \
  --config configs/agent.yaml \
  --motion_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0-300" \
  --device auto \
  --output_dir ./outputs/run_0_300_v2

'''