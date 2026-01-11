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
import json
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


def main():

    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MotionVQVAE with config file')
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
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--validate_only', action='store_true', help='Only run validation on existing model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Parse motion IDs
    motion_ids = parse_motion_ids(args.motion_ids, args.motion_id, args.motion_ids_file)
    
    log.info("Starting MotionVQVAE training with config file")
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

# Examples of different motion ID formats:

# Range format (existing):
python scripts/train_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --motion_file /home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0-500" \
  --device auto \
  --output_dir ./outputs/run_0_500_switching




# NPY file format (supports both PKL and NPY):
python scripts/train_vqvae.py \
  --config configs/agent_codebook_1s_npy.yaml \
  --motion_file /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy \
  --motion_ids "0-500" \
  --device auto \
  --output_dir ./outputs/agent_codebook_1s_npy



'''