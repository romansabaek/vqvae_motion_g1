#!/usr/bin/env python3
"""
Generate motion for each individual motion block (codebook entry).
Creates separate PKL files for each motion block to understand what each block represents.

Global/world export: decodes local features and converts to AMASS-like globals
by integrating root motion, same approach as in generate_motion_from_vqvae_s2.py.
"""

import numpy as np
import torch
import joblib
from pathlib import Path
import argparse
from typing import List
import sys

# Add motion_vqvae to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.vqvae_gen_init import (
    load_config_and_agent,
    load_original_pkl,
    ensure_stats,
    initialize_model,
    convert_to_amass_global_from_local,
)


class MotionBlockGenerator:
    """Generate motion for each individual motion block."""
    
    def __init__(self, config_path: str, checkpoint_path: str, input_pkl_file: str, base_motion_id: int = 0, heading_mode: str = "preserve"):
        """Initialize with config, checkpoint, and input PKL data."""
        self.input_pkl_file = input_pkl_file
        self.base_motion_id = int(base_motion_id)
        self.heading_mode = heading_mode
        
        # Shared init (same as generate_motion_from_vqvae_s2.py)
        self.config, self.agent, self.motion_adapter = load_config_and_agent(config_path, checkpoint_path)
        
        # Load original AMASS data
        print(f"Loading original AMASS data from: {input_pkl_file}")
        self.original_motions, self.original_keys = load_original_pkl(input_pkl_file)
        print(f"Loaded {len(self.original_keys)} original motions")
        
        # Load multiple motions for proper normalization statistics (same as eval_vqvae.py)
        max_motions_for_stats = min(300, len(self.original_keys))
        subset_motion_ids = list(range(max_motions_for_stats))
        print(f"Loading first {max_motions_for_stats} motions for normalization stats...")
        
        mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
            input_pkl_file, subset_motion_ids
        )
        
        print(f"Loaded data for stats: shape={mocap_data.shape}, frame_size={frame_size}")
        print(f"Using device: {self.agent.device}")
        
        # Calculate normalization stats and initialize model (same as eval_vqvae.py)
        self.agent.frame_size = int(frame_size)
        self.frame_size = int(frame_size)  # Store for later use
        ensure_stats(self.agent, mocap_data)
        initialize_model(self.agent, self.config, self.frame_size, checkpoint_path)
        
        print(f"Window size: {self.config['window_size']}")
        print(f"Codebook size: {self.config['nb_code']}")
        base_idx_info = self.base_motion_id if 0 <= self.base_motion_id < len(self.original_keys) else 0
        print(f"Base motion id for orientation context: {base_idx_info} ({self.original_keys[base_idx_info]})")
        print(f"Heading mode: {self.heading_mode} (preserve|zero)")
    
    
    def generate_motion_per_block(self, block_ids: List[int] = None, output_dir: str = "./outputs/motion_blocks", repeat_blocks: int = 1):
        """Generate motion for each individual motion block."""
        print(f"\n=== Generating Motion Per Motion Block (global) ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # If no specific block IDs provided, generate for all blocks
        if block_ids is None:
            block_ids = list(range(self.config['nb_code']))
        
        print(f"Generating motion for {len(block_ids)} motion blocks...")
        print(f"Output directory: {output_path}")
        
        generated_count = 0
        
        for block_id in block_ids:
            print(f"\nProcessing motion block {block_id}...")
            
            try:
                # Generate motion for this block
                motion_data = self._generate_single_block_motion(block_id, repeat_blocks)
                
                if motion_data is not None:
                    # Save PKL file for this block
                    block_file = output_path / f"motion_block_{block_id:03d}.pkl"
                    motion_key = f"motion_block_{block_id:03d}"
                    single_motion_dict = {motion_key: motion_data}
                    
                    joblib.dump(single_motion_dict, block_file)
                    generated_count += 1
                    
                    print(f"✅ Saved motion block {block_id} to: {block_file}")
                else:
                    print(f"❌ Failed to generate motion for block {block_id}")
                    
            except Exception as e:
                print(f"❌ Error generating motion for block {block_id}: {e}")
                continue
        
        print(f"\n=== Generation Complete ===")
        print(f"Generated {generated_count}/{len(block_ids)} motion blocks")
        print(f"Saved to: {output_path}")
        print("Space: global")
        
        return generated_count
    
    def _generate_single_block_motion(self, block_id: int, repeat_blocks: int = 1):
        """Generate motion for a single block."""
        try:
            with torch.no_grad():
                # Create sequence with repeated block
                window_size = self.config['window_size']
                upsampling_factor = 2 ** self.config['down_t']  # 4x upsampling
                
                # Create repeated block sequence
                repeated_sequence = torch.full((repeat_blocks,), block_id, dtype=torch.long, device=self.agent.device)
                
                # Pad to window_size if needed
                if repeat_blocks < window_size:
                    padding_needed = window_size - repeat_blocks
                    padded_sequence = torch.cat([
                        repeated_sequence,
                        torch.full((padding_needed,), block_id, dtype=torch.long, device=self.agent.device)
                    ])
                else:
                    padded_sequence = repeated_sequence
                
                # Generate motion using VQVAE decoder
                motion_output = self.agent.model.forward_decoder(padded_sequence)
                
                # Remove batch dimension and truncate to expected length
                if motion_output.dim() == 3:
                    motion_output = motion_output.squeeze(0)
                
                expected_frames = repeat_blocks * upsampling_factor
                full_motion = motion_output[:expected_frames]
                
                # Denormalize
                full_motion = full_motion * self.agent.std + self.agent.mean
                
                # Convert to AMASS format
                base_idx = self.base_motion_id if 0 <= self.base_motion_id < len(self.original_keys) else 0
                original_amass = self.original_motions[self.original_keys[base_idx]].copy()
                if self.heading_mode.lower() == "zero":
                    T_base = original_amass["root_rot"].shape[0]
                    neutral_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    original_amass["root_rot"] = np.tile(neutral_quat[None, :], (T_base, 1)).astype(np.float32)
                amass_motion = convert_to_amass_global_from_local(
                    local_features=full_motion.cpu().numpy().astype(np.float32),
                    original_motion=original_amass,
                    motion_id=block_id,
                )
                
                return amass_motion
                
        except Exception as e:
            print(f"Error generating motion for block {block_id}: {e}")
            return None
    


def main():
    parser = argparse.ArgumentParser(description='Generate motion for each motion block')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to input PKL motion data file')
    parser.add_argument('--block_ids', type=str, default='0-10', help='Block IDs to generate (e.g., "0,1,2" or "0-10")')
    parser.add_argument('--output_dir', type=str, default='./outputs/motion_blocks', help='Output directory')
    parser.add_argument('--repeat_blocks', type=int, default=1, help='Number of times to repeat each block during generation')
    parser.add_argument('--base_motion_id', type=int, default=0, help='Base motion id for initial orientation/pose context')
    parser.add_argument('--heading_mode', type=str, default='preserve', choices=['preserve', 'zero'], help='Heading handling for base orientation')
    
    args = parser.parse_args()
    
    # Parse block IDs
    block_ids = []
    for x in args.block_ids.split(','):
        x = x.strip()
        if '-' in x:
            start, end = map(int, x.split('-'))
            block_ids.extend(range(start, end + 1))
        else:
            block_ids.append(int(x))
    
    # Initialize generator and generate motions
    generator = MotionBlockGenerator(
        args.config,
        args.checkpoint,
        args.input_pkl,
        base_motion_id=args.base_motion_id,
        heading_mode=args.heading_mode,
    )
    generated_count = generator.generate_motion_per_block(block_ids, args.output_dir, args.repeat_blocks)
    
    print(f"\n🎉 Generated {generated_count} motion blocks in: {args.output_dir}")


if __name__ == "__main__":
    main()


'''
# Example usage:
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_v2/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --block_ids "1" \
    --output_dir ./outputs/motion_blocks
'''