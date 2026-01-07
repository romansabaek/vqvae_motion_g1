#!/usr/bin/env python3
"""
Generate raw motion representation for each individual motion block (codebook entry).
Outputs denormalized local features directly from the decoder - no AMASS conversion.

Creates separate NPY files for each motion block containing the raw motion features:
- Shape: [T, frame_size] where frame_size is typically 50
- Format: [root_deltas_local(4), dof_positions(23), dof_velocities(23)]
- These are denormalized local features ready for use
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import sys

# Add motion_vqvae to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.vqvae_gen_init import (
    load_config_and_agent,
    ensure_stats,
    initialize_model,
)


class MotionBlockGenerator:
    """Generate raw motion representation for each individual motion block."""
    
    def __init__(self, config_path: str, checkpoint_path: str, input_npy_file: Optional[str] = None, input_dir: Optional[str] = None):
        """
        Initialize with config and checkpoint.
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
            input_npy_file: Optional single NPY file for computing normalization stats.
                          Only needed if mean/std are not saved in checkpoint.
            input_dir: Optional directory containing NPY files for computing normalization stats.
                      If provided, all files in the directory will be used for statistics.
                      Takes precedence over input_npy_file if both are provided.
        """
        # Load config and agent
        self.config, self.agent, self.motion_adapter = load_config_and_agent(config_path, checkpoint_path)
        
        # Get frame_size: try checkpoint mean/std, then config, then adapter constant
        frame_size = None
        ckpt = torch.load(checkpoint_path, map_location=self.agent.device)
        if "mean" in ckpt and ckpt["mean"] is not None:
            frame_size = int(torch.as_tensor(ckpt["mean"]).numel())
        frame_size = frame_size or self.config.get('frame_size') or self.motion_adapter.TOTAL_FRAME_SIZE
        
        self.agent.frame_size = self.frame_size = int(frame_size)
        
        # Initialize model (loads mean/std from checkpoint if available)
        initialize_model(self.agent, self.config, self.frame_size, checkpoint_path)
        
        # Ensure stats: compute from input file(s) if not in checkpoint
        if getattr(self.agent, "mean", None) is None or getattr(self.agent, "std", None) is None:
            if input_dir is None and input_npy_file is None:
                raise ValueError("Normalization stats not in checkpoint. Please provide --input_dir or --input_npy to compute stats.")
            
            # Prefer input_dir over input_npy_file (loads all files for better statistics)
            if input_dir is not None:
                # Load all files from directory for statistics
                # The adapter's load_motion_data automatically handles directories
                # and loads all NPY files matching saved_desired_states_*.npy pattern
                print(f"Loading all NPY files from directory for statistics: {input_dir}")
                mocap_data, end_indices, _ = self.motion_adapter.load_motion_data(input_dir, None)  # None = load all motions
                ensure_stats(self.agent, mocap_data)
                num_files = len(end_indices) if end_indices is not None else 1
                print(f"Computed stats from {num_files} motion file(s) in directory: {input_dir}")
            else:
                # Use single file (backward compatibility)
                # Use adapter's load_motion_data which automatically uses _load_npy_data for .npy files
                # This properly handles NPY format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id]
                print(f"Loading single NPY file for statistics: {input_npy_file}")
                mocap_data, _, _ = self.motion_adapter.load_motion_data(input_npy_file, [0])
                ensure_stats(self.agent, mocap_data)
                print(f"Computed stats from: {input_npy_file}")
        
        print(f"Frame size: {self.frame_size}, Codebook: {self.config['nb_code']}, Window: {self.config['window_size']}")
        
        # Verify codebook is properly initialized
        if hasattr(self.agent.model, 'vqvae') and hasattr(self.agent.model.vqvae, 'quantizer'):
            quantizer = self.agent.model.vqvae.quantizer
            if hasattr(quantizer, 'codebook'):
                codebook = quantizer.codebook
                print(f"Codebook shape: {codebook.shape if isinstance(codebook, torch.Tensor) else 'N/A'}")
                # Check that different codebook entries are different
                if isinstance(codebook, torch.Tensor) and codebook.shape[0] > 1:
                    sample_entries = codebook[:min(3, codebook.shape[0])]
                    diffs = torch.abs(sample_entries[1:] - sample_entries[0:1])
                    max_diff = torch.max(diffs).item()
                    print(f"Codebook entries differ: max difference between first 3 entries = {max_diff:.6f}")
                    if max_diff < 1e-6:
                        print("⚠️  WARNING: Codebook entries appear to be identical! This may indicate a model loading issue.")
    
    
    def generate_motion_per_block(self, block_ids: List[int] = None, output_dir: str = "./outputs/motion_blocks", repeat_blocks: int = 1):
        """
        Generate raw motion representation for each individual motion block.
        
        Outputs denormalized local features: [T, frame_size] where frame_size is typically 50.
        Format: [root_deltas_local(4), dof_positions(23), dof_velocities(23)]
        """
        print(f"\n=== Generating Raw Motion Per Motion Block ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # If no specific block IDs provided, generate for all blocks
        if block_ids is None:
            block_ids = list(range(self.config['nb_code']))
        
        print(f"Generating motion for {len(block_ids)} motion blocks...")
        print(f"Output directory: {output_path}")
        print(f"Output format: Raw denormalized local features [T, {self.frame_size}]")
        
        generated_count = 0
        
        for block_id in block_ids:
            print(f"\nProcessing motion block {block_id}...")
            
            try:
                # Generate raw motion features for this block
                raw_motion = self._generate_single_block_motion(block_id, repeat_blocks)
                
                if raw_motion is not None:
                    # Save raw motion features directly (no conversion)
                    block_file = output_path / f"motion_block_{block_id:03d}.npy"
                    np.save(block_file, raw_motion)
                    generated_count += 1
                    
                    # Compute some statistics to verify uniqueness
                    motion_mean = np.mean(raw_motion, axis=0)
                    motion_std = np.std(raw_motion, axis=0)
                    motion_sum = np.sum(raw_motion)
                    
                    print(f"✅ Saved motion block {block_id} to: {block_file}")
                    print(f"   Shape: {raw_motion.shape}, dtype: {raw_motion.dtype}")
                    print(f"   Format: [frames, features] = [{raw_motion.shape[0]}, {raw_motion.shape[1]}]")
                    print(f"   Stats: sum={motion_sum:.3f}, mean(0)={motion_mean[0]:.3f}, std(0)={motion_std[0]:.3f}")
                else:
                    print(f"❌ Failed to generate motion for block {block_id}")
                    
            except Exception as e:
                print(f"❌ Error generating motion for block {block_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n=== Generation Complete ===")
        print(f"Generated {generated_count}/{len(block_ids)} motion blocks")
        print(f"Saved to: {output_path}")
        print("Format: Raw denormalized local features (no AMASS conversion)")
        
        # Verify that different blocks produce different outputs
        if generated_count >= 2 and len(block_ids) >= 2:
            try:
                # Load first two generated blocks and compare
                block_0_path = output_path / f"motion_block_{block_ids[0]:03d}.npy"
                block_1_path = output_path / f"motion_block_{block_ids[1]:03d}.npy"
                if block_0_path.exists() and block_1_path.exists():
                    block_0 = np.load(block_0_path)
                    block_1 = np.load(block_1_path)
                    # Compare shapes and values
                    if block_0.shape == block_1.shape:
                        diff = np.abs(block_0 - block_1)
                        max_diff = np.max(diff)
                        mean_diff = np.mean(diff)
                        print(f"\n🔍 Validation: Comparing block {block_ids[0]} vs {block_ids[1]}")
                        print(f"   Max difference: {max_diff:.6f}")
                        print(f"   Mean difference: {mean_diff:.6f}")
                        if max_diff < 1e-6:
                            print(f"   ⚠️  WARNING: Blocks {block_ids[0]} and {block_ids[1]} produce identical outputs!")
                            print(f"   This suggests the codebook may not be properly initialized or used.")
                        else:
                            print(f"   ✓ Blocks produce different outputs (as expected)")
            except Exception as e:
                print(f"   Could not validate block differences: {e}")
        
        return generated_count
    
    def _generate_single_block_motion(self, block_id: int, repeat_blocks: int = 1):
        """
        Generate raw motion representation for a single block.
        
        Returns:
            numpy array of shape [T, frame_size] with denormalized local features
        """
        try:
            # Ensure model is in eval mode
            self.agent.model.eval()
            
            with torch.no_grad():
                # Create sequence with repeated block
                window_size = self.config['window_size']
                upsampling_factor = 2 ** self.config['down_t']  # Typically 4x upsampling
                
                # Validate block_id is within codebook range
                nb_code = self.config['nb_code']
                if block_id < 0 or block_id >= nb_code:
                    raise ValueError(f"Block ID {block_id} is out of range [0, {nb_code-1}]")
                
                # Create repeated block sequence (1D tensor of codebook indices)
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
                
                # Verify the sequence contains the expected block_id
                unique_ids = torch.unique(padded_sequence).cpu().numpy()
                if len(unique_ids) != 1 or unique_ids[0] != block_id:
                    raise ValueError(f"Sequence validation failed: expected all {block_id}, got {unique_ids}")
                
                # Generate motion using VQVAE decoder (outputs normalized features)
                # forward_decoder expects 1D tensor of codebook indices [N]
                motion_output = self.agent.model.forward_decoder(padded_sequence)
                
                # Remove batch dimension and truncate to expected length
                if motion_output.dim() == 3:
                    motion_output = motion_output.squeeze(0)
                
                expected_frames = repeat_blocks * upsampling_factor
                full_motion = motion_output[:expected_frames]
                
                # Denormalize to get raw motion features
                full_motion = full_motion * self.agent.std + self.agent.mean
                
                # Convert to numpy and return raw features (no AMASS conversion)
                return full_motion.cpu().numpy().astype(np.float32)
                
        except Exception as e:
            print(f"Error generating motion for block {block_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    


def main():
    parser = argparse.ArgumentParser(
        description='Generate raw motion representation for each motion block (codebook entry). '
                    'Outputs denormalized local features directly from decoder - no AMASS conversion.'
    )
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Optional: Directory containing NPY files (saved_desired_states_*.npy) for computing normalization stats. '
                            'All files in the directory will be used for better statistics. '
                            'Only needed if mean/std are not saved in checkpoint.')
    parser.add_argument('--input_npy', type=str, default=None, 
                       help='Optional: Path to single input motion data file (NPY) for computing normalization stats. '
                            'Only needed if mean/std are not saved in checkpoint and --input_dir is not provided. '
                            '--input_dir takes precedence if both are provided.')
    parser.add_argument('--block_ids', type=str, default='0-10', help='Block IDs to generate (e.g., "0,1,2" or "0-10")')
    parser.add_argument('--output_dir', type=str, default='./outputs/motion_blocks', help='Output directory')
    parser.add_argument('--repeat_blocks', type=int, default=1, 
                       help='Number of times to repeat each block during generation')
    
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
        args.input_npy,
        args.input_dir,
    )
    generated_count = generator.generate_motion_per_block(block_ids, args.output_dir, args.repeat_blocks)
    
    print(f"\n🎉 Generated {generated_count} motion blocks in: {args.output_dir}")


if __name__ == "__main__":
    main()


'''
# Example usage - input_dir or input_npy is optional if stats are in checkpoint:

# If stats are not in checkpoint, use input_dir to load all files for better statistics (recommended):
python scripts/generate_motion_per_block_s2_npy.py \
    --config configs/agent_codebook_1s_npy.yaml \
    --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/best_model.ckpt \
    --input_dir /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy \
    --block_ids "0-128" \
    --output_dir ./outputs/motion_blocks_npy

# Alternative: use single input_npy file (backward compatibility):
python scripts/generate_motion_per_block_s2_npy.py \
    --config configs/agent_codebook_1s_npy.yaml \
    --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/best_model.ckpt \
    --input_npy /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy/saved_desired_states_1.npy \
    --block_ids "0-128" \
    --output_dir ./outputs/motion_blocks_npy


'''