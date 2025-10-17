#!/usr/bin/env python3
"""
Generate motion for each individual motion block (codebook entry).
Creates separate PKL files for each motion block to understand what each block represents.
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

from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from motion_vqvae.utils.init_utils import VQVAEInitHelper


class MotionBlockGenerator:
    """Generate motion for each individual motion block."""
    
    def __init__(self, config_path: str, checkpoint_path: str, input_pkl_file: str):
        """Initialize with config, checkpoint, and input PKL data."""
        # Load config via shared helper
        self.init_helper = VQVAEInitHelper(config_path, checkpoint_path)
        self.config = self.init_helper.config
        
        # Store input file path
        self.input_pkl_file = input_pkl_file
        
        # Load original AMASS data to get the exact format
        print(f"Loading original AMASS data from: {input_pkl_file}")
        self.original_motions = joblib.load(input_pkl_file)
        self.original_keys = list(self.original_motions.keys())
        print(f"Loaded {len(self.original_keys)} original motions")
        
        # Initialize agent and motion adapter
        self.agent = self.init_helper.agent
        self.checkpoint_path = checkpoint_path
        self.motion_adapter = self.init_helper.motion_adapter
        
        # Load motion data in MVQ format (50 dimensions for G1)
        print(f"Loading motion data in MVQ format...")
        self.mocap_data, self.end_indices, self.frame_size = self.motion_adapter.load_motion_data(input_pkl_file, [0])
        print(f"Loaded motion data: {self.mocap_data.shape}, frame_size: {self.frame_size}")
        
        # Setup agent with motion data and normalization statistics
        self.agent.mocap_data = self.mocap_data
        self.agent.end_indices = self.end_indices
        self.agent.frame_size = self.frame_size
        
        # Calculate normalization statistics (dataset-wide) if not in ckpt
        self.init_helper.ensure_stats(self.mocap_data)
        
        # Initialize model
        self.init_helper.initialize_model(self.frame_size)
        
        print(f"Window size: {self.config['window_size']}")
        print(f"Codebook size: {self.config['nb_code']}")
    
    def _initialize_model(self):
        """Initialize and load the trained VQVAE model - following the same pattern as generate_motion_from_vqvae_s2.py"""
        from motion_vqvae.models.models import MotionVQVAE
        
        # Ensure frame_size is in config
        self.agent.config['frame_size'] = self.frame_size
        
        # Initialize VQVAE model - same as reference code
        self.agent.model = MotionVQVAE(
            self.agent,  # Pass agent as args
            self.config['nb_code'],
            self.config['code_dim'],
            self.config['output_emb_width'],
            self.config['down_t'],
            self.config['stride_t'],
            self.config['width'],
            self.config['depth'],
            self.config['dilation_growth_rate'],
            self.config['vq_act'],
            self.config['vq_norm']
        ).to(self.agent.device)
        
        # Load trained model - same as reference code
        checkpoint = torch.load(self.checkpoint_path, map_location=self.agent.device)
        self.agent.model.load_state_dict(checkpoint['model'])
        self.agent.model.eval()
        
        print(f"Loaded trained model from: {self.checkpoint_path}")
        print(f"Model initialized with frame_size: {self.frame_size}")
    
    def generate_motion_per_block(self, block_ids: List[int] = None, output_dir: str = "./outputs/motion_blocks", repeat_blocks: int = 1):
        """Generate motion for each individual motion block - separate file per block ID.
        
        Args:
            block_ids: List of block IDs to generate
            output_dir: Output directory for PKL files
            repeat_blocks: Number of times to repeat each block in the sequence (default: 1)
        """
        print(f"\n=== Generating Motion Per Motion Block ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # If no specific block IDs provided, generate for all blocks
        if block_ids is None:
            block_ids = list(range(self.config['nb_code']))
        
        print(f"Generating motion for {len(block_ids)} motion blocks...")
        print(f"Output directory: {output_path}")
        print(f"Repeat blocks: {repeat_blocks} times")
        
        generated_blocks = []
        
        for block_id in block_ids:
            print(f"\nProcessing motion block {block_id}...")
            
            try:
                # Generate motion using simplified VQVAE approach with repetition
                motion_data = self._generate_single_block_motion(block_id, repeat_blocks)
                
                if motion_data is not None:
                    # Save separate PKL file for each block ID
                    block_file = output_path / f"motion_block_{block_id:03d}.pkl"
                    
                    # Create single-motion dictionary (same format as generate_motion_from_vqvae_s2.py)
                    motion_key = f"motion_block_{block_id:03d}"
                    single_motion_dict = {motion_key: motion_data}
                    
                    joblib.dump(single_motion_dict, block_file)
                    
                    # Store summary info
                    generated_blocks.append({
                        'block_id': block_id,
                        'motion_key': motion_key,
                        'file_path': str(block_file),
                        'motion_length': motion_data['dof'].shape[0],
                        'duration': motion_data['dof'].shape[0] / 30.0,  # Assuming 30 FPS
                        'repeat_blocks': repeat_blocks,
                        'code_indices_length': motion_data.get('code_indices_length', 0),
                        'decoder_upsampling_factor': motion_data.get('decoder_upsampling_factor', 1)
                    })
                    
                    print(f"✅ Saved motion block {block_id} to: {block_file}")
                    print(f"   Duration: {generated_blocks[-1]['duration']:.2f}s")
                else:
                    print(f"❌ Failed to generate motion for block {block_id}")
                    
            except Exception as e:
                print(f"❌ Error generating motion for block {block_id}: {e}")
                continue
        
        # Save summary CSV
        self._save_block_summary(generated_blocks, output_path)
        
        print(f"\n=== Generation Complete ===")
        print(f"Generated {len(generated_blocks)} motion blocks")
        print(f"Saved to: {output_path}")
        
        return generated_blocks
    
    def _generate_single_block_motion(self, block_id: int, repeat_blocks: int = 1):
        """Generate motion for a single block using proper VQVAE v2 approach.
        
        Args:
            block_id: The codebook block ID to generate
            repeat_blocks: Number of times to repeat the block in the sequence
        """
        try:
            # Ensure agent has proper normalization statistics (dataset-wide)
            print(f"  Block {block_id}: Using dataset-wide normalization statistics")
            
            # REPEATED BLOCK: Create sequence with the target block repeated multiple times
            with torch.no_grad():
                # Create sequence with the same block repeated multiple times
                # The decoder expects sequences of length window_size (32)
                window_size = self.config['window_size']
                
                # Calculate decoder upsampling factor from config
                # Decoder has down_t=2 upsampling layers with scale_factor=2 each
                upsampling_factor = 2 ** self.config['down_t']  # 2^2 = 4x upsampling
                
                # Create a sequence by repeating the same block multiple times
                # Each repetition adds one more instance of the block
                repeated_block_sequence = torch.full((repeat_blocks,), block_id, dtype=torch.long, device=self.agent.device)
                total_sequence_length = repeat_blocks
                
                print(f"  Block {block_id}: Repeated block sequence (length {total_sequence_length}): [{block_id}] * {total_sequence_length}")
                print(f"  Block {block_id}: Repeat factor: {repeat_blocks}x")
                print(f"  Block {block_id}: Decoder upsampling factor: {upsampling_factor}x")
                print(f"  Block {block_id}: Expected final motion length: {total_sequence_length * upsampling_factor} frames")
                
                # Process the sequence - we need to pad to window_size for decoder expectations
                if total_sequence_length < window_size:
                    # Pad the sequence to window_size by repeating the last element
                    padding_needed = window_size - total_sequence_length
                    padded_sequence = torch.cat([
                        repeated_block_sequence,
                        torch.full((padding_needed,), block_id, dtype=torch.long, device=self.agent.device)
                    ])
                else:
                    padded_sequence = repeated_block_sequence
                
                # Use proper VQVAE v2 forward_decoder method
                # This method expects code indices and handles dequantization internally
                motion_output = self.agent.model.forward_decoder(padded_sequence)
                
                # motion_output shape should be [1, seq_len, features] - squeeze batch dimension
                if motion_output.dim() == 3:
                    motion_output = motion_output.squeeze(0)  # Remove batch dimension
                
                # Take only the first total_sequence_length * upsampling_factor frames
                # (since we padded the input, we need to truncate the output)
                expected_frames = total_sequence_length * upsampling_factor
                full_motion = motion_output[:expected_frames]
                
                # Denormalize the motion using dataset-wide statistics
                full_motion = full_motion * self.agent.std + self.agent.mean
                
                print(f"  Block {block_id}: Generated motion shape: {full_motion.shape}")
                
                # Convert directly to AMASS format with minimal processing
                # Use motion 0 as reference for AMASS format structure
                amass_motion = self._convert_to_amass_format(full_motion, block_id, template_motion_id=0)
                
                # Add repeat information to the motion data
                amass_motion['repeat_blocks'] = repeat_blocks
                amass_motion['total_sequence_length'] = total_sequence_length
                amass_motion['decoder_upsampling_factor'] = upsampling_factor
                amass_motion['code_indices_length'] = total_sequence_length
                amass_motion['motion_frames_length'] = full_motion.shape[0]
                
                return amass_motion
                
        except Exception as e:
            print(f"Error generating motion for block {block_id}: {e}")
            return None
    
    
    def _convert_to_amass_format(self, reconstructed_motion: torch.Tensor, block_id: int, template_motion_id: int):
        """
        Convert VQVAE output to AMASS format for G1 humanoid with motion_lib-style features.
        - Decoder output: [root_deltas(4), dof_positions(23), dof_velocities(23)] = 50
        - Root deltas are local frame per-frame deltas (dx, dy, dz, dyaw)
        - Quaternion convention: XYZW
        """
        # Get reference motion for basic structure
        reference_motion = self.original_motions[self.original_keys[template_motion_id]]
        
        mvq_np = reconstructed_motion.cpu().numpy()
        seq_len = mvq_np.shape[0]
        fps = reference_motion.get('fps', 30)
        
        # Extract MVQ components for G1 format
        # Root deltas are now local frame velocities (per-frame deltas)
        root_deltas = mvq_np[:, 0:4]  # [T, 4] - dx, dy, dz, dyaw in local frame (per-frame deltas)
        dof_positions = mvq_np[:, 4:27]  # [T, 23] - DOF positions
        dof_velocities = mvq_np[:, 27:50]  # [T, 23] - DOF velocities
        
        # Convert local frame deltas to global positions
        root_trans_offset = np.zeros((seq_len, 3))
        root_trans_offset[0] = reference_motion['root_trans_offset'][0]
        
        # Initialize root rotation first (XYZW)
        root_rot = np.zeros((seq_len, 4))
        root_rot[0] = reference_motion['root_rot'][0]
        
        for i in range(1, seq_len):
            # Root deltas are already local frame per-frame deltas
            local_delta = root_deltas[i, :3]  # dx, dy, dz in local frame
            yaw_delta = root_deltas[i, 3]  # dyaw
            
            # Rotate local delta to global frame using quaternion (XYZW)
            q_prev = root_rot[i-1]
            global_delta = self._quat_rotate_xyzw(q_prev, local_delta)
            
            root_trans_offset[i] = root_trans_offset[i-1] + global_delta
            
            # Integrate yaw delta via quaternion multiply (XYZW)
            half = 0.5 * yaw_delta
            q_delta = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
            q_new = self._quat_mul_xyzw(q_prev, q_delta)
            norm = np.linalg.norm(q_new)
            root_rot[i] = q_new / (norm if norm > 0 else 1.0)
        
        # Use DOF positions directly (G1 format)
        dof = dof_positions.astype(np.float32)
        
        # Create simple contact mask based on root height
        contact_mask = np.zeros((seq_len, 2))
        # For G1, we'll use a simple heuristic based on root height
        root_height = root_trans_offset[:, 2]
        contact_threshold = np.percentile(root_height, 10)  # Bottom 10% as contact
        contact_mask[:, 0] = root_height < contact_threshold  # Left foot
        contact_mask[:, 1] = root_height < contact_threshold  # Right foot
        
        # Create AMASS format structure for G1 (XYZW quaternion)
        amass_format_motion = {
            'root_trans_offset': root_trans_offset.astype(np.float32),
            'root_rot': root_rot.astype(np.float32),
            'pose_aa': reference_motion['pose_aa'][:seq_len].copy().astype(np.float32),  # Keep original
            'smpl_joints': reference_motion['smpl_joints'][:seq_len].copy().astype(np.float32),  # Keep original for compatibility
            'contact_mask': contact_mask.astype(np.float32),
            'dof': dof,
            'fps': fps,
            'block_id': block_id,
            'sequence_length': seq_len,
            'generation_method': 'g1_vqvae_motionlib_smooth'  # Mark as G1 with motion_lib smoothing
        }
        
        return amass_format_motion

    @staticmethod
    def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiply q1*q2 in XYZW convention."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1*w2 - (x1*x2 + y1*y2 + z1*z2)
        x = w1*x2 + w2*x1 + (y1*z2 - z1*y2)
        y = w1*y2 + w2*y1 + (z1*x2 - x1*z2)
        z = w1*z2 + w2*z1 + (x1*y2 - y1*x2)
        return np.array([x, y, z, w])

    @staticmethod
    def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q (XYZW)."""
        x, y, z, w = q
        q_vec = np.array([x, y, z])
        
        a = v * (2.0 * w * w - 1.0)
        b = 2.0 * w * np.cross(q_vec, v)
        c = 2.0 * q_vec * np.dot(q_vec, v)
        return a + b + c
    
    def _save_block_summary(self, generated_blocks: List[dict], output_path: Path):
        """Save summary of generated motion blocks."""
        import pandas as pd
        
        if not generated_blocks:
            print("No blocks generated to save summary")
            return
        
        # Create summary DataFrame
        df = pd.DataFrame(generated_blocks)
        
        # Save summary CSV
        summary_file = output_path / "motion_blocks_summary.csv"
        df.to_csv(summary_file, index=False)
        
        print(f"📊 Block summary saved to: {summary_file}")
        print(f"📈 Summary:")
        print(f"  - Total blocks generated: {len(generated_blocks)}")
        print(f"  - Average duration: {df['duration'].mean():.2f}s")
        print(f"  - Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")
    


def main():
    parser = argparse.ArgumentParser(description='Generate motion for each motion block')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt', help='Path to model checkpoint')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to input PKL motion data file')
    parser.add_argument('--block_ids', type=str, default=None, help='Comma-separated block IDs to generate (e.g., "0,1,2" or "0-10")')
    parser.add_argument('--output_dir', type=str, default='./outputs/motion_blocks', help='Output directory for motion blocks')
    parser.add_argument('--generate_all', action='store_true', help='Generate motion for all blocks')
    parser.add_argument('--repeat_blocks', type=int, default=1, help='Number of times to repeat each block in the sequence (default: 1)')
    
    args = parser.parse_args()
    
    # Parse block IDs
    block_ids = None
    if args.block_ids:
        block_ids = []
        for x in args.block_ids.split(','):
            x = x.strip()
            if '-' in x:
                start, end = map(int, x.split('-'))
                block_ids.extend(range(start, end + 1))
            else:
                block_ids.append(int(x))
    
    # Initialize generator
    generator = MotionBlockGenerator(args.config, args.checkpoint, args.input_pkl)
    
    # Generate motions
    if args.generate_all:
        all_block_ids = list(range(generator.config['nb_code']))
        generated_blocks = generator.generate_motion_per_block(all_block_ids, args.output_dir, args.repeat_blocks)
    else:
        generated_blocks = generator.generate_motion_per_block(block_ids, args.output_dir, args.repeat_blocks)
    
    print(f"\n🎉 Motion block generation complete!")
    print(f"Generated {len(generated_blocks)} motion blocks")
    print(f"Check the results in: {args.output_dir}")


if __name__ == "__main__":
    main()


'''
# Generate motion for specific blocks (separate file per block ID)
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --block_ids "1-10" \
    --output_dir ./outputs/motion_blocks

# Generate motion for specific blocks with repetition (longer sequences)
# Note: repeat_blocks=4 repeats the same codebook block 4 times
# Each code gets upsampled by 4x, so 4 codes → 16 frames → 0.53 seconds at 30 FPS
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --block_ids "1-5" \
    --repeat_blocks 4 \
    --output_dir ./outputs/motion_blocks_long

# Generate motion for all blocks
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --generate_all \
    --output_dir ./outputs/motion_blocks

# Generate motion for all blocks with repetition (very long sequences)
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --generate_all \
    --repeat_blocks 8 \
    --output_dir ./outputs/motion_blocks_very_long
'''