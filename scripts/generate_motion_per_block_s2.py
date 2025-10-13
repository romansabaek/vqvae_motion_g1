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

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter


class MotionBlockGenerator:
    """Generate motion for each individual motion block."""
    
    def __init__(self, config_path: str, checkpoint_path: str, input_pkl_file: str):
        """Initialize with config, checkpoint, and input PKL data."""
        # Load config
        config_loader = ConfigLoader()
        self.config = config_loader.load_config(config_path)
        
        # Store input file path
        self.input_pkl_file = input_pkl_file
        
        # Load original AMASS data to get the exact format
        print(f"Loading original AMASS data from: {input_pkl_file}")
        self.original_motions = joblib.load(input_pkl_file)
        self.original_keys = list(self.original_motions.keys())
        print(f"Loaded {len(self.original_keys)} original motions")
        
        # Initialize agent and motion adapter
        self.agent = MVQVAEAgent(config=self.config)
        self.checkpoint_path = checkpoint_path
        self.motion_adapter = MotionDataAdapter(self.config)
        
        # Load motion data in MVQ format (50 dimensions for G1)
        print(f"Loading motion data in MVQ format...")
        self.mocap_data, self.end_indices, self.frame_size = self.motion_adapter.load_motion_data(input_pkl_file, [0])
        print(f"Loaded motion data: {self.mocap_data.shape}, frame_size: {self.frame_size}")
        
        # Setup agent with motion data and normalization statistics
        self.agent.mocap_data = self.mocap_data
        self.agent.end_indices = self.end_indices
        self.agent.frame_size = self.frame_size
        
        # Calculate normalization statistics (dataset-wide)
        mean = self.mocap_data.mean(dim=0)
        std = self.mocap_data.std(dim=0)
        std[std == 0] = 1.0
        
        self.agent.mean = mean
        self.agent.std = std
        
        # Initialize model
        self._initialize_model()
        
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
    
    def generate_motion_per_block(self, block_ids: List[int] = None, output_dir: str = "./outputs/motion_blocks"):
        """Generate motion for each individual motion block - separate file per block ID."""
        print(f"\n=== Generating Motion Per Motion Block ===")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # If no specific block IDs provided, generate for all blocks
        if block_ids is None:
            block_ids = list(range(self.config['nb_code']))
        
        print(f"Generating motion for {len(block_ids)} motion blocks...")
        print(f"Output directory: {output_path}")
        
        generated_blocks = []
        
        for block_id in block_ids:
            print(f"\nProcessing motion block {block_id}...")
            
            try:
                # Generate motion using simplified VQVAE approach
                motion_data = self._generate_single_block_motion(block_id)
                
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
                        'duration': motion_data['dof'].shape[0] / 30.0  # Assuming 30 FPS
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
    
    def _generate_single_block_motion(self, block_id: int):
        """Generate motion for a single block - NO REPETITION, just one instance of the block."""
        try:
            # Ensure agent has proper normalization statistics (dataset-wide)
            print(f"  Block {block_id}: Using dataset-wide normalization statistics")
            
            # SINGLE BLOCK: Create sequence with just ONE instance of the target block
            with torch.no_grad():
                # Create sequence with only ONE block (no repetition)
                single_block_sequence = torch.tensor([block_id], dtype=torch.long, device=self.agent.device)
                print(f"  Block {block_id}: Single block sequence: [{block_id}]")
                
                # Generate motion directly from single codebook sequence
                reconstructed_motion = self.agent.evalulate_from_codebook_seq(single_block_sequence)
                print(f"  Block {block_id}: Generated motion shape: {reconstructed_motion.shape}")
                
                # Convert directly to AMASS format with minimal processing
                # Use motion 0 as reference for AMASS format structure
                amass_motion = self._convert_to_amass_format(reconstructed_motion, block_id, template_motion_id=0)
                
                return amass_motion
                
        except Exception as e:
            print(f"Error generating motion for block {block_id}: {e}")
            return None
    
    
    def _convert_to_amass_format(self, reconstructed_motion: torch.Tensor, block_id: int, template_motion_id: int):
        """
        UPDATED: Convert VQVAE output to AMASS format for G1 humanoid with motion_lib-style features.
        The VQVAE decoder outputs G1 format: [root_deltas(4), dof_positions(23), dof_velocities(23)] = 50 dimensions
        Root deltas are now local frame velocities (dx, dy, dz, dyaw) that need to be integrated.
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
        
        # Initialize root rotation first
        root_rot = np.zeros((seq_len, 4))
        root_rot[0] = reference_motion['root_rot'][0]
        
        for i in range(1, seq_len):
            # Root deltas are already local frame per-frame deltas
            local_delta = root_deltas[i, :3]  # dx, dy, dz in local frame
            yaw_delta = root_deltas[i, 3]  # dyaw
            
            # Get previous frame's yaw
            prev_yaw = np.arctan2(root_rot[i-1, 3], root_rot[i-1, 0]) * 2
            
            # Rotate local delta to global frame
            cos_yaw = np.cos(prev_yaw)
            sin_yaw = np.sin(prev_yaw)
            global_delta = np.array([
                cos_yaw * local_delta[0] - sin_yaw * local_delta[1],
                sin_yaw * local_delta[0] + cos_yaw * local_delta[1],
                local_delta[2]
            ])
            
            root_trans_offset[i] = root_trans_offset[i-1] + global_delta
            
            # Reconstruct root rotation from yaw deltas
            prev_yaw = np.arctan2(root_rot[i-1, 3], root_rot[i-1, 0]) * 2
            new_yaw = prev_yaw + yaw_delta
            
            # Create new quaternion with stabilized roll/pitch
            root_rot[i] = np.array([np.cos(new_yaw/2), 0, 0, np.sin(new_yaw/2)])
        
        # Use DOF positions directly (G1 format)
        dof = dof_positions.astype(np.float32)
        
        # Create simple contact mask based on root height
        contact_mask = np.zeros((seq_len, 2))
        # For G1, we'll use a simple heuristic based on root height
        root_height = root_trans_offset[:, 2]
        contact_threshold = np.percentile(root_height, 10)  # Bottom 10% as contact
        contact_mask[:, 0] = root_height < contact_threshold  # Left foot
        contact_mask[:, 1] = root_height < contact_threshold  # Right foot
        
        # Create AMASS format structure for G1
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
        generated_blocks = generator.generate_motion_per_block(all_block_ids, args.output_dir)
    else:
        generated_blocks = generator.generate_motion_per_block(block_ids, args.output_dir)
    
    print(f"\n🎉 Motion block generation complete!")
    print(f"Generated {len(generated_blocks)} motion blocks")
    print(f"Check the results in: {args.output_dir}")


if __name__ == "__main__":
    main()


'''
motion id 1:
328,117,448,279,334,299,327,280,117,378,395,164,184,151,417,21,83,15,117,206,380,380,380,222

# Generate motion for specific blocks (separate file per block ID)
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --block_ids "0-10" \
    --output_dir ./outputs/motion_blocks

# Generate motion for all blocks
python scripts/generate_motion_per_block_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --generate_all \
    --output_dir ./outputs/motion_blocks
'''