#!/usr/bin/env python3
"""
Generate PKL files in exact AMASS format from trained VQVAE model.
Creates reconstructed motion files that can be used directly in MuJoCo.
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


class AMASSFormatGenerator:
    """Generate PKL files in exact AMASS format from trained VQVAE model."""
    
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
        
        # Initialize model
        self._initialize_model()
        
        print(f"Window size: {self.config['window_size']}")
    
    def _initialize_model(self):
        """Initialize and load the trained VQVAE model."""
        from motion_vqvae.models.models import MotionVQVAE
        
        # Ensure frame_size is in config
        self.agent.config['frame_size'] = self.frame_size
        
        # Initialize VQVAE model
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
        
        # Load trained model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.agent.device)
        self.agent.model.load_state_dict(checkpoint['model'])
        self.agent.model.eval()
        
        print(f"Loaded trained model from: {self.checkpoint_path}")
        print(f"Model initialized with frame_size: {self.frame_size}")
    
    def generate_amass_format_pkl(self, motion_ids: List[int], output_dir: str = None, csv_file: str = None):
        """Generate separate PKL files in AMASS format for each motion ID."""
        print(f"\n=== Generating AMASS Format PKL Files ===")
        
        # Create output directory
        if output_dir is None:
            output_dir = "./outputs/vqvae_motions"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        codebook_data = []  # Store codebook sequences for CSV
        
        for motion_id in motion_ids:
            if motion_id >= len(self.original_keys):
                print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
                continue
                
            print(f"Processing motion ID {motion_id}...")
            
            # Generate reconstructed motion in AMASS format
            reconstructed_motion, codebook_sequence = self._generate_single_motion_amass_format(motion_id)
            
            # Display codebook sequence
            if codebook_sequence is not None:
                print(f"  Codebook Sequence Length: {len(codebook_sequence)} blocks")
                print(f"  Unique Blocks: {len(np.unique(codebook_sequence))}")
                print(f"  Codebook Sequence: {','.join(map(str, codebook_sequence))}")
                
                # Show in readable format
                chunk_size = 20
                print(f"  Codebook Sequence (readable):")
                for i in range(0, len(codebook_sequence), chunk_size):
                    chunk = codebook_sequence[i:i+chunk_size]
                    chunk_str = ' '.join(f"{block:3d}" for block in chunk)
                    print(f"    {i:3d}-{i+len(chunk)-1:3d}: {chunk_str}")
                print()
            
            if reconstructed_motion is not None:
                # Use original motion key for consistency
                original_key = self.original_keys[motion_id]
                
                # Create single-motion dictionary (same format as generate_motion_per_block.py)
                single_motion_dict = {original_key: reconstructed_motion}
                
                # Save individual PKL file for this motion
                motion_file = output_path / f"vqvae_motion_{motion_id:03d}.pkl"
                joblib.dump(single_motion_dict, motion_file)
                
                generated_files.append(str(motion_file))
                print(f"✅ Generated motion: {original_key} -> {motion_file}")
                
                # Store codebook sequence data
                codebook_data.append({
                    'motion_id': motion_id,
                    'motion_key': original_key,
                    'file_path': str(motion_file),
                    'sequence_length': reconstructed_motion['dof'].shape[0],
                    'codebook_sequence': codebook_sequence,
                    'unique_blocks': len(np.unique(codebook_sequence)),
                    'total_blocks': len(codebook_sequence)
                })
        
        print(f"\nSaved {len(generated_files)} individual AMASS format PKL files to: {output_path}")
        print(f"Generated {len(generated_files)} VQVAE motions in AMASS format")
        
        # Save codebook sequences to CSV
        if csv_file and codebook_data:
            self._save_codebook_csv(codebook_data, csv_file)
        
        return generated_files, output_dir
    
    def _generate_single_motion_amass_format(self, motion_id: int):
        """SIMPLIFIED: Generate reconstructed motion in AMASS format."""
        try:
            # Load motion data in MVQ format for this specific motion
            mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
                self.input_pkl_file, [motion_id]
            )
            
            # Setup agent with motion data
            self.agent.mocap_data = mocap_data
            self.agent.end_indices = end_indices
            self.agent.frame_size = frame_size
            
            # Calculate normalization statistics
            mean = mocap_data.mean(dim=0)
            std = mocap_data.std(dim=0)
            std[std == 0] = 1.0
            
            self.agent.mean = mean
            self.agent.std = std
            
            # Reconstruct motion using VQVAE
            with torch.no_grad():
                reconstructed_motion, original_seq, codebook_sequence = self.agent.evaluate_policy_rec(torch.tensor(0))
            
            # Convert to exact AMASS format
            amass_format_motion = self._convert_to_exact_amass_format(
                reconstructed_motion, original_seq, motion_id
            )
            
            # Convert codebook sequence to numpy
            codebook_sequence_np = codebook_sequence.cpu().numpy() if isinstance(codebook_sequence, torch.Tensor) else codebook_sequence
            
            return amass_format_motion, codebook_sequence_np
            
        except Exception as e:
            print(f"Error generating motion {motion_id}: {e}")
            return None, None
    
    def _convert_to_exact_amass_format(self, reconstructed_motion: torch.Tensor, original_motion: torch.Tensor, motion_id: int):
        """
        UPDATED: Convert VQVAE output to AMASS format for G1 humanoid with motion_lib-style features.
        The VQVAE decoder outputs G1 format: [root_deltas(4), dof_positions(23), dof_velocities(23)] = 50 dimensions
        Root deltas are now local frame velocities (dx, dy, dz, dyaw) that need to be integrated.
        """
        # Get original motion for reference format
        original_amass = self.original_motions[self.original_keys[motion_id]]
        
        mvq_np = reconstructed_motion.cpu().numpy()
        seq_len = mvq_np.shape[0]
        fps = original_amass.get('fps', 30)
        
        # Extract MVQ components for G1 format
        # Root deltas are now local frame velocities (per-frame deltas)
        root_deltas = mvq_np[:, 0:4]  # [T, 4] - dx, dy, dz, dyaw in local frame (per-frame deltas)
        dof_positions = mvq_np[:, 4:27]  # [T, 23] - DOF positions
        dof_velocities = mvq_np[:, 27:50]  # [T, 23] - DOF velocities
        
        # Convert local frame root deltas to global positions
        root_trans_offset = np.zeros((seq_len, 3))
        root_trans_offset[0] = original_amass['root_trans_offset'][0]
        
        # Initialize root rotation first
        root_rot = np.zeros((seq_len, 4))
        root_rot[0] = original_amass['root_rot'][0]
        
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
            'pose_aa': original_amass['pose_aa'].copy().astype(np.float32),  # Keep original
            'smpl_joints': original_amass['smpl_joints'].copy().astype(np.float32),  # Keep original for compatibility
            'contact_mask': contact_mask.astype(np.float32),
            'dof': dof,
            'fps': fps
        }
        
        return amass_format_motion
    
    def _save_codebook_csv(self, codebook_data: List[dict], csv_file: str):
        """Save codebook sequences to CSV file."""
        import pandas as pd
        
        # Create CSV path
        csv_path = Path(csv_file)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for CSV
        csv_rows = []
        for data in codebook_data:
            # Convert codebook sequence to string
            codebook_str = ','.join(map(str, data['codebook_sequence']))
            
            csv_rows.append({
                'motion_id': data['motion_id'],
                'motion_key': data['motion_key'],
                'file_path': data.get('file_path', ''),
                'sequence_length': data['sequence_length'],
                'total_blocks': data['total_blocks'],
                'unique_blocks': data['unique_blocks'],
                'codebook_sequence': codebook_str
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_rows)
        
        # Check if CSV exists and append or create new
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.sort_values('motion_id').reset_index(drop=True)
            combined_df.to_csv(csv_path, index=False)
            print(f"📊 Codebook data appended to: {csv_path}")
        else:
            df = df.sort_values('motion_id').reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            print(f"📊 Codebook data saved to: {csv_path}")
        
        # Print summary
        print(f"📈 Codebook Summary:")
        print(f"  - Total motions: {len(df)}")
        print(f"  - Average blocks per motion: {df['total_blocks'].mean():.1f}")
        print(f"  - Average unique blocks: {df['unique_blocks'].mean():.1f}")
    
    # All unnecessary evaluation and statistics methods removed


def main():
    parser = argparse.ArgumentParser(description='Generate PKL files in AMASS format from trained VQVAE model')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt', help='Path to model checkpoint')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to input PKL motion data file')
    parser.add_argument('--motion_ids', type=str, default='0', help='Comma-separated motion IDs to generate')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for PKL files')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file path for codebook sequences')
    
    args = parser.parse_args()
    
    # Parse motion IDs (support ranges like "0-20")
    motion_ids = []
    for x in args.motion_ids.split(','):
        x = x.strip()
        if '-' in x:
            start, end = map(int, x.split('-'))
            motion_ids.extend(range(start, end + 1))
        else:
            motion_ids.append(int(x))
    
    # Initialize generator
    generator = AMASSFormatGenerator(args.config, args.checkpoint, args.input_pkl)
    
    # Generate AMASS format motions
    generated_files, output_dir = generator.generate_amass_format_pkl(motion_ids, args.output_dir, args.csv_file)
    
    print(f"\n=== Generation Complete ===")
    print(f"Generated {len(generated_files)} VQVAE motions in AMASS format")
    print(f"Output directory: {output_dir}")
    print(f"Motion IDs: {motion_ids}")
    print(f"Files created:")
    for file_path in generated_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()

'''
checkpoints/best_model.ckpt \


python scripts/generate_motion_from_vqvae_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "1"


python scripts/generate_motion_from_vqvae_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "0-20"

'''
