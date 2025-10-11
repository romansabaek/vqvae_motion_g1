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
        
        # Load motion data in MVQ format (364 dimensions)
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
    
    def generate_amass_format_pkl(self, motion_ids: List[int], output_file: str = None, csv_file: str = None):
        """SIMPLIFIED: Generate PKL files in AMASS format for specified motion IDs."""
        print(f"\n=== Generating AMASS Format PKL File ===")
        
        # Create simple filename
        if output_file is None:
            output_file = f"./outputs/vqvae_motion_{motion_ids[0]}.pkl"
        
        # Create output directory
        output_path = Path(output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_motions = {}
        codebook_data = []  # Store codebook sequences for CSV
        
        for motion_id in motion_ids:
            if motion_id >= len(self.original_keys):
                print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
                continue
                
            print(f"Processing motion ID {motion_id}...")
            
            # Generate reconstructed motion in AMASS format
            reconstructed_motion, codebook_sequence = self._generate_single_motion_amass_format(motion_id)
            
            if reconstructed_motion is not None:
                # Use original motion key for consistency
                original_key = self.original_keys[motion_id]
                generated_motions[original_key] = reconstructed_motion
                print(f"Generated motion: {original_key}")
                
                # Store codebook sequence data
                codebook_data.append({
                    'motion_id': motion_id,
                    'motion_key': original_key,
                    'sequence_length': reconstructed_motion['dof'].shape[0],
                    'codebook_sequence': codebook_sequence,
                    'unique_blocks': len(np.unique(codebook_sequence)),
                    'total_blocks': len(codebook_sequence)
                })
        
        # Save in AMASS format
        joblib.dump(generated_motions, output_file)
        print(f"\nSaved AMASS format PKL file to: {output_file}")
        print(f"Generated {len(generated_motions)} VQVAE motions in AMASS format")
        
        # Save codebook sequences to CSV
        if csv_file and codebook_data:
            self._save_codebook_csv(codebook_data, csv_file)
        
        return generated_motions, output_file
    
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
        SIMPLIFIED: Convert VQVAE output directly to AMASS format.
        The VQVAE decoder outputs local frame features - use them directly with minimal processing.
        """
        # Get original motion for reference format
        original_amass = self.original_motions[self.original_keys[motion_id]]
        
        mvq_np = reconstructed_motion.cpu().numpy()
        seq_len = mvq_np.shape[0]
        
        # Extract MVQ components
        root_deltas = mvq_np[:, 0:4]  # [T, 4] - dx, dy, dz, dyaw
        joint_positions = mvq_np[:, 4:76]  # [T, 72] - 24 joints * 3
        
        # REVERT TO WORKING APPROACH: Use global features (like before local features change)
        # This was working before with only minor roll/pitch issues
        
        # Simple integration of root deltas (global approach)
        root_trans_offset = np.zeros((seq_len, 3))
        root_trans_offset[0] = original_amass['root_trans_offset'][0]
        
        for i in range(1, seq_len):
            # Simple global delta integration
            root_trans_offset[i] = root_trans_offset[i-1] + root_deltas[i, :3]
        
        # Simple root rotation reconstruction (stabilize roll/pitch)
        root_rot = np.zeros((seq_len, 4))
        root_rot[0] = original_amass['root_rot'][0]
        
        for i in range(1, seq_len):
            # Stabilize roll and pitch, only use yaw delta
            prev_rot = root_rot[i-1]
            yaw_delta = root_deltas[i, 3]
            
            # Extract current yaw
            current_yaw = np.arctan2(prev_rot[3], prev_rot[0]) * 2
            new_yaw = current_yaw + yaw_delta
            
            # Create new quaternion with stabilized roll/pitch (close to zero)
            root_rot[i] = np.array([np.cos(new_yaw/2), 0, 0, np.sin(new_yaw/2)])
        
        # Use joint positions directly (like before)
        smpl_joints = joint_positions.reshape(seq_len, 24, 3)
        
        # Use original DOF structure (this was working before)
        dof = original_amass['dof'].copy()
        
        # Simple contact mask (foot height < threshold)
        contact_mask = np.zeros((seq_len, 2))
        if smpl_joints.shape[1] >= 2:
            contact_mask[:, 0] = smpl_joints[:, 0, 2] < 0.1  # Left foot
            contact_mask[:, 1] = smpl_joints[:, 1, 2] < 0.1  # Right foot
        
        # Create AMASS format structure
        amass_format_motion = {
            'root_trans_offset': root_trans_offset.astype(np.float32),
            'root_rot': root_rot.astype(np.float32),
            'pose_aa': original_amass['pose_aa'].copy().astype(np.float32),  # Keep original
            'smpl_joints': smpl_joints.astype(np.float32),
            'contact_mask': contact_mask.astype(np.float32),
            'dof': dof.astype(np.float32),
            'fps': 30
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
    parser.add_argument('--output_file', type=str, default=None, help='Output PKL file path')
    parser.add_argument('--csv_file', type=str, default=None, help='CSV file path for codebook sequences')
    
    args = parser.parse_args()
    
    # Parse motion IDs
    motion_ids = [int(x.strip()) for x in args.motion_ids.split(',')]
    
    # Initialize generator
    generator = AMASSFormatGenerator(args.config, args.checkpoint, args.input_pkl)
    
    # Generate AMASS format motions
    generated_motions, output_file = generator.generate_amass_format_pkl(motion_ids, args.output_file, args.csv_file)
    
    print(f"\n=== Generation Complete ===")
    print(f"Generated {len(generated_motions)} VQVAE motions in AMASS format")
    print(f"Output file: {output_file}")
    print(f"Motion IDs: {motion_ids}")


if __name__ == "__main__":
    main()

'''
checkpoints/best_model.ckpt \

python scripts/generate_motion_from_vqvae_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "0-300" \
    --output_file ./outputs/vqvae_motion_0-300.pkl \
    --csv_file ./outputs/vqvae_motion_0-300_codebook.csv



python scripts/generate_motion_from_vqvae_s2.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "1"

'''
