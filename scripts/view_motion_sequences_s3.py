#!/usr/bin/env python3
"""
Simple script: Given motion ID from AMASS, analyze and output the codebook sequence.
"""

import argparse
from pathlib import Path
import sys

# Add motion_vqvae to path
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
import joblib
import torch


def get_codebook_sequence(config_path: str, checkpoint_path: str, input_pkl_file: str, motion_id: int):
    """
    Get the codebook sequence for a specific AMASS motion ID.
    """
    print(f"🎯 Motion ID: {motion_id}")
    print("=" * 30)
    
    # Load config and initialize components
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)
    
    # Load original AMASS data
    original_motions = joblib.load(input_pkl_file)
    original_keys = list(original_motions.keys())
    
    if motion_id >= len(original_keys):
        print(f"❌ Motion ID {motion_id} out of range (max: {len(original_keys)-1})")
        return None
    
    # Get motion info
    motion_key = original_keys[motion_id]
    original_motion = original_motions[motion_key]
    
    print(f"Motion Key: {motion_key}")
    print(f"Duration: {original_motion['dof'].shape[0] / 30.0:.2f}s ({original_motion['dof'].shape[0]} frames)")
    
    # Initialize agent and motion adapter
    agent = MVQVAEAgent(config=config)
    motion_adapter = MotionDataAdapter(config)
    
    # Load motion data for this specific motion
    mocap_data, end_indices, frame_size = motion_adapter.load_motion_data(input_pkl_file, [motion_id])
    
    # Setup agent with motion data
    agent.mocap_data = mocap_data
    agent.end_indices = end_indices
    agent.frame_size = frame_size
    
    # Calculate normalization statistics
    mean = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)
    std[std == 0] = 1.0
    
    agent.mean = mean
    agent.std = std
    
    # Initialize model
    from motion_vqvae.models.models import MotionVQVAE
    
    agent.config['frame_size'] = frame_size
    agent.model = MotionVQVAE(
        agent,
        config['nb_code'],
        config['code_dim'],
        config['output_emb_width'],
        config['down_t'],
        config['stride_t'],
        config['width'],
        config['depth'],
        config['dilation_growth_rate'],
        config['vq_act'],
        config['vq_norm']
    ).to(agent.device)
    
    # Load trained model
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.model.load_state_dict(checkpoint['model'])
    agent.model.eval()
    
    # Get codebook sequence
    with torch.no_grad():
        _, _, codebook_sequence = agent.evaluate_policy_rec(torch.tensor(0))
    
    # Convert to numpy
    codebook_sequence_np = codebook_sequence.cpu().numpy() if isinstance(codebook_sequence, torch.Tensor) else codebook_sequence
    
    print(f"Codebook Sequence Length: {len(codebook_sequence_np)} blocks")
    print(f"Unique Blocks: {len(set(codebook_sequence_np))}")
    print()
    
    # Output the sequence
    print("🔢 Codebook Sequence:")
    print(",".join(map(str, codebook_sequence_np)))
    print()
    
    # Also show in readable format
    print("📋 Codebook Sequence (readable):")
    chunk_size = 20
    for i in range(0, len(codebook_sequence_np), chunk_size):
        chunk = codebook_sequence_np[i:i+chunk_size]
        chunk_str = ' '.join(f"{block:3d}" for block in chunk)
        print(f"  {i:3d}-{i+len(chunk)-1:3d}: {chunk_str}")
    
    return codebook_sequence_np


def view_motion_sequences(csv_file: str, motion_ids: list = None, max_sequences: int = 10):
    """View motion block sequences for specified motion IDs (legacy function)."""
    
    # Load the analysis results
    df = pd.read_csv(csv_file)
    
    print(f"📊 Motion Block Sequences Analysis")
    print(f"=================================")
    print(f"Total motions in dataset: {len(df)}")
    print()
    
    # If no specific motion IDs provided, show first few
    if motion_ids is None:
        motion_ids = df['motion_id'].head(max_sequences).tolist()
    
    # Filter for requested motion IDs
    available_ids = df['motion_id'].tolist()
    requested_ids = [mid for mid in motion_ids if mid in available_ids]
    missing_ids = [mid for mid in motion_ids if mid not in available_ids]
    
    if missing_ids:
        print(f"⚠️  Motion IDs not found: {missing_ids}")
        print()
    
    if not requested_ids:
        print("❌ No valid motion IDs found!")
        return
    
    print(f"🎯 Showing sequences for {len(requested_ids)} motions:")
    print()
    
    # Display sequences for each motion
    for motion_id in requested_ids:
        motion_data = df[df['motion_id'] == motion_id].iloc[0]
        
        print(f"Motion ID: {motion_id}")
        print(f"Motion Key: {motion_data['motion_key']}")
        print(f"Duration: {motion_data['duration']:.2f}s")
        print(f"Total Blocks: {motion_data['total_blocks']}")
        print(f"Unique Blocks: {motion_data['unique_blocks']}")
        print(f"Diversity: {motion_data['block_diversity']:.3f}")
        print(f"Most Common Block: {motion_data['most_common_block']} (appears {motion_data['most_common_block_count']} times)")
        
        # Parse and display the sequence
        sequence_str = motion_data['codebook_sequence']
        sequence = [int(x.strip()) for x in sequence_str.split(',')]
        
        print(f"Motion Block Sequence ({len(sequence)} blocks):")
        
        # Display sequence in chunks for readability
        chunk_size = 20
        for i in range(0, len(sequence), chunk_size):
            chunk = sequence[i:i+chunk_size]
            chunk_str = ', '.join(f"{block:3d}" for block in chunk)
            print(f"  Blocks {i:3d}-{i+len(chunk)-1:3d}: {chunk_str}")
        
        print()
        print("-" * 80)
        print()




def main():
    parser = argparse.ArgumentParser(description='Get codebook sequence for AMASS motion ID')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', 
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt', 
                       help='Path to model checkpoint')
    parser.add_argument('--input_pkl', type=str, required=True,
                       help='Path to input PKL motion data file')
    parser.add_argument('--motion_id', type=int, required=True,
                       help='Motion ID to analyze')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        return
        
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return
        
    if not Path(args.input_pkl).exists():
        print(f"❌ Input PKL file not found: {args.input_pkl}")
        return
    
    # Get codebook sequence for the motion ID
    get_codebook_sequence(args.config, args.checkpoint, args.input_pkl, args.motion_id)


if __name__ == "__main__":
    main()

'''
# Get codebook sequence for motion ID 1
python scripts/view_motion_sequences_s3.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_id 1

# Get codebook sequence for motion ID 10
python scripts/view_motion_sequences_s3.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_id 10
'''