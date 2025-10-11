#!/usr/bin/env python3
"""
Inspect the VQVAE codebook to see if different blocks are actually different.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys

# Add motion_vqvae to path
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader

def inspect_codebook(config_path: str, checkpoint_path: str):
    """Inspect the VQVAE codebook embeddings."""
    
    print(f"🔍 Inspecting VQVAE codebook...")
    
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)
    
    # Initialize agent
    agent = MVQVAEAgent(config=config)
    
    # Load model
    from motion_vqvae.models.models import MotionVQVAE
    
    # Create dummy motion data to get frame_size
    from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
    motion_adapter = MotionDataAdapter(config)
    
    # Load a small amount of data to get frame_size
    dummy_file = "/home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl"
    mocap_data, end_indices, frame_size = motion_adapter.load_motion_data(dummy_file, [0])
    
    agent.config['frame_size'] = frame_size
    
    # Initialize VQVAE model
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
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Codebook size: {config['nb_code']}")
    print(f"📊 Code dimension: {config['code_dim']}")
    
    # Access the codebook
    codebook = agent.model.vqvae.quantizer.codebook  # [nb_code, code_dim]
    print(f"📊 Codebook shape: {codebook.shape}")
    
    # Check similarity between different blocks
    print(f"\n🔍 Codebook Similarity Analysis:")
    
    # Check first 10 blocks
    block_ids_to_check = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for i in range(len(block_ids_to_check)):
        for j in range(i+1, len(block_ids_to_check)):
            block1_id = block_ids_to_check[i]
            block2_id = block_ids_to_check[j]
            
            # Get embeddings
            emb1 = codebook[block1_id]  # [code_dim]
            emb2 = codebook[block2_id]  # [code_dim]
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
            
            # Compute L2 distance
            l2_distance = torch.norm(emb1 - emb2).item()
            
            print(f"  Block {block1_id} vs Block {block2_id}:")
            print(f"    Cosine similarity: {similarity:.4f}")
            print(f"    L2 distance: {l2_distance:.4f}")
            
            if similarity > 0.99:
                print(f"    🚨 ALMOST IDENTICAL!")
            elif similarity > 0.9:
                print(f"    ⚠️  Very similar")
            elif similarity > 0.5:
                print(f"    ✅ Moderately different")
            else:
                print(f"    ✅ Very different")
    
    # Check codebook statistics
    print(f"\n📈 Codebook Statistics:")
    print(f"  Mean embedding norm: {torch.norm(codebook, dim=1).mean():.4f}")
    print(f"  Std embedding norm: {torch.norm(codebook, dim=1).std():.4f}")
    print(f"  Min embedding norm: {torch.norm(codebook, dim=1).min():.4f}")
    print(f"  Max embedding norm: {torch.norm(codebook, dim=1).max():.4f}")
    
    # Check if any embeddings are identical
    print(f"\n🔍 Checking for identical embeddings...")
    identical_pairs = []
    for i in range(min(100, config['nb_code'])):  # Check first 100 blocks
        for j in range(i+1, min(100, config['nb_code'])):
            emb1 = codebook[i]
            emb2 = codebook[j]
            similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
            if similarity > 0.999:  # Almost identical
                identical_pairs.append((i, j, similarity))
    
    if identical_pairs:
        print(f"  🚨 Found {len(identical_pairs)} almost identical pairs:")
        for block1, block2, sim in identical_pairs[:10]:  # Show first 10
            print(f"    Block {block1} vs Block {block2}: {sim:.6f}")
    else:
        print(f"  ✅ No identical embeddings found in first 100 blocks")

def main():
    parser = argparse.ArgumentParser(description='Inspect VQVAE codebook embeddings')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='outputs/run_0_300/best_model.ckpt', help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    inspect_codebook(args.config, args.checkpoint)

if __name__ == "__main__":
    main()

'''
python scripts/inspect_codebook.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300/best_model.ckpt
'''
