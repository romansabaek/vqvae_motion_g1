#!/usr/bin/env python3
"""
Compare sequence data (e.g., 108_8) with individual motion data (e.g., 8 and 108).
Analyzes feature differences, statistics, and distributions.
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent dir so imports work similarly to your training script
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from motion_vqvae.config_loader import ConfigLoader
from scripts.eval_policy_id_prediction_vqvae import (
    load_npy_motion_data,
    parse_policy_ids_from_filename,
    convert_motion_ids_to_policy_ids
)


def extract_motion_segments_from_sequence(
    sequence_data: np.ndarray,
    sequence_policy_ids: np.ndarray,
    sequence_motion_ids: np.ndarray,
    target_motion_ids: list
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract individual motion segments from sequence data based on motion_id.
    
    Args:
        sequence_data: (T, 50) feature array
        sequence_policy_ids: (T,) policy ID array
        sequence_motion_ids: (T,) motion ID array from raw npy file
        target_motion_ids: List of motion IDs to extract (e.g., [108, 8])
    
    Returns:
        Dict mapping motion_id -> (features, policy_ids)
    """
    segments = {}
    unique_motion_ids = np.unique(sequence_motion_ids)
    
    print(f"  Unique motion_ids in sequence: {unique_motion_ids.tolist()}")
    print(f"  Target motion_ids: {target_motion_ids}")
    
    # Map target motion_ids to sequence motion_ids
    # The sequence might have different motion_id values, so we need to map them
    if len(unique_motion_ids) == len(target_motion_ids):
        # Direct mapping: first unique -> first target, etc.
        for i, target_id in enumerate(target_motion_ids):
            seq_motion_id = unique_motion_ids[i]
            mask = sequence_motion_ids == seq_motion_id
            if mask.any():
                segments[target_id] = (sequence_data[mask], sequence_policy_ids[mask])
                print(f"  Mapped sequence motion_id {seq_motion_id} -> target {target_id} ({mask.sum()} frames)")
    else:
        # Try to match by order or find closest match
        for target_id in target_motion_ids:
            # Find the segment that best matches
            # For now, split by order
            if len(unique_motion_ids) > 0:
                # Use first unique motion_id for first target, etc.
                idx = target_motion_ids.index(target_id)
                if idx < len(unique_motion_ids):
                    seq_motion_id = unique_motion_ids[idx]
                    mask = sequence_motion_ids == seq_motion_id
                    if mask.any():
                        segments[target_id] = (sequence_data[mask], sequence_policy_ids[mask])
                        print(f"  Mapped sequence motion_id {seq_motion_id} -> target {target_id} ({mask.sum()} frames)")
    
    return segments


def load_sequence_with_motion_ids(npy_path: Path, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sequence data and return features, policy_ids, and motion_ids.
    """
    # Load raw data to get motion_ids
    data = np.load(npy_path, allow_pickle=True)
    motion_ids = data[:, -1].astype(int)
    
    # Load features using the same method
    features, policy_ids = load_npy_motion_data(npy_path, device=device)
    
    return features, policy_ids, motion_ids


def compute_statistics(features: np.ndarray, name: str) -> Dict:
    """Compute comprehensive statistics for features."""
    return {
        'name': name,
        'shape': features.shape,
        'mean': np.mean(features, axis=0),
        'std': np.std(features, axis=0),
        'min': np.min(features, axis=0),
        'max': np.max(features, axis=0),
        'median': np.median(features, axis=0),
        'q25': np.percentile(features, 25, axis=0),
        'q75': np.percentile(features, 75, axis=0),
    }


def compare_features(
    seq_features: np.ndarray,
    seq_name: str,
    individual_features: Dict[int, np.ndarray],
    output_dir: Path
):
    """Compare and visualize feature differences."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    seq_stats = compute_statistics(seq_features, seq_name)
    individual_stats = {
        motion_id: compute_statistics(features, f"motion_{motion_id}")
        for motion_id, features in individual_features.items()
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("FEATURE STATISTICS COMPARISON")
    print("="*80)
    
    print(f"\nSequence Data: {seq_name}")
    print(f"  Shape: {seq_stats['shape']}")
    print(f"  Mean (per feature): {seq_stats['mean'][:10]}...")  # Show first 10
    print(f"  Std (per feature): {seq_stats['std'][:10]}...")
    
    for motion_id, stats in individual_stats.items():
        print(f"\nIndividual Motion {motion_id}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean (per feature): {stats['mean'][:10]}...")
        print(f"  Std (per feature): {stats['std'][:10]}...")
    
    # Compare concatenated individual motions vs sequence
    if len(individual_features) > 1:
        concat_features = np.concatenate([features for features in individual_features.values()], axis=0)
        concat_stats = compute_statistics(concat_features, "concatenated_individual")
        
        print(f"\nConcatenated Individual Motions:")
        print(f"  Shape: {concat_stats['shape']}")
        print(f"  Mean (per feature): {concat_stats['mean'][:10]}...")
        print(f"  Std (per feature): {concat_stats['std'][:10]}...")
        
        # Compute differences
        mean_diff = np.abs(seq_stats['mean'] - concat_stats['mean'])
        std_diff = np.abs(seq_stats['std'] - concat_stats['std'])
        
        print(f"\nDifferences (Sequence vs Concatenated):")
        print(f"  Mean absolute difference: {np.mean(mean_diff):.6f} (max: {np.max(mean_diff):.6f})")
        print(f"  Std absolute difference: {np.mean(std_diff):.6f} (max: {np.max(std_diff):.6f})")
    
    # Create visualizations
    feature_names = [
        'root_dx', 'root_dy', 'root_dz', 'root_dyaw',
        *[f'dof_pos_{i}' for i in range(23)],
        *[f'dof_vel_{i}' for i in range(23)]
    ]
    
    # Plot 1: Feature mean comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean comparison
    ax = axes[0, 0]
    ax.plot(seq_stats['mean'], label='Sequence', alpha=0.7, linewidth=2)
    for motion_id, stats in individual_stats.items():
        ax.plot(stats['mean'], label=f'Motion {motion_id}', alpha=0.7, linestyle='--')
    if len(individual_features) > 1:
        ax.plot(concat_stats['mean'], label='Concatenated', alpha=0.7, linestyle=':')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Value')
    ax.set_title('Feature Mean Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Std comparison
    ax = axes[0, 1]
    ax.plot(seq_stats['std'], label='Sequence', alpha=0.7, linewidth=2)
    for motion_id, stats in individual_stats.items():
        ax.plot(stats['std'], label=f'Motion {motion_id}', alpha=0.7, linestyle='--')
    if len(individual_features) > 1:
        ax.plot(concat_stats['std'], label='Concatenated', alpha=0.7, linestyle=':')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Std Value')
    ax.set_title('Feature Std Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution comparison for key features
    ax = axes[1, 0]
    key_feature_idx = 0  # root_dx
    ax.hist(seq_features[:, key_feature_idx], bins=50, alpha=0.5, label='Sequence', density=True)
    for motion_id, features in individual_features.items():
        ax.hist(features[:, key_feature_idx], bins=50, alpha=0.5, label=f'Motion {motion_id}', density=True)
    ax.set_xlabel(f'Feature {key_feature_idx} ({feature_names[key_feature_idx]})')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution: {feature_names[key_feature_idx]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution comparison for DOF position
    ax = axes[1, 1]
    dof_feature_idx = 10  # dof_pos_6
    ax.hist(seq_features[:, dof_feature_idx], bins=50, alpha=0.5, label='Sequence', density=True)
    for motion_id, features in individual_features.items():
        ax.hist(features[:, dof_feature_idx], bins=50, alpha=0.5, label=f'Motion {motion_id}', density=True)
    ax.set_xlabel(f'Feature {dof_feature_idx} ({feature_names[dof_feature_idx]})')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution: {feature_names[dof_feature_idx]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "feature_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # Plot 2: Feature-wise differences heatmap
    if len(individual_features) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Mean difference heatmap
        ax = axes[0]
        mean_diff_reshaped = mean_diff.reshape(1, -1)
        im = ax.imshow(mean_diff_reshaped, aspect='auto', cmap='viridis')
        ax.set_xlabel('Feature Index')
        ax.set_title('Mean Absolute Difference (Sequence vs Concatenated)')
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
        
        # Std difference heatmap
        ax = axes[1]
        std_diff_reshaped = std_diff.reshape(1, -1)
        im = ax.imshow(std_diff_reshaped, aspect='auto', cmap='viridis')
        ax.set_xlabel('Feature Index')
        ax.set_title('Std Absolute Difference (Sequence vs Concatenated)')
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        diff_plot_path = output_dir / "feature_differences.png"
        plt.savefig(diff_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved difference plot to: {diff_plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequence data with individual motion data"
    )
    parser.add_argument("--seq_file", type=str, required=True,
                       help="Path to sequence npy file (e.g., saved_desired_states_108_8_fg.npy)")
    parser.add_argument("--individual_dir", type=str, required=True,
                       help="Path to directory containing individual motion npy files")
    parser.add_argument("--motion_ids", type=int, nargs='+', required=True,
                       help="Motion IDs to compare (e.g., 108 8)")
    parser.add_argument("--output_dir", type=str, default="./comparison_output",
                       help="Directory to save comparison results")
    parser.add_argument("--config", type=str, default="configs/agent_codebook_switching.yaml",
                       help="Path to config file (for device setup)")
    
    args = parser.parse_args()
    
    # Setup device
    config_loader = ConfigLoader(args.config)
    cfg = config_loader.to_dict()
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Using device: {device}")
    print(f"Sequence file: {args.seq_file}")
    print(f"Individual dir: {args.individual_dir}")
    print(f"Motion IDs: {args.motion_ids}")
    
    # Load sequence data
    seq_path = Path(args.seq_file)
    if not seq_path.exists():
        print(f"Error: Sequence file not found: {seq_path}")
        return
    
    print(f"\nLoading sequence data: {seq_path.name}")
    seq_features, seq_policy_ids, seq_motion_ids = load_sequence_with_motion_ids(seq_path, device)
    print(f"  Sequence shape: {seq_features.shape}")
    print(f"  Unique motion_ids in sequence: {np.unique(seq_motion_ids).tolist()}")
    print(f"  Unique policy_ids in sequence: {np.unique(seq_policy_ids).tolist()}")
    
    # Load individual motion files
    individual_dir = Path(args.individual_dir)
    individual_features = {}
    
    for motion_id in args.motion_ids:
        individual_file = individual_dir / f"saved_desired_states_{motion_id}.npy"
        if not individual_file.exists():
            print(f"Warning: Individual file not found: {individual_file}")
            continue
        
        print(f"\nLoading individual motion {motion_id}: {individual_file.name}")
        features, policy_ids = load_npy_motion_data(individual_file, device=device)
        individual_features[motion_id] = features
        print(f"  Shape: {features.shape}")
        print(f"  Unique policy_ids: {np.unique(policy_ids).tolist()}")
    
    if not individual_features:
        print("Error: No individual motion files loaded")
        return
    
    # Extract segments from sequence data
    print(f"\nExtracting motion segments from sequence data...")
    seq_segments = extract_motion_segments_from_sequence(
        seq_features, seq_policy_ids, seq_motion_ids, args.motion_ids
    )
    
    print(f"Extracted segments:")
    for motion_id, (features, policy_ids) in seq_segments.items():
        print(f"  Motion {motion_id}: shape={features.shape}, policy_ids={np.unique(policy_ids).tolist()}")
    
    # Compare sequence segments with individual motions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COMPARING SEQUENCE SEGMENTS vs INDIVIDUAL MOTIONS")
    print(f"{'='*80}")
    
    for motion_id in args.motion_ids:
        if motion_id not in individual_features or motion_id not in seq_segments:
            continue
        
        seq_seg_features, seq_seg_policy_ids = seq_segments[motion_id]
        ind_features = individual_features[motion_id]
        
        print(f"\nMotion {motion_id}:")
        print(f"  Sequence segment shape: {seq_seg_features.shape}")
        print(f"  Individual shape: {ind_features.shape}")
        
        # Compare statistics
        seq_seg_stats = compute_statistics(seq_seg_features, f"seq_segment_{motion_id}")
        ind_stats = compute_statistics(ind_features, f"individual_{motion_id}")
        
        mean_diff = np.abs(seq_seg_stats['mean'] - ind_stats['mean'])
        std_diff = np.abs(seq_seg_stats['std'] - ind_stats['std'])
        
        print(f"  Mean absolute difference: {np.mean(mean_diff):.6f} (max: {np.max(mean_diff):.6f})")
        print(f"  Std absolute difference: {np.mean(std_diff):.6f} (max: {np.max(std_diff):.6f})")
    
    # Compare full sequence vs concatenated individual motions
    print(f"\n{'='*80}")
    print("COMPARING FULL SEQUENCE vs CONCATENATED INDIVIDUAL MOTIONS")
    print(f"{'='*80}")
    
    compare_features(
        seq_features=seq_features,
        seq_name=seq_path.stem,
        individual_features=individual_features,
        output_dir=output_dir
    )
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

'''
# Example usage:

# Compare sequence 108_8 with individual motions 108 and 8
python scripts/compare_seq_vs_individual_data.py \
  --seq_file /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_training/saved_desired_states_108_8_fg.npy \
  --individual_dir /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy \
  --motion_ids 108 8 \
  --output_dir ./comparison_output_108_8 \
  --config configs/agent_codebook_switching.yaml

'''

