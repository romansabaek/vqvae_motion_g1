#!/usr/bin/env python3
"""
Simple script to plot codebook trajectories for selected motion IDs.
Shows 2x1 plot: codebook trajectory and motion changes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import argparse
import os
from typing import List

# Set Times New Roman font and increase default font sizes
# Use font list with fallbacks for Linux compatibility
# matplotlib.rcParams['font.family'] = ['Times New Roman', 'Times', 'serif', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 20

def plot_motion_trajectory(motion_ids: List[int], base_dir: str, save_path: str = None):
    """
    Plot codebook trajectory and motion changes for selected motion IDs.
    Connects multiple CSV files based on motion IDs.
    Creates a 2x1 subplot layout.
    """
    # Load and connect CSV files based on motion IDs
    combined_data = []
    accumulated_frames = 0
    
    for motion_id in motion_ids:
        csv_file = f"codebook_sequence_{motion_id:03d}.csv"
        csv_path = os.path.join(base_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"Warning: File {csv_file} not found, skipping motion ID {motion_id}")
            continue
            
        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_path)
        
        # Add accumulated frame offset
        df['accumulated_frame_idx'] = df['frame_idx'] + accumulated_frames
        df['motion_id'] = motion_id  # Ensure motion_id is set correctly
        df['file_name'] = csv_file
        
        combined_data.append(df)
        
        # Update accumulated frames for next file
        accumulated_frames += df['frame_idx'].max() + 1
    
    if not combined_data:
        print(f"No valid CSV files found for motion IDs: {motion_ids}")
        return
        
    # Combine all data
    motion_data = pd.concat(combined_data, ignore_index=True)
    
    # Create 2x1 subplot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(motion_ids)))
    
    # Plot 1: Codebook trajectory (connected across all motion IDs)
    axes[0].set_title('Codebook Trajectory Over Time (Connected Sequences)')
    
    # Plot the overall trajectory
    axes[0].plot(motion_data['accumulated_frame_idx'], motion_data['codebook_idx'], 
                linewidth=2, alpha=0.8, color='blue', 
                label='Connected Codebook Trajectory')
    
    # Add vertical lines to separate different motion sequences
    current_frame = 0
    for i, motion_id in enumerate(motion_ids):
        data = motion_data[motion_data['motion_id'] == motion_id]
        if not data.empty:
            # Get motion description
            motion_desc = data['original_key'].iloc[0] if 'original_key' in data.columns else f"Motion {motion_id}"
            
            # Add vertical separator line
            if i > 0:  # Don't draw line at the very beginning
                axes[0].axvline(x=current_frame, color='red', linestyle='--', alpha=0.7)
                axes[0].text(current_frame, axes[0].get_ylim()[1] * 0.95, f'Motion {motion_id}', 
                            rotation=90, ha='right', va='top', fontsize=14, color='red')
            
            # Mark codebook changes for this motion
            if 'codebook_changed' in data.columns:
                changes = data[data['codebook_changed'] == True]
                if not changes.empty:
                    axes[0].scatter(changes['accumulated_frame_idx'], changes['codebook_idx'], 
                                   color=colors[i], s=50, alpha=0.9, marker='o', zorder=5)
            
            # Update current frame for next motion
            current_frame += len(data)
    
    axes[0].set_xlabel('Accumulated Frame Index')
    axes[0].set_ylabel('Codebook Index')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of codebook indices used
    axes[1].set_title('Histogram of Codebook Indices Used')
    
    # Create histogram of all codebook indices
    all_codebook_indices = motion_data['codebook_idx'].values
    axes[1].hist(all_codebook_indices, bins=50, alpha=0.7, color='skyblue', 
                edgecolor='black', label='All Codebook Indices')
    
    # Add statistics text
    unique_codebooks = len(set(all_codebook_indices))
    total_frames = len(all_codebook_indices)
    min_codebook = min(all_codebook_indices)
    max_codebook = max(all_codebook_indices)
    
    stats_text = f'Unique Codebooks: {unique_codebooks}\nTotal Frames: {total_frames}\nRange: {min_codebook}-{max_codebook}'
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes, 
                verticalalignment='top', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[1].set_xlabel('Codebook Index')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def analyze_motion_info(motion_ids: List[int], base_dir: str):
    """
    Print motion information for selected IDs across connected CSV files.
    """
    print(f"\nMotion Analysis for Connected Sequences:")
    print("=" * 60)
    
    total_frames = 0
    all_codebooks = []
    
    for motion_id in motion_ids:
        csv_file = f"codebook_sequence_{motion_id:03d}.csv"
        csv_path = os.path.join(base_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"Motion ID {motion_id}: File {csv_file} not found")
            continue
            
        df = pd.read_csv(csv_path)
        motion_desc = df['original_key'].iloc[0] if 'original_key' in df.columns else f"Motion {motion_id}"
        
        print(f"\nMotion ID {motion_id} ({csv_file}):")
        print(f"  Description: {motion_desc}")
        print(f"  Frames: {len(df)}")
        print(f"  Codebook range: {df['codebook_idx'].min()} - {df['codebook_idx'].max()}")
        print(f"  Unique codebooks: {df['codebook_idx'].nunique()}")
        
        if 'codebook_changed' in df.columns:
            changes = df['codebook_changed'].sum()
            print(f"  Codebook changes: {changes}")
        
        total_frames += len(df)
        all_codebooks.extend(df['codebook_idx'].tolist())
    
    # Overall statistics
    print(f"\nOverall Connected Sequence:")
    print(f"  Total frames: {total_frames}")
    print(f"  Overall codebook range: {min(all_codebooks)} - {max(all_codebooks)}")
    print(f"  Total unique codebooks: {len(set(all_codebooks))}")

def main():
    parser = argparse.ArgumentParser(description='Plot connected motion trajectories for selected motion IDs')
    parser.add_argument('--motion-ids', nargs='+', type=int, required=True,
                       help='Motion IDs to analyze and connect (e.g., --motion-ids 0 1 255)')
    parser.add_argument('--output', type=str, default='connected_motion_trajectory.pdf',
                       help='Output file name (default: connected_motion_trajectory.pdf)')
    parser.add_argument('--base-dir', type=str, 
                       default='/home/baekdh/dh_workspace/vqvae_motion_g1/outputs/codebook_sequences',
                       help='Base directory containing CSV files')
    
    args = parser.parse_args()
    
    # Analyze motion information
    analyze_motion_info(args.motion_ids, args.base_dir)
    
    # Create plot
    plot_motion_trajectory(args.motion_ids, args.base_dir, args.output)

if __name__ == "__main__":
    main()

'''

# Connect motion IDs 0, 1, and 255
python scripts/plot_codebook_trajectory.py --motion-ids 145 47

'''