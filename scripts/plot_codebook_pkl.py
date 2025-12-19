#!/usr/bin/env python3
"""
Plot original vs reconstructed motion from codebook sequence PKL files.

Loads PKL files created by generate_motion_from_vqvae_s2.py and creates plots
comparing original and reconstructed motion features.
"""

import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional

# Set matplotlib parameters
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16

# Motion feature indices (from MotionDataAdapter)
ROOT_DELTAS_START = 0
ROOT_DELTAS_END = 4  # dx, dy, dz, dyaw
DOF_POSITIONS_START = 4
DOF_POSITIONS_END = 27  # 23 DOF positions
DOF_VELOCITIES_START = 27
DOF_VELOCITIES_END = 50  # 23 DOF velocities

ROOT_DELTA_LABELS = ["dx", "dy", "dz", "dyaw"]
FPS = 30.0


def load_pkl_data(file_path: Path) -> Optional[dict]:
    """Load codebook data from PKL file and print keys."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None
    
    if file_path.suffix != '.pkl':
        print(f"Error: File must be a .pkl file, got: {file_path.suffix}")
        return None
    
    try:
        data = joblib.load(file_path)
        
        # Print keys and their information
        print(f"\n{'='*60}")
        print(f"PKL File: {file_path.name}")
        print(f"{'='*60}")
        print(f"Loaded data contains {len(data)} keys:\n")
        
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  - {key:30s}: shape {str(value.shape):20s} dtype {value.dtype}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"  - {key:30s}: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"  - {key:30s}: {type(value).__name__} of length {len(value)}")
            else:
                print(f"  - {key:30s}: {type(value).__name__}")
        
        print(f"{'='*60}\n")
        
        return data
    except Exception as e:
        print(f"Error loading PKL file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_original_vs_reconstructed(data: dict, output_path: Optional[Path] = None, 
                                   show_plot: bool = True, num_dof_to_plot: int = 5):
    """
    Plot original vs reconstructed motion features.
    
    Args:
        data: Dictionary containing codebook data from PKL
        output_path: Path to save the plot
        show_plot: Whether to display the plot
        num_dof_to_plot: Number of DOF to plot (default: 5)
    """
    # Check for required keys
    required_keys = ['original_motion', 'reconstructed_motion']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        print(f"Error: Missing required keys: {missing_keys}")
        print(f"Available keys: {list(data.keys())}")
        return
    
    original_motion = data['original_motion']
    reconstructed_motion = data['reconstructed_motion']
    motion_id = data.get('motion_id', 'Unknown')
    original_key = data.get('original_key', 'Unknown')
    
    num_frames = min(original_motion.shape[0], reconstructed_motion.shape[0])
    time_axis = np.arange(num_frames) / FPS  # Time in seconds
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    
    # === First Row: Root Deltas ===
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f'Root Deltas - Motion ID: {motion_id} ({original_key})', 
                  fontsize=14, fontweight='bold')
    
    for i in range(4):
        ax1.plot(time_axis, original_motion[:num_frames, ROOT_DELTAS_START + i], 
                label=f'Original {ROOT_DELTA_LABELS[i]}', linewidth=2, alpha=0.8)
        ax1.plot(time_axis, reconstructed_motion[:num_frames, ROOT_DELTAS_START + i], 
                '--', label=f'Reconstructed {ROOT_DELTA_LABELS[i]}', linewidth=2, alpha=0.8)
    
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(ncol=2, loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # === Second Row: DOF Positions ===
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_title(f'DOF Positions (First {num_dof_to_plot})', fontsize=14, fontweight='bold')
    
    num_dof_available = min(num_dof_to_plot, DOF_POSITIONS_END - DOF_POSITIONS_START)
    for i in range(num_dof_available):
        ax2.plot(time_axis, original_motion[:num_frames, DOF_POSITIONS_START + i], 
                label=f'Original DOF[{i}]', linewidth=1.5, alpha=0.7)
        ax2.plot(time_axis, reconstructed_motion[:num_frames, DOF_POSITIONS_START + i], 
                '--', label=f'Reconstructed DOF[{i}]', linewidth=1.5, alpha=0.7)
    
    ax2.set_ylabel('Position (rad)', fontsize=12)
    ax2.legend(ncol=2, loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === Third Row: Codebook Changes (if available) ===
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_title('Codebook Changes Over Time', fontsize=14, fontweight='bold')
    
    if 'codebook_per_frame' in data and 'codebook_changed' in data:
        codebook_per_frame = data['codebook_per_frame']
        codebook_changed = data['codebook_changed']
        
        # Plot codebook indices
        ax3.plot(time_axis, codebook_per_frame[:num_frames], 
                linewidth=2, color='blue', alpha=0.7, label='Codebook Index')
        
        # Mark codebook changes
        change_indices = np.where(codebook_changed[:num_frames])[0]
        if len(change_indices) > 0:
            change_times = time_axis[change_indices]
            change_values = codebook_per_frame[change_indices]
            ax3.scatter(change_times, change_values, 
                       color='red', s=50, marker='o', zorder=5, 
                       label=f'Codebook Changes ({len(change_indices)})', alpha=0.8)
        
        ax3.set_ylabel('Codebook Index', fontsize=12)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text
        unique_codebooks = len(np.unique(codebook_per_frame[:num_frames]))
        min_codebook = np.min(codebook_per_frame[:num_frames])
        max_codebook = np.max(codebook_per_frame[:num_frames])
        num_changes = np.sum(codebook_changed[:num_frames])
        
        stats_text = (f'Unique Codebooks: {unique_codebooks} | '
                     f'Range: {min_codebook}-{max_codebook} | '
                     f'Changes: {num_changes}')
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No codebook data available', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot original vs reconstructed motion from codebook sequence PKL files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to codebook sequence PKL file (e.g., codebook_sequence_001.pkl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot (default: same directory as input with .png extension)"
    )
    parser.add_argument(
        "--num-dof",
        type=int,
        default=5,
        help="Number of DOF to plot (default: 5)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Load PKL data (will print keys)
    print(f"Loading PKL file: {input_path}")
    data = load_pkl_data(input_path)
    
    if data is None:
        print("Failed to load PKL data.")
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_plot.png"
    
    # Create plot
    plot_original_vs_reconstructed(
        data,
        output_path=output_path,
        show_plot=not args.no_show,
        num_dof_to_plot=args.num_dof
    )


if __name__ == "__main__":
    main()


'''
Example usage:

# Plot from PKL file
python scripts/plot_codebook_pkl.py \
  --input outputs/codebook_sequences/codebook_sequence_000.pkl \
  --output evaluation_plots_v2/motion_1_original_vs_reconstructed.png


python scripts/plot_codebook_pkl.py \
  --input outputs/codebook_sequences_npy/codebook_sequence_000.pkl \
  --output evaluation_plots_v2/motion_1_original_vs_reconstructed_npy.png


'''

