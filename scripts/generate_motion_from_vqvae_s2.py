#!/usr/bin/env python3
"""
Generate motion files in AMASS format (XYZW quaternion) from a trained Motion-VQVAE.
Supports both NPY and PKL input/output formats.

Input formats:
  - NPY: Directory containing saved_desired_states_*.npy files
  - PKL: Original AMASS PKL file (dict format with motion sequences)

Output formats:
  - NPY: NPY files in AMASS-like format
  - PKL: PKL files in AMASS format

Global/world export only: integrates local root deltas to world (AMASS-like globals).
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import glob
import re

import numpy as np
import torch
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Add motion_vqvae to path (repo layout assumption)
sys.path.append(str(Path(__file__).parent.parent))

from scripts.vqvae_gen_init import (
    load_config_and_agent,
    load_original_pkl,
    infer_frame_size,
    initialize_model,
    ensure_stats,
    parse_motion_ids,
    quat_mul_xyzw,
    quat_rotate_xyzw_numpy,
    convert_to_amass_global_from_local,
    generate_motion_pkl_files,
)

# Set matplotlib parameters
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16

FPS = 30.0  # Frames per second for time axis


def load_original_npy(input_npy_file: str) -> Tuple[Dict, List[str]]:
    """
    Load npy file and convert to AMASS-like format.
    Format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
    Root rotation is in WXYZ format, converted to XYZW for AMASS compatibility.
    
    Note: npy files typically contain a single motion, so we use "motion_0" as the key
    to match the list index (motion_id=0) used elsewhere in the code.
    """
    trajectory_data = np.load(input_npy_file)
    
    # Format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
    # Root rotation is WXYZ, need to convert to XYZW
    num_frames = trajectory_data.shape[0]
    
    # Extract data
    time_stamps = trajectory_data[:, 0]
    root_pos = trajectory_data[:, 1:4]  # [T, 3]
    root_rot_wxyz = trajectory_data[:, 4:8]  # [T, 4] in WXYZ format
    dof_pos = trajectory_data[:, 8:31]  # [T, 23]
    # Note: We ignore the motion_id in the file and use index 0 for consistency
    
    # Convert root rotation from WXYZ to XYZW
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]  # [x, y, z, w] = [1, 2, 3, 0]
    
    # Calculate fps from time stamps
    if len(time_stamps) > 1:
        dt = np.mean(np.diff(time_stamps))
        fps = 1.0 / dt if dt > 0 else 30.0
    else:
        fps = 30.0
    
    # Create AMASS-like motion dict
    # Use "motion_0" as key to match the list index (motion_id=0) used in the code
    motion_key = "motion_0"
    motion_data = {
        "root_trans_offset": root_pos.astype(np.float32),
        "root_rot": root_rot_xyzw.astype(np.float32),  # XYZW format
        "dof": dof_pos.astype(np.float32),
        "fps": fps,
        "pose_aa": np.zeros((num_frames, 72), dtype=np.float32),  # Placeholder for compatibility
        "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),  # Placeholder for compatibility
    }
    
    motions = {motion_key: motion_data}
    keys = [motion_key]
    
    return motions, keys


def find_npy_files(input_dir: str, motion_ids: Optional[List[int]] = None) -> List[Tuple[str, int]]:
    """
    Find npy files in the input directory.
    
    Args:
        input_dir: Directory containing npy files
        motion_ids: Optional list of motion IDs to filter. If None, process all files.
    
    Returns:
        List of tuples (file_path, motion_id) for each matching file.
        Files are matched by pattern: saved_desired_states_{motion_id}.npy
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all npy files matching the pattern saved_desired_states_*.npy
    pattern = input_path / "saved_desired_states_*.npy"
    all_files = sorted(glob.glob(str(pattern)))
    
    if not all_files:
        raise ValueError(f"No files matching pattern 'saved_desired_states_*.npy' found in {input_dir}")
    
    # Extract motion_id from each filename
    file_motion_map = []
    for file_path in all_files:
        # Extract number from filename: saved_desired_states_{number}.npy
        match = re.search(r'saved_desired_states_(\d+)\.npy', Path(file_path).name)
        if match:
            file_motion_id = int(match.group(1))
            if motion_ids is None or file_motion_id in motion_ids:
                file_motion_map.append((file_path, file_motion_id))
    
    if motion_ids is not None and not file_motion_map:
        raise ValueError(f"No files found for motion_ids {motion_ids} in {input_dir}")
    
    return file_motion_map


def plot_codebook_changes(codebook_per_frame: np.ndarray, codebook_changed: np.ndarray,
                          motion_id: int, original_key: str, output_path: Path,
                          show_plot: bool = False):
    """
    Plot codebook changes over time from codebook sequence data.
    
    Args:
        codebook_per_frame: Codebook index for each frame [T]
        codebook_changed: Boolean array indicating codebook changes [T]
        motion_id: Motion ID for labeling
        original_key: Original motion key for labeling
        output_path: Path to save the plot
        show_plot: Whether to display the plot (default: False)
    """
    num_frames = len(codebook_per_frame)
    time_axis = np.arange(num_frames) / FPS  # Time in seconds
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.set_title(f'Codebook Changes Over Time - Motion ID: {motion_id} ({original_key})', 
                 fontsize=16, fontweight='bold')
    
    # Plot codebook indices as a line
    ax.plot(time_axis, codebook_per_frame, 
            linewidth=2, color='blue', alpha=0.7, label='Codebook Index', zorder=1)
    
    # Mark codebook changes with red dots
    change_indices = np.where(codebook_changed)[0]
    if len(change_indices) > 0:
        change_times = time_axis[change_indices]
        change_values = codebook_per_frame[change_indices]
        ax.scatter(change_times, change_values, 
                  color='red', s=80, marker='o', zorder=5, 
                  label=f'Codebook Changes ({len(change_indices)})', alpha=0.9, 
                  edgecolors='darkred', linewidths=1.5)
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Codebook Index', fontsize=14)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    unique_codebooks = len(np.unique(codebook_per_frame))
    min_codebook = np.min(codebook_per_frame)
    max_codebook = np.max(codebook_per_frame)
    num_changes = np.sum(codebook_changed)
    
    # Calculate average duration per codebook
    if num_changes > 0:
        change_intervals = np.diff(np.concatenate(([0], change_indices, [num_frames])))
        avg_duration = np.mean(change_intervals) / FPS  # in seconds
    else:
        avg_duration = num_frames / FPS  # entire sequence uses one codebook
    
    stats_text = (f'Unique Codebooks: {unique_codebooks}\n'
                 f'Range: {min_codebook} - {max_codebook}\n'
                 f'Total Changes: {num_changes}\n'
                 f'Avg Duration: {avg_duration:.2f}s')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved codebook changes plot: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


class AMASSFormatGenerator:
    """Generate motion files in AMASS format from a trained VQVAE model.
    
    Supports both NPY and PKL input/output formats based on the format flag.
    """

    def __init__(self, config_path: str, checkpoint_path: str, input_file: str, 
                 format_type: str = "pkl", eval_stride: int = None, file_motion_id: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
            input_file: Input file path (PKL file for PKL format, or NPY file path for NPY format)
            format_type: "pkl" or "npy" to specify input/output format
            eval_stride: Optional stride for overlapped reconstruction
            file_motion_id: Optional motion ID for NPY format (extracted from filename if not provided)
        """
        self.input_file = input_file
        self.format_type = format_type.lower()
        
        if self.format_type not in ["pkl", "npy"]:
            raise ValueError(f"format_type must be 'pkl' or 'npy', got '{format_type}'")
        
        # Extract motion_id from filename for NPY format if not provided
        if self.format_type == "npy" and file_motion_id is None:
            match = re.search(r'saved_desired_states_(\d+)\.npy', Path(input_file).name)
            if match:
                self.file_motion_id = int(match.group(1))
            else:
                self.file_motion_id = 0  # Default
        else:
            self.file_motion_id = file_motion_id

        # Shared init
        self.config, self.agent, self.motion_adapter = load_config_and_agent(config_path, checkpoint_path)

        # Allow overriding evaluation stride for overlapped reconstruction
        if eval_stride is not None:
            self.config["eval_stride"] = int(eval_stride)
            self.agent.config["eval_stride"] = int(eval_stride)

        # Load original motion data
        if self.format_type == "npy":
            print(f"Loading original motion data from npy file: {input_file}")
            self.original_motions, self.original_keys = load_original_npy(input_file)
            print(f"Loaded {len(self.original_keys)} original motions from npy file")
            
            # Load motion data for normalization statistics
            # For npy files, typically only one motion, so we use motion_id 0
            subset_motion_ids = [0] if len(self.original_keys) > 0 else []
            print(f"Loading motion data for normalization stats...")
            
            # Load the npy data and convert it to AMASS-like format for the adapter
            mocap_data, end_indices, frame_size = self._load_npy_motion_data(input_file, subset_motion_ids)
        else:  # PKL format
            print(f"Loading original AMASS PKL data from: {input_file}")
            self.original_motions, self.original_keys = load_original_pkl(input_file)
            print(f"Loaded {len(self.original_keys)} original AMASS motions")
            
            # Load multiple motions for proper normalization statistics (same as eval_vqvae.py)
            max_motions_for_stats = min(500, len(self.original_keys))
            subset_motion_ids = list(range(max_motions_for_stats))
            print(f"Loading first {max_motions_for_stats} motions for normalization stats...")
            
            mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
                input_file, subset_motion_ids
            )
        
        print(f"Loaded data for stats: shape={mocap_data.shape}, frame_size={frame_size}")
        print(f"Using device: {self.agent.device}")

        # Calculate normalization stats and initialize model (same as eval_vqvae.py)
        self.agent.frame_size = int(frame_size)
        self.frame_size = int(frame_size)  # Store for later use
        ensure_stats(self.agent, mocap_data)
        initialize_model(self.agent, self.config, self.frame_size, checkpoint_path)

    def generate_amass_format(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Generate motion files in AMASS format (NPY or PKL based on format_type)."""
        if self.format_type == "npy":
            return self.generate_amass_format_npy(motion_ids, output_dir)
        else:
            return self.generate_amass_format_pkl(motion_ids, output_dir)

    def generate_amass_format_npy(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Generate one NPY per motion id (global/world export) in the same format as input NPY files."""
        print(f"\n=== Generating AMASS Format NPY Files ===")
        out_dir = Path(output_dir or "./outputs/vqvae_amass_motions_npy")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_files: List[str] = []

        for motion_id in motion_ids:
            if motion_id < 0 or motion_id >= len(self.original_keys):
                print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
                continue

            print(f"Processing motion ID {motion_id} (file motion_id: {self.file_motion_id})...")
            amass_motion = self._generate_single_motion(motion_id)

            if amass_motion is None:
                print(f"   Failed to generate motion {motion_id}")
                continue

            # Convert AMASS format to NPY format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
            # Root rotation needs to be converted from XYZW to WXYZ
            num_frames = amass_motion["root_trans_offset"].shape[0]
            fps = amass_motion.get("fps", 30.0)
            
            # Generate time stamps
            time_stamps = np.arange(num_frames) / fps
            
            # Extract data
            root_pos = amass_motion["root_trans_offset"]  # [T, 3]
            root_rot_xyzw = amass_motion["root_rot"]  # [T, 4] in XYZW format
            dof_pos = amass_motion["dof"]  # [T, 23]
            
            # Convert root rotation from XYZW to WXYZ
            root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]  # [w, x, y, z] = [3, 0, 1, 2]
            
            # Create output array: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
            output_data = np.zeros((num_frames, 32), dtype=np.float32)
            output_data[:, 0] = time_stamps
            output_data[:, 1:4] = root_pos
            output_data[:, 4:8] = root_rot_wxyz
            output_data[:, 8:31] = dof_pos
            output_data[:, 31] = self.file_motion_id  # Use file_motion_id
            
            # Use file_motion_id for filename to match the input file
            out_path = out_dir / f"vqvae_motion_{self.file_motion_id:03d}.npy"
            np.save(out_path, output_data)
            generated_files.append(str(out_path))
            original_key = self.original_keys[motion_id]
            print(f"   Saved: {original_key} -> {out_path} (shape: {output_data.shape})")

        print(f"\nSaved {len(generated_files)} files to: {out_dir}")
        return generated_files, str(out_dir)

    def generate_amass_format_pkl(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Generate one AMASS format PKL per motion id (global/world export).
        
        Works only with original AMASS PKL input data.
        """
        return generate_motion_pkl_files(
            motion_ids=motion_ids,
            original_keys=self.original_keys,
            motion_generator_func=self._generate_single_motion,
            output_dir=output_dir or "./outputs/vqvae_amass_motions",
            filename_prefix="vqvae_motion_",
            generation_type="AMASS Format"
        )

    def extract_codebook_sequences(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Extract codebook sequences and full motion data for given motion IDs.
        
        For NPY format: saves NPY and PKL files
        For PKL format: saves CSV and PKL files
        """
        if self.format_type == "npy":
            return self._extract_codebook_sequences_npy(motion_ids, output_dir)
        else:
            return self._extract_codebook_sequences_pkl(motion_ids, output_dir)

    def _extract_codebook_sequences_npy(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Extract codebook sequences for NPY format - saves NPY and PKL files."""
        print(f"\n=== Extracting Codebook Sequences ===")
        out_dir = Path(output_dir or "./outputs/codebook_sequences")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_files: List[str] = []

        for motion_id in motion_ids:
            if motion_id < 0 or motion_id >= len(self.original_keys):
                print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
                continue

            print(f"Extracting codebook sequence for motion ID {motion_id} (file motion_id: {self.file_motion_id})...")
            codebook_data = self._extract_single_codebook_sequence(motion_id)

            if codebook_data is None:
                print(f"   Failed to extract codebook sequence for motion {motion_id}")
                continue

            original_key = self.original_keys[motion_id]
            
            # Get motion data
            reconstructed_motion = codebook_data['reconstructed_motion']
            original_motion = codebook_data['original_motion']
            codebook_sequence = codebook_data['codebook_sequence']
            
            # The codebook sequence contains codebook indices after temporal downsampling
            # Due to down_t and stride_t, each window produces multiple codebook indices
            # Formula: codebook_indices_per_window = window_size / (2^down_t)
            # We need to map codebook indices to each original motion frame (30Hz)
            motion_length = min(reconstructed_motion.shape[0], original_motion.shape[0])
            codebook_length = len(codebook_sequence)
            
            window_size = self.config.get('window_size', 64)  # Default window size
            down_t = self.config.get('down_t', 4)
            num_windows = (motion_length + window_size - 1) // window_size  # Ceiling division
            codebook_indices_per_window = window_size // (2 ** down_t)
            
            print(f"    Motion data shape: {reconstructed_motion.shape}, Original shape: {original_motion.shape}")
            print(f"    Original motion length: {motion_length} (frames at 30Hz)")
            print(f"    Window size: {window_size}, Number of windows: {num_windows}")
            print(f"    Downsampling: down_t={down_t} -> {codebook_indices_per_window} codebook indices per window")
            print(f"    Codebook sequence length: {codebook_length} (total codebook indices)")
            
            # Map codebook indices to each motion frame
            # Each codebook index covers multiple frames due to temporal downsampling
            codebook_per_frame = np.zeros(motion_length, dtype=np.int32)
            
            # Calculate how many frames each codebook index should cover
            frames_per_codebook = motion_length // codebook_length
            remainder_frames = motion_length % codebook_length
            
            frame_idx = 0
            for i, codebook_idx in enumerate(codebook_sequence):
                # Distribute remainder frames across first few codebook indices
                current_frames = frames_per_codebook + (1 if i < remainder_frames else 0)
                codebook_per_frame[frame_idx:frame_idx + current_frames] = codebook_idx
                frame_idx += current_frames
            
            # Use original motion length for the sequence
            sequence_length = motion_length
            
            # Slice arrays to motion_length to ensure all frames are included
            reconstructed_motion_sliced = reconstructed_motion[:motion_length]
            original_motion_sliced = original_motion[:motion_length]
            
            # Save codebook sequence as NPY file
            codebook_seq_path = out_dir / f"codebook_sequence_{self.file_motion_id:03d}.npy"
            np.save(codebook_seq_path, codebook_sequence.astype(np.int32))
            generated_files.append(str(codebook_seq_path))
            print(f"    Saved NPY: {original_key} -> {codebook_seq_path.name} (length: {codebook_length})")
            
            # Calculate reconstruction errors
            reconstruction_errors = np.abs(reconstructed_motion_sliced - original_motion_sliced)
            codebook_changes = np.diff(codebook_per_frame[:sequence_length], prepend=codebook_per_frame[0])
            
            # Save PKL file with structured data
            pkl_path = out_dir / f"codebook_sequence_{self.file_motion_id:03d}.pkl"
            pkl_data = {
                # Raw codebook data (window-level)
                'codebook_sequence': codebook_sequence.astype(np.int32),  # Window-level codebook indices
                
                # Frame-level mapped data (30Hz)
                'codebook_per_frame': codebook_per_frame[:sequence_length].astype(np.int32),  # Codebook index for each frame
                'reconstructed_motion': reconstructed_motion_sliced.astype(np.float32),  # Reconstructed features [T, F]
                'original_motion': original_motion_sliced.astype(np.float32),  # Original features [T, F]
                'reconstruction_errors': reconstruction_errors.astype(np.float32),  # Per-feature errors [T, F]
                'total_reconstruction_error': np.sum(reconstruction_errors, axis=1).astype(np.float32),  # Per-frame total error [T]
                'codebook_changed': (codebook_changes != 0).astype(bool),  # Boolean array indicating codebook changes [T]
                
                # Metadata
                'motion_id': motion_id,
                'file_motion_id': self.file_motion_id,  # File motion ID (from filename)
                'original_key': original_key,
                'sequence_length': sequence_length,
                'codebook_length': codebook_length,  # Number of windows
                'num_features': reconstructed_motion_sliced.shape[1],
                'window_size': window_size,
            }
            
            joblib.dump(pkl_data, pkl_path)
            generated_files.append(str(pkl_path))
            print(f"    Saved PKL: {original_key} -> {pkl_path}")
            print(f"    Motion frames (30Hz): {sequence_length}")
            print(f"    Motion features: {reconstructed_motion_sliced.shape[1]}")
            print(f"    Codebook windows: {codebook_length}")
            print(f"    Codebook changes: {np.sum(codebook_changes != 0)}")
            
            # Save codebook changes plot
            plot_path = out_dir / f"codebook_sequence_{self.file_motion_id:03d}_codebook_changes.png"
            plot_codebook_changes(
                codebook_per_frame[:sequence_length],
                codebook_changes != 0,
                motion_id,
                original_key,
                plot_path,
                show_plot=False
            )
            generated_files.append(str(plot_path))

        print(f"\nSaved {len(generated_files)} codebook sequence files (NPY + PKL + PNG) to: {out_dir}")
        return generated_files, str(out_dir)

    def _extract_codebook_sequences_pkl(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Extract codebook sequences for PKL format - saves CSV and PKL files."""
        print(f"\n=== Extracting Codebook Sequences ===")
        out_dir = Path(output_dir or "./outputs/codebook_sequences")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_files: List[str] = []

        for motion_id in motion_ids:
            if motion_id < 0 or motion_id >= len(self.original_keys):
                print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
                continue

            print(f"Extracting codebook sequence for motion ID {motion_id}...")
            codebook_data = self._extract_single_codebook_sequence(motion_id)

            if codebook_data is None:
                print(f"   Failed to extract codebook sequence for motion {motion_id}")
                continue

            # Save to CSV with all motion data at each step
            original_key = self.original_keys[motion_id]
            out_path = out_dir / f"codebook_sequence_{motion_id:03d}.csv"
            
            # Get motion data
            reconstructed_motion = codebook_data['reconstructed_motion']
            original_motion = codebook_data['original_motion']
            codebook_sequence = codebook_data['codebook_sequence']
            
            # The codebook sequence contains codebook indices after temporal downsampling
            # Due to down_t and stride_t, each window produces multiple codebook indices
            # Formula: codebook_indices_per_window = window_size / (2^down_t)
            # We need to map codebook indices to each original motion frame (30Hz)
            motion_length = min(reconstructed_motion.shape[0], original_motion.shape[0])
            codebook_length = len(codebook_sequence)
            
            window_size = self.config.get('window_size', 64)  # Default window size
            down_t = self.config.get('down_t', 4)
            num_windows = (motion_length + window_size - 1) // window_size  # Ceiling division
            codebook_indices_per_window = window_size // (2 ** down_t)
            
            print(f"    Motion data shape: {reconstructed_motion.shape}, Original shape: {original_motion.shape}")
            print(f"    Original motion length: {motion_length} (frames at 30Hz)")
            print(f"    Window size: {window_size}, Number of windows: {num_windows}")
            print(f"    Downsampling: down_t={down_t} -> {codebook_indices_per_window} codebook indices per window")
            print(f"    Codebook sequence length: {codebook_length} (total codebook indices)")
            
            # Map codebook indices to each motion frame
            # Each codebook index covers multiple frames due to temporal downsampling
            codebook_per_frame = np.zeros(motion_length, dtype=np.int32)
            
            # Calculate how many frames each codebook index should cover
            frames_per_codebook = motion_length // codebook_length
            remainder_frames = motion_length % codebook_length
            
            frame_idx = 0
            for i, codebook_idx in enumerate(codebook_sequence):
                # Distribute remainder frames across first few codebook indices
                current_frames = frames_per_codebook + (1 if i < remainder_frames else 0)
                codebook_per_frame[frame_idx:frame_idx + current_frames] = codebook_idx
                frame_idx += current_frames
            
            # Use original motion length for the sequence
            sequence_length = motion_length
            
            # Slice arrays to motion_length to ensure all frames are included
            reconstructed_motion_sliced = reconstructed_motion[:motion_length]
            original_motion_sliced = original_motion[:motion_length]
            
            # Calculate time axis (in seconds at 30 FPS)
            time_axis = np.arange(sequence_length) / FPS
            
            # Prepare data for DataFrame
            data_dict = {
                'frame_idx': range(sequence_length),
                'time': time_axis,  # Time in seconds
                'codebook_idx': codebook_per_frame[:sequence_length],  # Use mapped codebook indices per frame
                'motion_id': [motion_id] * sequence_length,
                'original_key': [original_key] * sequence_length,
                'sequence_length': [sequence_length] * sequence_length
            }
            
            # Add each feature dimension as a separate column (sliced to motion_length)
            for feat_idx in range(reconstructed_motion_sliced.shape[1]):
                data_dict[f'reconstructed_feat_{feat_idx}'] = reconstructed_motion_sliced[:, feat_idx]
                data_dict[f'original_feat_{feat_idx}'] = original_motion_sliced[:, feat_idx]
            
            # Calculate reconstruction error for each frame
            reconstruction_errors = np.abs(reconstructed_motion_sliced - original_motion_sliced)
            for feat_idx in range(reconstruction_errors.shape[1]):
                data_dict[f'reconstruction_error_feat_{feat_idx}'] = reconstruction_errors[:, feat_idx]
            
            # Calculate total reconstruction error per frame
            data_dict['total_reconstruction_error'] = np.sum(reconstruction_errors, axis=1)
            
            # Calculate codebook change indicators (when codebook index changes)
            codebook_changes = np.diff(codebook_per_frame[:sequence_length], prepend=codebook_per_frame[0])
            data_dict['codebook_changed'] = codebook_changes != 0
            
            df = pd.DataFrame(data_dict)
            df.to_csv(out_path, index=False)
            generated_files.append(str(out_path))
            print(f"    Saved CSV: {original_key} -> {out_path}")
            
            # Save PKL file with structured data
            pkl_path = out_dir / f"codebook_sequence_{motion_id:03d}.pkl"
            pkl_data = {
                # Raw codebook data (window-level)
                'codebook_sequence': codebook_sequence.astype(np.int32),  # Window-level codebook indices
                
                # Frame-level mapped data (30Hz)
                'codebook_per_frame': codebook_per_frame[:sequence_length].astype(np.int32),  # Codebook index for each frame
                'reconstructed_motion': reconstructed_motion_sliced.astype(np.float32),  # Reconstructed features [T, F]
                'original_motion': original_motion_sliced.astype(np.float32),  # Original features [T, F]
                'reconstruction_errors': reconstruction_errors.astype(np.float32),  # Per-feature errors [T, F]
                'total_reconstruction_error': np.sum(reconstruction_errors, axis=1).astype(np.float32),  # Per-frame total error [T]
                'codebook_changed': (codebook_changes != 0).astype(bool),  # Boolean array indicating codebook changes [T]
                
                # Metadata
                'motion_id': motion_id,
                'original_key': original_key,
                'sequence_length': sequence_length,
                'codebook_length': codebook_length,  # Number of windows
                'num_features': reconstructed_motion_sliced.shape[1],
                'window_size': window_size,
            }
            
            joblib.dump(pkl_data, pkl_path)
            generated_files.append(str(pkl_path))
            print(f"    Saved PKL: {original_key} -> {pkl_path}")
            print(f"    Motion frames (30Hz): {sequence_length}")
            print(f"    Motion features: {reconstructed_motion_sliced.shape[1]}")
            print(f"    Codebook windows: {codebook_length}")
            print(f"    Codebook changes: {np.sum(codebook_changes != 0)}")
            
            # Save codebook changes plot
            plot_path = out_dir / f"codebook_sequence_{motion_id:03d}_codebook_changes.png"
            plot_codebook_changes(
                codebook_per_frame[:sequence_length],
                codebook_changes != 0,
                motion_id,
                original_key,
                plot_path,
                show_plot=False
            )
            generated_files.append(str(plot_path))

        print(f"\nSaved {len(generated_files)} codebook sequence files (CSV + PKL + PNG) to: {out_dir}")
        return generated_files, str(out_dir)

    # ---- internal helpers ----

    def _load_npy_motion_data(self, npy_file: str, motion_ids: List[int]) -> Tuple[torch.Tensor, np.ndarray, int]:
        """
        Load npy file and convert to MVQ format for the motion adapter.
        This converts npy data to the format expected by the VQVAE model.
        """
        # Load npy file
        trajectory_data = np.load(npy_file)
        
        # Format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
        num_frames = trajectory_data.shape[0]
        
        # Extract data
        time_stamps = trajectory_data[:, 0]
        root_pos = trajectory_data[:, 1:4]  # [T, 3]
        root_rot_wxyz = trajectory_data[:, 4:8]  # [T, 4] in WXYZ format
        dof_pos = trajectory_data[:, 8:31]  # [T, 23]
        
        # Convert root rotation from WXYZ to XYZW
        root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]  # [x, y, z, w] = [1, 2, 3, 0]
        
        # Calculate fps from time stamps
        if len(time_stamps) > 1:
            dt = np.mean(np.diff(time_stamps))
            fps = 1.0 / dt if dt > 0 else 30.0
        else:
            fps = 30.0
        
        # Convert to AMASS-like format for the adapter
        motion_data = {
            "root_trans_offset": root_pos.astype(np.float32),
            "root_rot": root_rot_xyzw.astype(np.float32),  # XYZW format
            "dof": dof_pos.astype(np.float32),
            "fps": fps,
        }
        
        # Use the adapter's feature extraction method
        motion_features = self.motion_adapter._extract_g1_features(motion_data)
        
        # Return as if it's a single motion
        end_indices = np.array([num_frames - 1], dtype=np.int64)
        frame_size = motion_features.shape[1]
        
        return motion_features, end_indices, frame_size

    def _generate_single_motion(self, motion_id: int) -> Optional[Dict]:
        """Reconstruct a single motion and convert to AMASS-like global output."""
        try:
            # Load MVQ data for this motion only
            if self.format_type == "npy":
                mocap_data, end_indices, frame_size = self._load_npy_motion_data(
                    self.input_file, [motion_id]
                )
            else:  # PKL format
                mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
                    self.input_file, [motion_id]
                )
            
            # Move to model device
            if isinstance(mocap_data, torch.Tensor):
                mocap_data = mocap_data.to(self.agent.device, non_blocking=True)
            if isinstance(end_indices, torch.Tensor):
                end_indices = end_indices.to(self.agent.device)

            self.agent.mocap_data = mocap_data
            self.agent.end_indices = end_indices
            self.agent.frame_size = int(frame_size)

            # Reconstruct with the VQVAE
            with torch.no_grad():
                idx = torch.tensor(0, device=self.agent.device)
                reconstructed_motion, original_seq, _codebook = self.agent.evaluate_policy_rec(idx)

            # Convert to AMASS-like global representation
            mvq = reconstructed_motion.detach().cpu().numpy().astype(np.float32)
            original_amass = self.original_motions[self.original_keys[motion_id]]
            return convert_to_amass_global_from_local(
                local_features=mvq,
                original_motion=original_amass,
                motion_id=motion_id,
            )
        except Exception as e:
            print(f"Error generating motion {motion_id}: {e}")
            return None

    def _extract_single_codebook_sequence(self, motion_id: int) -> Optional[Dict]:
        """Extract codebook sequence and motion data for a single motion."""
        try:
            # Load MVQ data for this motion only
            if self.format_type == "npy":
                mocap_data, end_indices, frame_size = self._load_npy_motion_data(
                    self.input_file, [motion_id]
                )
            else:  # PKL format
                mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
                    self.input_file, [motion_id]
                )
            
            # Move to model device
            if isinstance(mocap_data, torch.Tensor):
                mocap_data = mocap_data.to(self.agent.device, non_blocking=True)
            if isinstance(end_indices, torch.Tensor):
                end_indices = end_indices.to(self.agent.device)

            self.agent.mocap_data = mocap_data
            self.agent.end_indices = end_indices
            self.agent.frame_size = int(frame_size)

            # Extract codebook sequence with the VQVAE
            with torch.no_grad():
                idx = torch.tensor(0, device=self.agent.device)
                reconstructed_motion, original_seq, codebook_seq = self.agent.evaluate_policy_rec(idx)

            # Convert to numpy and return
            codebook_sequence = codebook_seq.detach().cpu().numpy().astype(np.int32)
            reconstructed_motion_np = reconstructed_motion.detach().cpu().numpy().astype(np.float32)
            original_seq_np = original_seq.detach().cpu().numpy().astype(np.float32)
            
            return {
                'codebook_sequence': codebook_sequence,
                'reconstructed_motion': reconstructed_motion_np,
                'original_motion': original_seq_np,
                'sequence_length': len(codebook_sequence),
                'motion_id': motion_id,
                'original_key': self.original_keys[motion_id]
            }
        except Exception as e:
            print(f"Error extracting codebook sequence for motion {motion_id}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion files in AMASS format from trained VQVAE. "
                    "Supports both NPY and PKL input/output formats."
    )
    parser.add_argument("--config", type=str, default="configs/agent.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt/.pt)")
    parser.add_argument("--format", type=str, choices=["pkl", "npy"], default="pkl",
                       help="Input/output format: 'pkl' for AMASS PKL files, 'npy' for NPY files")
    parser.add_argument("--input_pkl", type=str, default=None,
                       help="Path to input AMASS PKL file (required if --format=pkl)")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Directory containing NPY files (required if --format=npy). "
                            "Files should match pattern: saved_desired_states_*.npy")
    parser.add_argument("--motion_ids", type=str, default=None,
                       help='Comma/range list, e.g. "0,2,5-12" or "all" for NPY format. '
                            'For PKL format, default is "0". For NPY format, processes all files if not specified.')
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--eval_stride", type=int, default=None,
                       help="Stride for overlapped reconstruction (default: window_size//2)")
    
    # Codebook extraction options
    parser.add_argument("--extract_codebook", action="store_true",
                       help="Extract codebook sequences (saves CSV+PKL for PKL format, NPY+PKL for NPY format)")
    parser.add_argument("--codebook_output_dir", type=str, default=None,
                       help="Directory to save codebook files")

    args = parser.parse_args()
    
    # Validate format-specific arguments
    if args.format == "pkl":
        if args.input_pkl is None:
            raise ValueError("--input_pkl is required when --format=pkl")
        if args.motion_ids is None:
            args.motion_ids = "0"
        motion_ids = parse_motion_ids(args.motion_ids)
        
        # Process single PKL file
        generator = AMASSFormatGenerator(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            input_file=args.input_pkl,
            format_type="pkl",
            eval_stride=args.eval_stride,
        )
        
        if args.extract_codebook:
            codebook_files, codebook_dir = generator.extract_codebook_sequences(
                motion_ids, args.codebook_output_dir
            )
            print("\n=== Codebook Extraction Complete ===")
            print(f"Codebook output directory: {codebook_dir}")
            print(f"Total codebook files: {len(codebook_files)}")
            for p in codebook_files:
                print(f"  - {p}")
        else:
            generated_files, out_dir = generator.generate_amass_format(motion_ids, args.output_dir)
            print("\n=== Generation Complete ===")
            print(f"Output directory: {out_dir}")
            print("Input format: Original AMASS PKL")
            print("Output format: AMASS PKL")
            print("Space: global")
            print(f"Total files: {len(generated_files)}")
            for p in generated_files:
                print(f"  - {p}")
    
    else:  # NPY format
        if args.input_dir is None:
            raise ValueError("--input_dir is required when --format=npy")
        
        # Parse motion_ids for NPY format
        if args.motion_ids and args.motion_ids.lower() != "all":
            motion_ids = parse_motion_ids(args.motion_ids)
        else:
            motion_ids = None  # Process all files
        
        # Find matching npy files
        npy_files = find_npy_files(args.input_dir, motion_ids)
        
        if not npy_files:
            print("No matching npy files found.")
            return
        
        print(f"Found {len(npy_files)} npy file(s) to process:")
        for file_path, motion_id in npy_files:
            print(f"  - {Path(file_path).name} (motion_id: {motion_id})")
        
        # Process each file
        all_codebook_files = []
        all_generated_files = []
        
        for input_npy_file, file_motion_id in npy_files:
            print(f"\n{'='*60}")
            print(f"Processing: {Path(input_npy_file).name} (motion_id: {file_motion_id})")
            print(f"{'='*60}")
            
            generator = AMASSFormatGenerator(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                input_file=input_npy_file,
                format_type="npy",
                eval_stride=args.eval_stride,
                file_motion_id=file_motion_id,
            )
            
            # Use motion_id from file (which should match file_motion_id)
            # For single motion files, we always use motion_id 0 for list index
            process_motion_ids = [0]  # Always use index 0 for single-motion npy files
            
            if args.extract_codebook:
                # Extract codebook sequences
                codebook_files, codebook_dir = generator.extract_codebook_sequences(
                    process_motion_ids, args.codebook_output_dir
                )
                all_codebook_files.extend(codebook_files)
            else:
                # Generate AMASS format NPY files (original functionality)
                generated_files, out_dir = generator.generate_amass_format(process_motion_ids, args.output_dir)
                all_generated_files.extend(generated_files)
        
        # Print summary
        if args.extract_codebook:
            print("\n=== Codebook Extraction Complete ===")
            codebook_dir = args.codebook_output_dir or "./outputs/codebook_sequences"
            print(f"Codebook output directory: {codebook_dir}")
            print(f"Total codebook files: {len(all_codebook_files)}")
            for p in all_codebook_files:
                print(f"  - {p}")
        else:
            print("\n=== Generation Complete ===")
            out_dir = args.output_dir or "./outputs/vqvae_amass_motions_npy"
            print(f"Output directory: {out_dir}")
            print("Input format: NPY files")
            print("Output format: NPY files")
            print("Space: global")
            print(f"Total files: {len(all_generated_files)}")
            for p in all_generated_files:
                print(f"  - {p}")


if __name__ == "__main__":
    main()


'''
# Example usage with PKL format:
python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/outputs/run_0_500_switching/best_model.ckpt \
  --format pkl \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0-500" \
  --output_dir ./outputs/vqvae_amass_motions



# Example usage for extracting codebook sequences with PKL format:
python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/outputs/run_0_500_switching/best_model.ckpt \
  --format pkl \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0-500" \
  --extract_codebook \
  --codebook_output_dir ./outputs/codebook_sequences


'''