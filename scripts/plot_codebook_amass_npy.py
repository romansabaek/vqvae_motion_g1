#!/usr/bin/env python3
"""
Plot original vs reconstructed motion from codebook sequence PKL/NPY files.

Loads PKL/NPY files created by generate_motion_from_vqvae_s2.py and creates plots
comparing original and reconstructed motion features.

Supports multiple formats:
1. Codebook sequence PKL files (with 'original_motion', 'reconstructed_motion' keys)
2. Codebook sequence NPY files (1D array of codebook indices) - requires corresponding PKL file
3. AMASS format PKL files (with motion keys like '0-ACCAD_...') - requires --original_pkl
4. Motion NPY files ([time, root_pos(3), root_rot(4), dof_pos(23), motion_id]) - requires --original_npy
   Note: NPY files are typically 500 Hz versions of single motions from PKL files (which are 30 Hz).
         FPS is automatically detected from time stamps, so different frame rates are handled correctly.
"""

import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import sys

# Add motion_vqvae to path
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from motion_vqvae.config_loader import ConfigLoader
from scripts.vqvae_gen_init import (
    load_config_and_agent,
    load_original_pkl,
    infer_frame_size,
    initialize_model,
    ensure_stats,
)
from scripts.generate_motion_from_vqvae_s2 import load_original_npy

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


def is_codebook_sequence_format(data: dict) -> bool:
    """Check if data is in codebook sequence format (has 'original_motion' key)."""
    return 'original_motion' in data and 'reconstructed_motion' in data


def is_amass_format(data: dict) -> bool:
    """Check if data is in AMASS format (has motion keys like '0-ACCAD_...')."""
    if not data:
        return False
    # AMASS format has motion keys as top-level keys, and values are dicts with motion data
    first_key = list(data.keys())[0]
    first_value = data[first_key]
    # AMASS format: keys are motion identifiers, values are dicts with 'root_trans_offset', 'root_rot', 'dof', etc.
    return isinstance(first_value, dict) and 'root_trans_offset' in first_value


def load_npy_data(file_path: Path) -> Optional[np.ndarray]:
    """Load NPY file and return numpy array."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return None
    
    if file_path.suffix != '.npy':
        print(f"Error: File must be a .npy file, got: {file_path.suffix}")
        return None
    
    try:
        data = np.load(file_path)
        print(f"\n{'='*60}")
        print(f"NPY File: {file_path.name}")
        print(f"{'='*60}")
        print(f"Loaded array shape: {data.shape}, dtype: {data.dtype}")
        print(f"{'='*60}\n")
        return data
    except Exception as e:
        print(f"Error loading NPY file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def is_codebook_sequence_npy(data: np.ndarray) -> bool:
    """Check if NPY data is a codebook sequence (1D array of integers)."""
    return data.ndim == 1 and (data.dtype == np.int32 or data.dtype == np.int64 or np.issubdtype(data.dtype, np.integer))


def is_motion_npy(data: np.ndarray) -> bool:
    """
    Check if NPY data is a motion file (2D array with 32 columns).
    
    Format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
    Can be at any frame rate (typically 500 Hz for NPY files vs 30 Hz for PKL files).
    """
    return data.ndim == 2 and data.shape[1] == 32


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


def extract_features_from_amass(amass_motion: dict, config_path: str = "configs/agent.yaml") -> np.ndarray:
    """Extract motion features from AMASS format motion dict using MotionDataAdapter."""
    # Load config and create adapter
    config = ConfigLoader().load_config(config_path)
    adapter = MotionDataAdapter(config)
    adapter.device = 'cpu'  # Use CPU for feature extraction
    
    # Extract features using the adapter's method
    features = adapter._extract_g1_features(amass_motion)
    return features.detach().cpu().numpy()


def extract_features_from_npy(npy_file_path: Path, config_path: str = "configs/agent.yaml") -> np.ndarray:
    """
    Extract motion features from NPY motion file using MotionDataAdapter.
    
    The NPY file is typically a high-frequency (e.g., 500 Hz) version of a single motion.
    The FPS is automatically detected from the time stamps in the file.
    Format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id] = 32 cols
    """
    # Load NPY file and convert to AMASS-like format
    # FPS is automatically calculated from time stamps (works with any frame rate)
    motions, keys = load_original_npy(str(npy_file_path))
    motion_key = keys[0]
    amass_motion = motions[motion_key]
    
    # Extract features using the adapter's method
    return extract_features_from_amass(amass_motion, config_path)


def extract_codebook_sequence_from_npy_motion(reconstructed_npy_path: Path, original_npy_path: Path,
                                               config_path: str, checkpoint_path: str) -> Optional[dict]:
    """
    Extract codebook sequence from NPY motion file using VQVAE model.
    
    Args:
        reconstructed_npy_path: Path to NPY motion file (reconstructed motion)
        original_npy_path: Path to original NPY motion file
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Dictionary with codebook sequence data in codebook sequence format
    """
    import tempfile
    import os
    
    print(f"Extracting codebook sequence from NPY motion file...")
    print(f"  Reconstructed motion: {reconstructed_npy_path}")
    print(f"  Original motion: {original_npy_path}")
    
    # Load config and initialize agent
    config, agent, motion_adapter = load_config_and_agent(config_path, checkpoint_path)
    
    # Convert NPY to AMASS-like format
    reconstructed_motions, reconstructed_keys = load_original_npy(str(reconstructed_npy_path))
    original_motions, original_keys = load_original_npy(str(original_npy_path))
    
    reconstructed_motion_amass = reconstructed_motions[reconstructed_keys[0]]
    original_motion_amass = original_motions[original_keys[0]]
    
    # Create temporary PKL file for the adapter
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
        temp_pkl_path = tmp_file.name
        single_motion_dict = {original_keys[0]: original_motion_amass}
        joblib.dump(single_motion_dict, temp_pkl_path)
    
    try:
        # Load motion data for stats (use the original motion)
        mocap_data_stats, _, frame_size = motion_adapter.load_motion_data(
            temp_pkl_path, [0]
        )
        
        # Ensure stats
        agent.frame_size = int(frame_size)
        if isinstance(mocap_data_stats, torch.Tensor):
            mocap_data_stats = mocap_data_stats.to(agent.device)
        ensure_stats(agent, mocap_data_stats)
        initialize_model(agent, config, frame_size, checkpoint_path)
        
        # Load the motion for codebook extraction
        mocap_data, end_indices, frame_size = motion_adapter.load_motion_data(
            temp_pkl_path, [0]
        )
        
        # Move to device
        if isinstance(mocap_data, torch.Tensor):
            mocap_data = mocap_data.to(agent.device, non_blocking=True)
        if isinstance(end_indices, torch.Tensor):
            end_indices = end_indices.to(agent.device)
        
        agent.mocap_data = mocap_data
        agent.end_indices = end_indices
        agent.frame_size = int(frame_size)
        
        # Extract codebook sequence with the VQVAE
        with torch.no_grad():
            idx = torch.tensor(0, device=agent.device)
            reconstructed_motion, original_seq, codebook_seq = agent.evaluate_policy_rec(idx)
        
        # Convert to numpy
        codebook_sequence = codebook_seq.detach().cpu().numpy().astype(np.int32)
        reconstructed_motion_np = reconstructed_motion.detach().cpu().numpy().astype(np.float32)
        original_seq_np = original_seq.detach().cpu().numpy().astype(np.float32)
        
        # Map codebook indices to each motion frame (window-level to frame-level)
        motion_length = min(reconstructed_motion_np.shape[0], original_seq_np.shape[0])
        codebook_length = len(codebook_sequence)
        
        window_size = config.get('window_size', 64)
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
        
        # Calculate codebook changes
        codebook_changes = np.diff(codebook_per_frame, prepend=codebook_per_frame[0])
        
        # Extract motion ID from filename
        import re
        match = re.search(r'(\d+)', reconstructed_npy_path.stem)
        file_motion_id = int(match.group(1)) if match else 0
        
        return {
            'codebook_sequence': codebook_sequence,  # Window-level
            'codebook_per_frame': codebook_per_frame,  # Frame-level
            'codebook_changed': (codebook_changes != 0).astype(bool),
            'reconstructed_motion': reconstructed_motion_np,
            'original_motion': original_seq_np,
            'motion_id': 0,
            'file_motion_id': file_motion_id,
            'original_key': original_keys[0],
            'sequence_length': motion_length,
            'codebook_length': codebook_length,
            'window_size': window_size,
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_pkl_path):
            os.remove(temp_pkl_path)


def extract_codebook_sequence_from_amass(amass_pkl_path: Path, original_pkl_path: Path,
                                         config_path: str, checkpoint_path: str) -> Optional[dict]:
    """
    Extract codebook sequence from AMASS format file using VQVAE model.
    
    Args:
        amass_pkl_path: Path to AMASS format PKL file (reconstructed motion)
        original_pkl_path: Path to original AMASS PKL file (for motion key matching)
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Dictionary with codebook sequence data in codebook sequence format
    """
    import tempfile
    import os
    
    print(f"Extracting codebook sequence from AMASS format file...")
    print(f"  Reconstructed motion: {amass_pkl_path}")
    print(f"  Original motion: {original_pkl_path}")
    
    # Load AMASS data
    amass_data = joblib.load(amass_pkl_path)
    original_data = joblib.load(original_pkl_path)
    
    if not is_amass_format(amass_data) or not is_amass_format(original_data):
        print(f"Error: Files must be in AMASS format")
        return None
    
    # Get motion key
    motion_keys = list(amass_data.keys())
    if len(motion_keys) != 1:
        print(f"Warning: Expected 1 motion, found {len(motion_keys)}. Using first motion.")
    motion_key = motion_keys[0]
    
    # Find matching motion in original file
    if motion_key not in original_data:
        original_keys = list(original_data.keys())
        if len(original_keys) == 0:
            print(f"Error: Original file has no motions")
            return None
        print(f"Warning: Motion key '{motion_key}' not found in original file. Using '{original_keys[0]}'")
        motion_key = original_keys[0]
    
    # Load config and initialize agent
    config, agent, motion_adapter = load_config_and_agent(config_path, checkpoint_path)
    
    # Create temporary PKL file with the motion for the adapter
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp_file:
        temp_pkl_path = tmp_file.name
        single_motion_dict = {motion_key: original_data[motion_key]}
        joblib.dump(single_motion_dict, temp_pkl_path)
    
    try:
        # Load motion data for stats (use subset of motions from original file)
        max_motions_for_stats = min(300, len(list(original_data.keys())))
        subset_motion_ids = list(range(max_motions_for_stats))
        mocap_data_stats, _, frame_size = motion_adapter.load_motion_data(
            str(original_pkl_path), subset_motion_ids
        )
        
        # Ensure stats
        agent.frame_size = int(frame_size)
        if isinstance(mocap_data_stats, torch.Tensor):
            mocap_data_stats = mocap_data_stats.to(agent.device)
        ensure_stats(agent, mocap_data_stats)
        initialize_model(agent, config, frame_size, checkpoint_path)
        
        # Load the single motion for codebook extraction
        mocap_data, end_indices, frame_size = motion_adapter.load_motion_data(
            temp_pkl_path, [0]
        )
        
        # Move to device
        if isinstance(mocap_data, torch.Tensor):
            mocap_data = mocap_data.to(agent.device, non_blocking=True)
        if isinstance(end_indices, torch.Tensor):
            end_indices = end_indices.to(agent.device)
        
        agent.mocap_data = mocap_data
        agent.end_indices = end_indices
        agent.frame_size = int(frame_size)
        
        # Extract codebook sequence with the VQVAE
        with torch.no_grad():
            idx = torch.tensor(0, device=agent.device)
            reconstructed_motion, original_seq, codebook_seq = agent.evaluate_policy_rec(idx)
        
        # Convert to numpy
        codebook_sequence = codebook_seq.detach().cpu().numpy().astype(np.int32)
        reconstructed_motion_np = reconstructed_motion.detach().cpu().numpy().astype(np.float32)
        original_seq_np = original_seq.detach().cpu().numpy().astype(np.float32)
        
        # Map codebook indices to each motion frame (window-level to frame-level)
        motion_length = min(reconstructed_motion_np.shape[0], original_seq_np.shape[0])
        codebook_length = len(codebook_sequence)
        
        window_size = config.get('window_size', 64)
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
        
        # Calculate codebook changes
        codebook_changes = np.diff(codebook_per_frame, prepend=codebook_per_frame[0])
        
        return {
            'codebook_sequence': codebook_sequence,  # Window-level
            'codebook_per_frame': codebook_per_frame,  # Frame-level
            'codebook_changed': (codebook_changes != 0).astype(bool),
            'reconstructed_motion': reconstructed_motion_np,
            'original_motion': original_seq_np,
            'motion_id': 0,
            'original_key': motion_key,
            'sequence_length': motion_length,
            'codebook_length': codebook_length,
            'window_size': window_size,
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_pkl_path):
            os.remove(temp_pkl_path)


def convert_amass_to_codebook_format(reconstructed_pkl_path: Path, original_pkl_path: Optional[Path] = None,
                                     config_path: str = "configs/agent.yaml") -> Optional[dict]:
    """Convert AMASS format PKL to codebook sequence format for plotting."""
    # Load reconstructed motion
    reconstructed_data = joblib.load(reconstructed_pkl_path)
    
    if not is_amass_format(reconstructed_data):
        print(f"Error: File does not appear to be in AMASS format")
        return None
    
    # Get the first (and typically only) motion key
    motion_keys = list(reconstructed_data.keys())
    if len(motion_keys) != 1:
        print(f"Warning: Expected 1 motion, found {len(motion_keys)}. Using first motion.")
    motion_key = motion_keys[0]
    reconstructed_motion_amass = reconstructed_data[motion_key]
    
    # Extract features from reconstructed motion
    print(f"Extracting features from reconstructed motion...")
    reconstructed_features = extract_features_from_amass(reconstructed_motion_amass, config_path)
    
    # Load original motion if provided
    if original_pkl_path is not None:
        if not original_pkl_path.exists():
            print(f"Error: Original file not found: {original_pkl_path}")
            return None
        
        # Check if original is PKL or NPY
        if original_pkl_path.suffix == '.npy':
            print(f"Loading original motion from NPY: {original_pkl_path}")
            original_features = extract_features_from_npy(original_pkl_path, config_path)
        else:
            print(f"Loading original motion from PKL: {original_pkl_path}")
            original_data = joblib.load(original_pkl_path)
            
            if not is_amass_format(original_data):
                print(f"Error: Original file does not appear to be in AMASS format")
                return None
            
            # Find matching motion key in original file
            if motion_key in original_data:
                original_motion_amass = original_data[motion_key]
            else:
                # Try to use first motion if key doesn't match
                original_keys = list(original_data.keys())
                if len(original_keys) == 0:
                    print(f"Error: Original file has no motions")
                    return None
                print(f"Warning: Motion key '{motion_key}' not found in original file. Using '{original_keys[0]}'")
                original_motion_amass = original_data[original_keys[0]]
            
            # Extract features from original motion
            print(f"Extracting features from original motion...")
            original_features = extract_features_from_amass(original_motion_amass, config_path)
    else:
        # If no original file provided, we can't compare - return error
        print(f"Error: AMASS format file requires --original_pkl/--original_npy for comparison")
        return None
    
    # Create codebook sequence format dict
    return {
        'original_motion': original_features,
        'reconstructed_motion': reconstructed_features,
        'motion_id': 0,  # Default motion ID
        'original_key': motion_key,
    }


def convert_npy_motion_to_codebook_format(reconstructed_npy_path: Path, original_npy_path: Optional[Path] = None,
                                           config_path: str = "configs/agent.yaml") -> Optional[dict]:
    """
    Convert NPY motion file to codebook sequence format for plotting.
    
    NPY files are typically 500 Hz versions of single motions from PKL files (which are 30 Hz).
    FPS is automatically detected from time stamps, so different frame rates are handled correctly.
    """
    # Extract features from reconstructed motion
    print(f"Extracting features from reconstructed NPY motion...")
    reconstructed_features = extract_features_from_npy(reconstructed_npy_path, config_path)
    
    # Load original motion if provided
    if original_npy_path is not None:
        if not original_npy_path.exists():
            print(f"Error: Original NPY file not found: {original_npy_path}")
            return None
        
        print(f"Extracting features from original NPY motion...")
        original_features = extract_features_from_npy(original_npy_path, config_path)
    else:
        print(f"Error: NPY motion format requires --original_npy for comparison")
        return None
    
    # Extract motion ID from filename if possible
    import re
    match = re.search(r'(\d+)', reconstructed_npy_path.stem)
    file_motion_id = int(match.group(1)) if match else 0
    
    # Create codebook sequence format dict
    return {
        'original_motion': original_features,
        'reconstructed_motion': reconstructed_features,
        'motion_id': 0,
        'file_motion_id': file_motion_id,
        'original_key': f"motion_{file_motion_id}",
    }


def load_codebook_sequence_from_npy(npy_file_path: Path) -> Optional[dict]:
    """Load codebook sequence NPY file and try to find corresponding PKL file."""
    codebook_sequence = load_npy_data(npy_file_path)
    if codebook_sequence is None:
        return None
    
    if not is_codebook_sequence_npy(codebook_sequence):
        print(f"Error: NPY file does not appear to be a codebook sequence (expected 1D integer array)")
        return None
    
    # Try to find corresponding PKL file in the same directory
    pkl_file_path = npy_file_path.parent / f"{npy_file_path.stem}.pkl"
    if pkl_file_path.exists():
        print(f"Found corresponding PKL file: {pkl_file_path.name}")
        pkl_data = load_pkl_data(pkl_file_path)
        if pkl_data is not None and is_codebook_sequence_format(pkl_data):
            return pkl_data
        else:
            print(f"Warning: PKL file does not contain codebook sequence format data")
    else:
        print(f"Warning: Could not find corresponding PKL file: {pkl_file_path}")
        print(f"Codebook sequence NPY files require corresponding PKL files with full motion data for plotting.")
        print(f"Expected PKL file at: {pkl_file_path}")
    
    return None


def save_codebook_sequence_csv(data: dict, output_path: Path):
    """
    Save codebook sequence to CSV file with time and codebook_id columns.
    
    Args:
        data: Dictionary containing codebook data from PKL (must have codebook_per_frame)
        output_path: Path to save the CSV file
    """
    if 'codebook_per_frame' not in data:
        print(f"Error: Missing codebook data. Required key: 'codebook_per_frame'")
        print(f"Available keys: {list(data.keys())}")
        return False
    
    codebook_per_frame = data['codebook_per_frame']
    num_frames = len(codebook_per_frame)
    
    # Generate time axis in seconds (30 FPS)
    time_axis = np.arange(num_frames) / FPS
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_axis,
        'codebook_id': codebook_per_frame
    })
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Codebook sequence saved to CSV: {output_path}")
    print(f"  Total frames: {num_frames}")
    print(f"  Duration: {time_axis[-1]:.2f} seconds")
    print(f"  Codebook range: {np.min(codebook_per_frame)} - {np.max(codebook_per_frame)}")
    print(f"  Unique codebooks: {len(np.unique(codebook_per_frame))}")
    
    return True


def plot_codebook_changes(data: dict, output_path: Optional[Path] = None, 
                          show_plot: bool = True):
    """
    Plot codebook changes over time from codebook sequence data.
    
    Args:
        data: Dictionary containing codebook data from PKL (must have codebook_per_frame and codebook_changed)
        output_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    # Check for required keys
    if 'codebook_per_frame' not in data or 'codebook_changed' not in data:
        print(f"Error: Missing codebook data. Required keys: 'codebook_per_frame', 'codebook_changed'")
        print(f"Available keys: {list(data.keys())}")
        return
    
    codebook_per_frame = data['codebook_per_frame']
    codebook_changed = data['codebook_changed']
    motion_id = data.get('motion_id', 'Unknown')
    original_key = data.get('original_key', 'Unknown')
    
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
                  label=f'Codebook Changes ({len(change_indices)})', alpha=0.9, edgecolors='darkred', linewidths=1.5)
    
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
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Codebook changes plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


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
    
    # Check if codebook data is available
    has_codebook_data = 'codebook_per_frame' in data and 'codebook_changed' in data
    
    # Create figure with subplots (2 rows if no codebook data, 3 rows if codebook data available)
    if has_codebook_data:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
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
    if not has_codebook_data:
        ax1.set_xlabel('Time (s)', fontsize=12)
    
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
    if not has_codebook_data:
        ax2.set_xlabel('Time (s)', fontsize=12)
    
    # === Third Row: Codebook Changes (only if available) ===
    if has_codebook_data:
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_title('Codebook Changes Over Time', fontsize=14, fontweight='bold')
        
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
        description="Plot original vs reconstructed motion from codebook sequence PKL/NPY files or AMASS format PKL/NPY files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to PKL/NPY file (codebook sequence, AMASS format PKL, or motion NPY)"
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
    parser.add_argument(
        "--original-pkl",
        type=str,
        default=None,
        help="Path to original PKL file (required for AMASS format PKL files)"
    )
    parser.add_argument(
        "--original-npy",
        type=str,
        default=None,
        help="Path to original NPY file (required for NPY motion files)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/agent.yaml",
        help="Path to config file (for AMASS/NPY format feature extraction)"
    )
    parser.add_argument(
        "--codebook-only",
        action="store_true",
        help="Only plot codebook changes over time (requires codebook sequence PKL/NPY file or AMASS/NPY format with --checkpoint)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.ckpt/.pt) - required for extracting codebook sequences from AMASS/NPY format files"
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Path to save codebook sequence as CSV file (with time and codebook_id columns)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine file type and load accordingly
    if input_path.suffix == '.npy':
        print(f"Loading NPY file: {input_path}")
        npy_data = load_npy_data(input_path)
        
        if npy_data is None:
            print("Failed to load NPY data.")
            return
        
        # Check if it's a codebook sequence NPY (1D array) or motion NPY (2D array)
        if is_codebook_sequence_npy(npy_data):
            print("Detected codebook sequence NPY format")
            # Try to load corresponding PKL file
            plot_data = load_codebook_sequence_from_npy(input_path)
            if plot_data is None:
                print("Error: Codebook sequence NPY file requires corresponding PKL file with full motion data.")
                expected_pkl = input_path.parent / f"{input_path.stem}.pkl"
                print(f"Expected PKL file at: {expected_pkl}")
                return
        elif is_motion_npy(npy_data):
            print("Detected motion NPY format (typically 500 Hz version of single motion)")
            original_npy_path = Path(args.original_npy) if args.original_npy else None
            
            if args.codebook_only:
                # For codebook-only plot, we need to extract codebook sequence using VQVAE model
                if args.checkpoint is None:
                    print("Error: --codebook-only with NPY motion format requires --checkpoint to extract codebook sequence")
                    return
                if original_npy_path is None:
                    print("Error: --codebook-only with NPY motion format requires --original-npy")
                    return
                
                plot_data = extract_codebook_sequence_from_npy_motion(
                    input_path, original_npy_path, args.config, args.checkpoint
                )
                if plot_data is None:
                    print("Failed to extract codebook sequence from NPY motion file.")
                    return
            else:
                # For comparison plot, just extract features
                plot_data = convert_npy_motion_to_codebook_format(input_path, original_npy_path, args.config)
                if plot_data is None:
                    print("Failed to convert NPY motion data.")
                    return
        else:
            print(f"Error: Unknown NPY file format")
            print(f"Expected either:")
            print(f"  1. Codebook sequence NPY (1D integer array) - requires corresponding PKL file")
            print(f"  2. Motion NPY (2D array with 32 columns) - requires --original-npy")
            return
    else:
        # PKL file
        print(f"Loading PKL file: {input_path}")
        data = load_pkl_data(input_path)
        
        if data is None:
            print("Failed to load PKL data.")
            return
        
        # Check format and convert if necessary
        if is_codebook_sequence_format(data):
            print("Detected codebook sequence format")
            plot_data = data
        elif is_amass_format(data):
            print("Detected AMASS format")
            original_pkl_path = Path(args.original_pkl) if args.original_pkl else None
            original_npy_path = Path(args.original_npy) if args.original_npy else None
            
            if args.codebook_only:
                # For codebook-only plot, we need to extract codebook sequence using VQVAE model
                if args.checkpoint is None:
                    print("Error: --codebook-only with AMASS format requires --checkpoint to extract codebook sequence")
                    return
                if original_pkl_path is None and original_npy_path is None:
                    print("Error: --codebook-only with AMASS format requires --original-pkl or --original-npy")
                    return
                
                original_path = original_pkl_path if original_pkl_path is not None else original_npy_path
                if original_path.suffix == '.npy':
                    plot_data = extract_codebook_sequence_from_npy_motion(
                        input_path, original_path, args.config, args.checkpoint
                    )
                else:
                    plot_data = extract_codebook_sequence_from_amass(
                        input_path, original_path, args.config, args.checkpoint
                    )
                if plot_data is None:
                    print("Failed to extract codebook sequence from AMASS format file.")
                    return
            else:
                # For comparison plot, just extract features
                # Support both PKL and NPY for original
                original_path = original_pkl_path if original_pkl_path is not None else original_npy_path
                plot_data = convert_amass_to_codebook_format(input_path, original_path, args.config)
                if plot_data is None:
                    print("Failed to convert AMASS format data.")
                    return
        else:
            print(f"Error: Unknown PKL file format")
            print(f"Expected either:")
            print(f"  1. Codebook sequence format (with 'original_motion' and 'reconstructed_motion' keys)")
            print(f"  2. AMASS format (with motion keys like '0-ACCAD_...') - requires --original-pkl or --original-npy")
            return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.codebook_only:
            output_path = input_path.parent / f"{input_path.stem}_codebook_changes.png"
        else:
            output_path = input_path.parent / f"{input_path.stem}_plot.png"
    
    # Save CSV if requested
    if args.save_csv:
        csv_path = Path(args.save_csv)
        if 'codebook_per_frame' not in plot_data:
            print("Error: Cannot save CSV - missing codebook_per_frame data")
            print("Available keys:", list(plot_data.keys()))
            return
        save_codebook_sequence_csv(plot_data, csv_path)
    
    # Create plot
    if args.codebook_only:
        # Only plot codebook changes
        if 'codebook_per_frame' not in plot_data or 'codebook_changed' not in plot_data:
            print("Error: --codebook-only requires codebook sequence PKL file with codebook data")
            print("Available keys:", list(plot_data.keys()))
            return
        plot_codebook_changes(
            plot_data,
            output_path=output_path,
            show_plot=not args.no_show
        )
    else:
        # Plot original vs reconstructed (with codebook if available)
        plot_original_vs_reconstructed(
            plot_data,
            output_path=output_path,
            show_plot=not args.no_show,
            num_dof_to_plot=args.num_dof
        )


if __name__ == "__main__":
    main()


'''
Example usage:


# Plot from codebook sequence NPY file (requires corresponding PKL file)
python scripts/plot_codebook_pkl.py \
  --input outputs/codebook_sequences/codebook_sequence_000.npy \
  --codebook-only \
  --output evaluation_plots_v2/motion_1_codebook_changes.png

# Plot from NPY motion file (requires original NPY file)
python scripts/plot_codebook_pkl.py \
  --input outputs/vqvae_amass_motions_npy/vqvae_motion_000.npy \
  --original-npy /path/to/original/saved_desired_states_000.npy \
  --output evaluation_plots_v2/motion_0_original_vs_reconstructed.png

# Plot codebook changes from NPY motion file (requires checkpoint and original NPY)
python scripts/plot_codebook_pkl.py \
  --input outputs/vqvae_amass_motions_npy/vqvae_motion_000.npy \
  --original-npy /path/to/original/saved_desired_states_000.npy \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/outputs/run_0_300_32/best_model.ckpt \
  --codebook-only \
  --output evaluation_plots_v2/motion_0_codebook_changes.png

# Save CSV and plot (can combine both)
python scripts/plot_codebook_pkl.py \
  --input outputs/vqvae_amass_motions/vqvae_motion_000.pkl \
  --original-pkl /home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/outputs/run_0_300_32/best_model.ckpt \
  --codebook-only \
  --save-csv /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/codebook_sequences/motion_0_codebook_ids.csv \
  --output evaluation_plots_v2/motion_0_codebook_changes.png

'''

