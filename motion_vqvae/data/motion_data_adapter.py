"""
Motion data adapter for G1 humanoid AMASS data.
Converts G1 humanoid motion data to MVQ format for VQ-VAE training.
Optimized for G1 humanoid with 23 DOF and proper local frame features.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import joblib
import glob
import re
from .torch_utils import quat_diff, quat_to_exp_map

logger = logging.getLogger(__name__)


class MotionDataAdapter:
    """
    G1 Humanoid motion data adapter for VQ-VAE training.
    Converts G1 humanoid AMASS data to MVQ format with proper local frame features.
    Optimized for G1 humanoid with 23 DOF structure.
    """
    
    # G1 Humanoid MVQ Format Parameters
    NUM_DOF = 23  # G1 humanoid has 23 DOF
    ROOT_DELTAS_DIM = 4  # dx, dy, dz, dyaw (local frame)
    DOF_POSITIONS_DIM = NUM_DOF  # 23 DOF positions
    DOF_VELOCITIES_DIM = NUM_DOF  # 23 DOF velocities
    # Note: Removed joint orientations for G1 - using DOF-based representation
    TOTAL_FRAME_SIZE = ROOT_DELTAS_DIM + DOF_POSITIONS_DIM + DOF_VELOCITIES_DIM  # 4 + 23 + 23 = 50
    
    # Feature Indices (G1 Humanoid)
    ROOT_DELTAS_START = 0
    ROOT_DELTAS_END = ROOT_DELTAS_DIM  # 0:4
    DOF_POSITIONS_START = ROOT_DELTAS_END  # 4
    DOF_POSITIONS_END = DOF_POSITIONS_START + DOF_POSITIONS_DIM  # 4:27
    DOF_VELOCITIES_START = DOF_POSITIONS_END  # 27
    DOF_VELOCITIES_END = DOF_VELOCITIES_START + DOF_VELOCITIES_DIM  # 27:50
    
    # Motion Parameters
    FPS = 30  # Frames per second
    DT = 1.0 / FPS  # Time step
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Motion data storage
        self.mocap_data = None
        self.end_indices = None
        self.frame_size = None
        self._loaded = False
    
    def load_motion_data(self, motion_file: str, motion_ids: Optional[list] = None) -> Tuple[torch.Tensor, np.ndarray, int]:
        """
        Load G1 humanoid motion data and convert to MVQ format.
        Supports both PKL (AMASS) and NPY file formats.
        Also supports directories containing multiple NPY files.
        Extracts: root deltas (local frame), DOF positions, DOF velocities.
        """
        logger.info(f"Loading motion data from: {motion_file}")
        
        motion_file_path = Path(motion_file)
        
        # Check if it's a directory
        if motion_file_path.is_dir():
            return self._load_npy_directory(motion_file, motion_ids)
        
        # Check file extension to determine format
        if motion_file_path.suffix.lower() == '.npy':
            return self._load_npy_data(motion_file, motion_ids)
        else:
            return self._load_pkl_data(motion_file, motion_ids)
    
    def _load_pkl_data(self, motion_file: str, motion_ids: Optional[list] = None) -> Tuple[torch.Tensor, np.ndarray, int]:
        """Load motion data from PKL file (AMASS format)."""
        # Load PKL file
        motion_data_dict = joblib.load(motion_file)
        motion_keys_all = list(motion_data_dict.keys())
        
        # By default, use ALL motions in the file (match reference behavior)
        if motion_ids is None:
            selected_keys = motion_keys_all
        else:
            if len(motion_ids) == 0:
                selected_keys = []
            elif isinstance(motion_ids[0], int):
                valid_idx = [i for i in motion_ids if 0 <= i < len(motion_keys_all)]
                if len(valid_idx) < len(motion_ids):
                    logger.warning("Some motion indices are out of bounds and will be skipped.")
                selected_keys = [motion_keys_all[i] for i in valid_idx]
            else:
                motion_ids_set = set(motion_ids)
                selected_keys = [k for k in motion_keys_all if k in motion_ids_set]
        
        # Extract features for specified motions
        all_features = []
        end_indices = []
        current_end = 0
        
        for motion_key in selected_keys:
            motion_data = motion_data_dict[motion_key]
            
            # Extract features for this motion
            motion_features = self._extract_g1_features(motion_data)
            all_features.append(motion_features)
            
            # Update end index
            current_end += motion_features.shape[0]
            end_indices.append(current_end - 1)
        
        # Concatenate all motion features
        # Keep dataset tensors on CPU for efficient DataLoader pin/move; move to device later in training loop
        self.mocap_data = torch.cat(all_features, dim=0).cpu()
        # Return end indices as numpy array for consistency with reference
        self.end_indices = np.array(end_indices, dtype=np.int64)
        self.frame_size = self.mocap_data.shape[1]
        
        logger.info(f"Loaded G1 humanoid motion data: {self.mocap_data.shape[0]} frames, {self.frame_size} features")
        logger.info(f"Frame size breakdown: root_deltas({self.ROOT_DELTAS_DIM}) + dof_positions({self.DOF_POSITIONS_DIM}) + dof_velocities({self.DOF_VELOCITIES_DIM}) = {self.TOTAL_FRAME_SIZE}")
        logger.info(f"Number of motion sequences: {len(self.end_indices)}")
        
        self._loaded = True
        return self.mocap_data, self.end_indices, self.frame_size
    
    def _load_npy_data(self, motion_file: str, motion_ids: Optional[list] = None) -> Tuple[torch.Tensor, np.ndarray, int]:
        """Load motion data from NPY file format."""
        # Load the continuous trajectory data from the .npy file
        trajectory_data = np.load(motion_file)
        logger.info(f"Loaded NPY file with shape: {trajectory_data.shape}")
        
        # NPY format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id/transition_flag]
        # The last column is used to split motions
        num_cols = trajectory_data.shape[1]
        
        # Determine if last column is motion_id or transition_flag
        # If values are integers and change infrequently, it's likely motion_id
        last_col = trajectory_data[:, -1]
        unique_values = np.unique(last_col)
        
        # Split trajectory into motion clips based on motion_id changes
        if len(trajectory_data) > 1:
            # Find indices where motion_id changes
            changes = np.where(last_col[:-1] != last_col[1:])[0] + 1
            split_indices = [0] + changes.tolist() + [len(trajectory_data)]
        else:
            split_indices = [0, len(trajectory_data)]
        
        # Create motion clips
        motion_clips = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            clip_data = trajectory_data[start_idx:end_idx]
            motion_clips.append(clip_data)
        
        logger.info(f"Detected {len(motion_clips)} motion clip(s) in NPY file")
        
        # Select which clips to process based on motion_ids
        if motion_ids is None:
            selected_clips = motion_clips
        else:
            if len(motion_ids) == 0:
                selected_clips = []
            else:
                # motion_ids are indices into the clips
                valid_idx = [i for i in motion_ids if 0 <= i < len(motion_clips)]
                if len(valid_idx) < len(motion_ids):
                    logger.warning(f"Some motion indices are out of bounds. Valid range: 0-{len(motion_clips)-1}")
                selected_clips = [motion_clips[i] for i in valid_idx]
        
        # Extract features for each selected clip
        all_features = []
        end_indices = []
        current_end = 0
        
        for clip_data in selected_clips:
            # Extract features from this clip
            motion_features = self._extract_g1_features_from_npy(clip_data)
            all_features.append(motion_features)
            
            # Update end index
            current_end += motion_features.shape[0]
            end_indices.append(current_end - 1)
        
        # Concatenate all motion features
        if len(all_features) == 0:
            raise ValueError("No valid motion clips found after filtering")
        
        self.mocap_data = torch.cat(all_features, dim=0).cpu()
        self.end_indices = np.array(end_indices, dtype=np.int64)
        self.frame_size = self.mocap_data.shape[1]
        
        logger.info(f"Loaded G1 humanoid motion data from NPY: {self.mocap_data.shape[0]} frames, {self.frame_size} features")
        logger.info(f"Frame size breakdown: root_deltas({self.ROOT_DELTAS_DIM}) + dof_positions({self.DOF_POSITIONS_DIM}) + dof_velocities({self.DOF_VELOCITIES_DIM}) = {self.TOTAL_FRAME_SIZE}")
        logger.info(f"Number of motion sequences: {len(self.end_indices)}")
        
        self._loaded = True
        return self.mocap_data, self.end_indices, self.frame_size
    
    def _load_npy_directory(self, directory: str, motion_ids: Optional[list] = None) -> Tuple[torch.Tensor, np.ndarray, int]:
        """Load multiple NPY files from a directory. Files should match saved_desired_states_*.npy pattern."""
        directory_path = Path(directory)
        if not directory_path.is_dir():
            raise ValueError(f"Expected directory, got: {directory}")
        
        # Find NPY files: try direct, then common subdirs, then recursive
        patterns = [
            directory_path / "*.npy",
            directory_path / "each_motion_npy/*.npy",
            directory_path / "motions/*.npy",
            directory_path / "**/*.npy",
        ]
        
        all_npy_files = []
        for pattern in patterns:
            all_npy_files = sorted(glob.glob(str(pattern), recursive=(pattern.name == "**")))
            if all_npy_files:
                break
        
        if not all_npy_files:
            raise ValueError(f"No NPY files found in directory: {directory}")
        
        logger.info(f"Found {len(all_npy_files)} NPY files")
        
        # Extract motion IDs from filenames: saved_desired_states_123.npy -> 123
        def get_motion_id(filepath: str) -> int:
            match = re.search(r'saved_desired_states_(\d+)\.npy', Path(filepath).name)
            return int(match.group(1)) if match else 0
        
        file_motion_map = [(fp, get_motion_id(fp)) for fp in all_npy_files]
        file_motion_map.sort(key=lambda x: x[1])  # Sort by motion_id
        
        # Filter by motion_ids if provided
        if motion_ids is not None:
            motion_ids_set = set(motion_ids)
            file_motion_map = [(fp, mid) for fp, mid in file_motion_map if mid in motion_ids_set]
            if not file_motion_map:
                raise ValueError(f"No NPY files match motion_ids {motion_ids}")
            logger.info(f"Filtered to {len(file_motion_map)} files")
        
        # Load each NPY file and extract features
        all_features = []
        end_indices = []
        current_end = 0
        
        for npy_file, motion_id in file_motion_map:
            logger.info(f"Loading: {Path(npy_file).name} (motion_id: {motion_id})")
            trajectory_data = np.load(npy_file)
            motion_features = self._extract_g1_features_from_npy(trajectory_data)
            all_features.append(motion_features)
            current_end += motion_features.shape[0]
            end_indices.append(current_end - 1)
        
        # Concatenate and store
        self.mocap_data = torch.cat(all_features, dim=0).cpu()
        self.end_indices = np.array(end_indices, dtype=np.int64)
        self.frame_size = self.mocap_data.shape[1]
        
        logger.info(f"Loaded {self.mocap_data.shape[0]} frames, {self.frame_size} features, {len(self.end_indices)} sequences")
        self._loaded = True
        return self.mocap_data, self.end_indices, self.frame_size
    
    def _extract_g1_features_from_npy(self, clip_data: np.ndarray) -> torch.Tensor:
        """
        Extract G1 humanoid features from NPY format data.
        NPY format: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id/transition_flag]
        Root rotation is in WXYZ format, needs conversion to XYZW.
        """
        num_frames = clip_data.shape[0]
        num_cols = clip_data.shape[1]
        
        # Extract time stamps
        time_stamps = clip_data[:, 0]
        dt = np.mean(np.diff(time_stamps)) if len(time_stamps) > 1 else 1.0 / self.FPS
        fps = 1.0 / dt
        
        # Extract root position (columns 1:4)
        root_pos = torch.tensor(clip_data[:, 1:4], dtype=torch.float32, device=self.device)
        
        # Extract root rotation (columns 4:8) - WXYZ format
        root_rot_wxyz = torch.tensor(clip_data[:, 4:8], dtype=torch.float32, device=self.device)
        # Convert WXYZ to XYZW format
        root_rot = root_rot_wxyz[:, [1, 2, 3, 0]]
        
        # Extract DOF positions
        # Handle both formats: with/without token column
        if num_cols == 33:
            # Has token column: [time, root_pos(3), root_rot(4), dof_pos(23), token(1), motion_id(1)]
            dof_pos = torch.tensor(clip_data[:, 8:31], dtype=torch.float32, device=self.device)
        else:
            # No token column: [time, root_pos(3), root_rot(4), dof_pos(23), motion_id(1)]
            dof_pos = torch.tensor(clip_data[:, 8:-1], dtype=torch.float32, device=self.device)
        
        # Validate DOF dimensions
        if dof_pos.shape[1] != self.NUM_DOF:
            raise ValueError(f"Expected {self.NUM_DOF} DOF, got {dof_pos.shape[1]}")
        
        # 1) Root linear velocity (global) - improved smoothing
        root_vel = torch.zeros_like(root_pos)
        root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
        root_vel[-1, :] = root_vel[-2, :]
        root_vel = self._smooth(root_vel, 19)
        
        # 2) Root angular velocity (global, exp-map via quaternion difference)
        root_ang_vel = torch.zeros_like(root_pos)
        root_drot = quat_diff(root_rot[:-1], root_rot[1:])
        root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot)
        root_ang_vel[-1, :] = root_ang_vel[-2, :]
        root_ang_vel = self._smooth(root_ang_vel, 19)
        
        # 3) DOF velocities
        dof_vel = torch.zeros_like(dof_pos)
        dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
        dof_vel[-1, :] = dof_vel[-2, :]
        dof_vel = self._smooth(dof_vel, 19)
        
        # 4) Convert velocities to LOCAL frame
        lin_vel_local = self.quat_rotate_inverse(root_rot, root_vel)
        ang_vel_local = self.quat_rotate_inverse(root_rot, root_ang_vel)
        
        # Vectorized assembly of MVQ frames (no Python loop)
        mvq_frames = torch.zeros(num_frames, self.TOTAL_FRAME_SIZE, dtype=torch.float32, device=self.device)
        
        # Local root deltas per frame (use smoothed velocities divided by fps)
        mvq_frames[:, self.ROOT_DELTAS_START:self.ROOT_DELTAS_START+3] = lin_vel_local / fps
        mvq_frames[:, self.ROOT_DELTAS_START+3] = ang_vel_local[:, 2] / fps  # Δyaw approximation from local wz
        
        # DOF positions and velocities
        mvq_frames[:, self.DOF_POSITIONS_START:self.DOF_POSITIONS_END] = dof_pos
        mvq_frames[:, self.DOF_VELOCITIES_START:self.DOF_VELOCITIES_END] = dof_vel
        
        return mvq_frames


    @property
    def FRAME_SIZE(self) -> int:
        if self.frame_size is None:
            return self.TOTAL_FRAME_SIZE
        return int(self.frame_size)

    def get_mvq_data(self) -> torch.Tensor:
        """
        Return cached MVQ data tensor [F, C]. Must call load_motion_data() first.
        """
        if not self._loaded or self.mocap_data is None:
            raise RuntimeError("MotionDataAdapter: call load_motion_data() before get_mvq_data().")
        return self.mocap_data

    def get_mvq_end_indices(self) -> np.ndarray:
        """
        Return cached end indices as numpy array.
        """
        if not self._loaded or self.end_indices is None:
            raise RuntimeError("MotionDataAdapter: call load_motion_data() before get_mvq_end_indices().")
        return self.end_indices
    
    def _extract_g1_features(self, motion_data) -> torch.Tensor:
        """
        Extract G1 humanoid features from AMASS motion data using motion_lib-style smoothing.
        Format: [root_deltas(4), dof_positions(23), dof_velocities(23)] = 50 dimensions
        """
        # Extract G1 humanoid motion data
        root_pos = torch.tensor(motion_data["root_trans_offset"], dtype=torch.float32, device=self.device)
        root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=self.device)  # XYZW format [x, y, z, w]
        dof_pos = torch.tensor(motion_data["dof"], dtype=torch.float32, device=self.device)

        num_frames = root_pos.shape[0]
        fps = motion_data.get("fps", self.FPS)

        # Validate DOF dimensions
        if dof_pos.shape[1] != self.NUM_DOF:
            raise ValueError(f"Expected {self.NUM_DOF} DOF, got {dof_pos.shape[1]}")

        # 1) Root linear velocity (global) - improved smoothing
        root_vel = torch.zeros_like(root_pos)
        root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
        root_vel[-1, :] = root_vel[-2, :]
        root_vel = self._smooth(root_vel, 19)

        # 2) Root angular velocity (global, exp-map via quaternion difference)
        root_ang_vel = torch.zeros_like(root_pos)
        root_drot = quat_diff(root_rot[:-1], root_rot[1:])
        root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot)
        root_ang_vel[-1, :] = root_ang_vel[-2, :]
        root_ang_vel = self._smooth(root_ang_vel, 19)

        # 3) DOF velocities
        dof_vel = torch.zeros_like(dof_pos)
        dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
        dof_vel[-1, :] = dof_vel[-2, :]
        dof_vel = self._smooth(dof_vel, 19)

        # 4) Convert velocities to LOCAL frame
        lin_vel_local = self.quat_rotate_inverse(root_rot, root_vel)
        ang_vel_local = self.quat_rotate_inverse(root_rot, root_ang_vel)

        # Vectorized assembly of MVQ frames (no Python loop)
        mvq_frames = torch.zeros(num_frames, self.TOTAL_FRAME_SIZE, dtype=torch.float32, device=self.device)

        # Local root deltas per frame (use smoothed velocities divided by fps)
        mvq_frames[:, self.ROOT_DELTAS_START:self.ROOT_DELTAS_START+3] = lin_vel_local / fps
        mvq_frames[:, self.ROOT_DELTAS_START+3] = ang_vel_local[:, 2] / fps  # Δyaw approximation from local wz

        # DOF positions and velocities
        mvq_frames[:, self.DOF_POSITIONS_START:self.DOF_POSITIONS_END] = dof_pos
        mvq_frames[:, self.DOF_VELOCITIES_START:self.DOF_VELOCITIES_END] = dof_vel

        return mvq_frames
    
    def _smooth(self, x, box_pts):
        """Smooth data using moving average (from motion_lib.py)."""
        box = torch.ones(box_pts, device=self.device) / box_pts
        num_channels = x.shape[1]
        x_reshaped = x.T.unsqueeze(0)
        smoothed = torch.nn.functional.conv1d(
            x_reshaped,
            box.view(1, 1, -1).expand(num_channels, 1, -1),
            groups=num_channels,
            padding='same'
        )
        return smoothed.squeeze(0).T
    
    def _normalize_velocities(self, velocities):
        """Normalize velocities to prevent extreme values that hurt training."""
        # Clip extreme velocities to prevent training instability
        max_vel = 10.0  # reasonable maximum velocity in rad/s or m/s
        velocities = torch.clamp(velocities, -max_vel, max_vel)
        
        # Apply soft normalization to reduce variance
        vel_std = velocities.std(dim=0, keepdim=True)
        vel_mean = velocities.mean(dim=0, keepdim=True)
        # Avoid division by zero
        vel_std = torch.where(vel_std < 1e-6, torch.ones_like(vel_std), vel_std)
        
        # Soft normalization: 0.1 * (x - mean) / std + 0.9 * x
        normalized = 0.1 * (velocities - vel_mean) / vel_std + 0.9 * velocities
        return normalized
    
    
    def quat_rotate_inverse(self, q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c
    
