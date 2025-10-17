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
        Extracts: root deltas (local frame), DOF positions, DOF velocities.
        """
        logger.info(f"Loading motion data from: {motion_file}")
        
        # Load PKL file directly (simplified - only PKL support)
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

        # 1) Root linear velocity (global)
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
    
