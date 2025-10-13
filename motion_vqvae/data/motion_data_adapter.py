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
    
    def load_motion_data(self, motion_file: str, motion_ids: Optional[list] = None) -> Tuple[torch.Tensor, list, int]:
        """
        Load G1 humanoid motion data and convert to MVQ format.
        Extracts: root deltas (local frame), DOF positions, DOF velocities.
        """
        logger.info(f"Loading motion data from: {motion_file}")
        
        # Load PKL file directly (simplified - only PKL support)
        motion_data_dict = joblib.load(motion_file)
        motion_keys = list(motion_data_dict.keys())
        
        if motion_ids is None:
            motion_ids = [0]  # Default to first motion
        
        # Extract features for specified motions
        all_features = []
        end_indices = []
        current_end = 0
        
        for motion_id in motion_ids:
            if motion_id >= len(motion_keys):
                logger.warning(f"Motion ID {motion_id} out of bounds. Skipping.")
                continue
                
            motion_key = motion_keys[motion_id]
            motion_data = motion_data_dict[motion_key]
            
            # Extract features for this motion
            motion_features = self._extract_g1_features(motion_data)
            all_features.append(motion_features)
            
            # Update end index
            current_end += motion_features.shape[0]
            end_indices.append(current_end - 1)
        
        # Concatenate all motion features
        self.mocap_data = torch.cat(all_features, dim=0)
        self.end_indices = end_indices
        self.frame_size = self.mocap_data.shape[1]
        
        logger.info(f"Loaded G1 humanoid motion data: {self.mocap_data.shape[0]} frames, {self.frame_size} features")
        logger.info(f"Frame size breakdown: root_deltas({self.ROOT_DELTAS_DIM}) + dof_positions({self.DOF_POSITIONS_DIM}) + dof_velocities({self.DOF_VELOCITIES_DIM}) = {self.TOTAL_FRAME_SIZE}")
        logger.info(f"Number of motion sequences: {len(self.end_indices)}")
        
        return self.mocap_data, self.end_indices, self.frame_size
    
    def _extract_g1_features(self, motion_data) -> torch.Tensor:
        """
        Extract G1 humanoid features from AMASS motion data using motion_lib-style smoothing.
        Format: [root_deltas(4), dof_positions(23), dof_velocities(23)] = 50 dimensions
        """
        # Extract G1 humanoid motion data
        root_pos = torch.tensor(motion_data["root_trans_offset"], dtype=torch.float32, device=self.device)
        root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=self.device)
        dof_pos = torch.tensor(motion_data["dof"], dtype=torch.float32, device=self.device)
        
        motion_length = root_pos.shape[0]
        fps = motion_data.get("fps", self.FPS)
        
        # Validate DOF dimensions
        if dof_pos.shape[1] != self.NUM_DOF:
            raise ValueError(f"Expected {self.NUM_DOF} DOF, got {dof_pos.shape[1]}")
        
        # MOTION_LIB-STYLE SMOOTHING: Compute velocities first, then smooth them
        # 1. Compute root velocities (global frame)
        root_vel = torch.zeros_like(root_pos)
        root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
        root_vel[-1, :] = root_vel[-2, :]
        root_vel = self._smooth(root_vel, 19)  # Smooth like motion_lib
        
        # 2. Compute root angular velocities using quaternion differences
        root_ang_vel = torch.zeros_like(root_pos)
        root_drot = self._quat_diff(root_rot[:-1], root_rot[1:])
        root_ang_vel[:-1, :] = fps * self._quat_to_exp_map(root_drot)
        root_ang_vel[-1, :] = root_ang_vel[-2, :]
        root_ang_vel = self._smooth(root_ang_vel, 19)  # Smooth like motion_lib
        
        # 3. Compute DOF velocities
        dof_vel = torch.zeros_like(dof_pos)
        dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
        dof_vel[-1, :] = dof_vel[-2, :]
        dof_vel = self._smooth(dof_vel, 19)  # Smooth like motion_lib
        
        # 4. Convert global velocities to LOCAL frame (like deploy_mujoco_falcon_motionlib.py)
        lin_vel_local = self._quat_rotate_inverse(root_rot, root_vel)
        ang_vel_local = self._quat_rotate_inverse(root_rot, root_ang_vel)
        
        # Create MVQ frame tensor for G1 humanoid
        # Format: [root_deltas(4), dof_positions(23), dof_velocities(23)] = 50 dimensions
        mvq_frames = torch.zeros(motion_length, self.TOTAL_FRAME_SIZE, dtype=torch.float32, device=self.device)
        
        # Process frames for LOCAL features
        for i in range(motion_length):
            # 1. LOCAL root deltas (dx, dy, dz, dyaw in local frame)
            # Use smoothed local velocities as deltas (divide by fps to get per-frame deltas)
            mvq_frames[i, self.ROOT_DELTAS_START:self.ROOT_DELTAS_START+3] = lin_vel_local[i] / fps  # LOCAL position deltas (dx, dy, dz)
            mvq_frames[i, self.ROOT_DELTAS_START+3] = ang_vel_local[i, 2] / fps  # Yaw delta (angular velocity around z-axis)
            
            # 2. DOF positions (use directly - no padding needed for G1)
            mvq_frames[i, self.DOF_POSITIONS_START:self.DOF_POSITIONS_END] = dof_pos[i]
            
            # 3. DOF velocities (use smoothed velocities)
            mvq_frames[i, self.DOF_VELOCITIES_START:self.DOF_VELOCITIES_END] = dof_vel[i]
        
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
    
    def _quat_diff(self, q1, q2):
        """Compute quaternion difference q1^{-1} * q2 (from motion_lib.py)."""
        # q1^{-1} * q2 where q^{-1} = [w, -x, -y, -z] for unit quaternions
        q1_inv = q1.clone()
        q1_inv[:, 1:] *= -1  # Negate x, y, z components
        
        # Quaternion multiplication
        w1, x1, y1, z1 = q1_inv[:, 0], q1_inv[:, 1], q1_inv[:, 2], q1_inv[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        result = torch.zeros_like(q1)
        result[:, 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2  # w component
        result[:, 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2  # x component
        result[:, 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2  # y component
        result[:, 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2  # z component
        
        return result
    
    def _quat_to_exp_map(self, q):
        """Convert quaternion to exponential map (rotation vector) (from motion_lib.py)."""
        # Extract rotation vector components
        x, y, z = q[:, 1], q[:, 2], q[:, 3]  # x, y, z components
        
        # Compute rotation angle
        sin_angle = torch.sqrt(x*x + y*y + z*z)
        angle = 2 * torch.atan2(sin_angle, q[:, 0])
        
        # Avoid division by zero
        mask = sin_angle > 1e-6
        result = torch.zeros(q.shape[0], 3, device=self.device)
        
        result[mask, 0] = x[mask] * angle[mask] / sin_angle[mask]
        result[mask, 1] = y[mask] * angle[mask] / sin_angle[mask]
        result[mask, 2] = z[mask] * angle[mask] / sin_angle[mask]
        
        return result
    
    def _quat_rotate_inverse(self, q, v):
        """Rotate vector by inverse of quaternion (from deploy_mujoco_falcon_motionlib.py)."""
        # q^{-1} * v * q where q^{-1} = [w, -x, -y, -z] for unit quaternions
        q_inv = q.clone()
        q_inv[:, 1:] *= -1  # Negate x, y, z components
        
        # Extract components
        q_w = q_inv[:, 0]
        q_vec = q_inv[:, 1:4]
        
        # Apply rotation: v' = v * (2w^2 - 1) + 2w * (q_vec × v) + 2 * q_vec * (q_vec · v)
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
        
        return a - b + c
    
