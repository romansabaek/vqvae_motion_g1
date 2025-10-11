"""
SIMPLIFIED Motion data adapter for loading and converting motion data to MVQ format for VQ-VAE training.
Focuses only on essential features: root deltas, joint positions, joint velocities.
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
    SIMPLIFIED adapter class to load motion data and convert to MVQ format for VQ-VAE training.
    Only extracts essential features: root deltas, joint positions, joint velocities.
    """
    
    # MVQ Format Parameters (matching reference implementation)
    NUM_JOINTS = 24  # SMPL has 24 joints
    ROOT_DELTAS_DIM = 4  # dx, dy, dz, dyaw
    JOINT_POSITIONS_DIM = NUM_JOINTS * 3  # 72 (24 joints * 3 coordinates)
    JOINT_VELOCITIES_DIM = NUM_JOINTS * 3  # 72 (24 joints * 3 velocities)
    JOINT_ORIENTATIONS_DIM = NUM_JOINTS * 9  # 216 (24 joints * 9-D rotation matrices)
    TOTAL_FRAME_SIZE = ROOT_DELTAS_DIM + JOINT_POSITIONS_DIM + JOINT_VELOCITIES_DIM + JOINT_ORIENTATIONS_DIM  # 364
    
    # Feature Indices
    ROOT_DELTAS_START = 0
    ROOT_DELTAS_END = ROOT_DELTAS_DIM
    JOINT_POSITIONS_START = ROOT_DELTAS_END
    JOINT_POSITIONS_END = JOINT_POSITIONS_START + JOINT_POSITIONS_DIM
    JOINT_VELOCITIES_START = JOINT_POSITIONS_END
    JOINT_VELOCITIES_END = JOINT_VELOCITIES_START + JOINT_VELOCITIES_DIM
    JOINT_ORIENTATIONS_START = JOINT_VELOCITIES_END
    JOINT_ORIENTATIONS_END = TOTAL_FRAME_SIZE
    
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
        SIMPLIFIED: Load motion data and convert to MVQ format.
        Only extracts essential features: root deltas, joint positions, joint velocities.
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
            motion_features = self._extract_simple_features(motion_data)
            all_features.append(motion_features)
            
            # Update end index
            current_end += motion_features.shape[0]
            end_indices.append(current_end - 1)
        
        # Concatenate all motion features
        self.mocap_data = torch.cat(all_features, dim=0)
        self.end_indices = end_indices
        self.frame_size = self.mocap_data.shape[1]
        
        logger.info(f"Loaded motion data: {self.mocap_data.shape[0]} frames, {self.frame_size} features")
        logger.info(f"Number of motion sequences: {len(self.end_indices)}")
        
        return self.mocap_data, self.end_indices, self.frame_size
    
    def _extract_simple_features(self, motion_data) -> torch.Tensor:
        """
        SIMPLIFIED: Extract only essential features from motion data.
        Features: root deltas, joint positions, joint velocities.
        """
        # Extract basic motion data
        root_pos = torch.tensor(motion_data["root_trans_offset"], dtype=torch.float32, device=self.device)
        root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=self.device)
        dof_pos = torch.tensor(motion_data["dof"], dtype=torch.float32, device=self.device)
        
        motion_length = root_pos.shape[0]
        
        # Create MVQ frame tensor - matching reference format exactly
        # Format: [root_deltas(4), joint_positions(72), joint_velocities(72), joint_orientations(216)] = 364 dimensions
        mvq_frames = torch.zeros(motion_length, self.TOTAL_FRAME_SIZE, dtype=torch.float32, device=self.device)
        
        # Process consecutive frames for LOCAL features (matching reference exactly)
        for i in range(motion_length):
            if i == 0:
                # First frame - use zeros for deltas and velocities
                mvq_frames[i, self.ROOT_DELTAS_START:self.ROOT_DELTAS_END] = 0  # Root deltas
                
                # Use DOF data for joint positions (pad to 72 dimensions)
                dof_data = dof_pos[i]  # Shape: [23]
                joint_pos = torch.zeros(self.JOINT_POSITIONS_DIM, device=self.device)
                for j in range(0, self.JOINT_POSITIONS_DIM, dof_data.shape[0]):
                    end_idx = min(j + dof_data.shape[0], self.JOINT_POSITIONS_DIM)
                    joint_pos[j:end_idx] = dof_data[:end_idx-j]
                mvq_frames[i, self.JOINT_POSITIONS_START:self.JOINT_POSITIONS_END] = joint_pos
                
                mvq_frames[i, self.JOINT_VELOCITIES_START:self.JOINT_VELOCITIES_END] = 0  # Joint velocities
                mvq_frames[i, self.JOINT_ORIENTATIONS_START:self.JOINT_ORIENTATIONS_END] = 0  # Joint orientations
            else:
                # Convert to LOCAL frame features (matching reference motion_lib_mvq.py)
                
                # 1. LOCAL root deltas (relative to robot's heading frame)
                global_delta = root_pos[i] - root_pos[i-1]
                local_delta = self._global_to_local_heading(global_delta, root_rot[i-1])
                yaw_delta = self._compute_yaw_delta(root_rot[i-1], root_rot[i])
                
                mvq_frames[i, self.ROOT_DELTAS_START:self.ROOT_DELTAS_START+3] = local_delta  # LOCAL position deltas (dx, dy, dz)
                mvq_frames[i, self.ROOT_DELTAS_START+3] = yaw_delta  # Yaw delta
                
                # 2. Joint positions (use DOF directly - pad to required size)
                # DOF has 23 elements, but we need 72 for joint positions
                dof_data = dof_pos[i]  # Shape: [23]
                # Pad DOF data to 72 dimensions (repeat pattern)
                joint_pos = torch.zeros(self.JOINT_POSITIONS_DIM, device=self.device)
                # Repeat DOF data to fill 72 dimensions (23 * 3 = 69, pad remaining 3)
                for j in range(0, self.JOINT_POSITIONS_DIM, dof_data.shape[0]):
                    end_idx = min(j + dof_data.shape[0], self.JOINT_POSITIONS_DIM)
                    joint_pos[j:end_idx] = dof_data[:end_idx-j]
                mvq_frames[i, self.JOINT_POSITIONS_START:self.JOINT_POSITIONS_END] = joint_pos
                
                # 3. Joint velocities (compute from DOF differences)
                dof_data_prev = dof_pos[i-1]  # Shape: [23]
                joint_pos_prev = torch.zeros(self.JOINT_POSITIONS_DIM, device=self.device)
                # Repeat previous DOF data to fill 72 dimensions
                for j in range(0, self.JOINT_POSITIONS_DIM, dof_data_prev.shape[0]):
                    end_idx = min(j + dof_data_prev.shape[0], self.JOINT_POSITIONS_DIM)
                    joint_pos_prev[j:end_idx] = dof_data_prev[:end_idx-j]
                joint_vel = (joint_pos - joint_pos_prev) / self.DT
                mvq_frames[i, self.JOINT_VELOCITIES_START:self.JOINT_VELOCITIES_END] = joint_vel
                
                # 4. Joint orientations (9-D rotation matrices) - simplified for now
                mvq_frames[i, self.JOINT_ORIENTATIONS_START:self.JOINT_ORIENTATIONS_END] = 0  # TODO: Implement proper joint orientations
        
        return mvq_frames
    
    def _compute_yaw_delta(self, rot_t, rot_tp1):
        """Compute yaw delta matching reference implementation."""
        # Extract yaw from quaternions (WXYZ format)
        yaw_t = torch.atan2(rot_t[3], rot_t[0]) * 2  # z, w components
        yaw_tp1 = torch.atan2(rot_tp1[3], rot_tp1[0]) * 2
        
        yaw_delta = yaw_tp1 - yaw_t
        
        # Normalize to [-π, π] (matching reference)
        yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
        
        return yaw_delta
    
    def _global_to_local_heading(self, global_delta, root_rot):
        """Convert global delta to local heading frame (matching reference)."""
        # Extract heading quaternion components (WXYZ format)
        cos_yaw = root_rot[0]  # w component
        sin_yaw = root_rot[3]  # z component
        
        # Convert global delta to local heading frame
        local_delta = torch.zeros_like(global_delta)
        local_delta[0] = global_delta[0] * cos_yaw + global_delta[1] * sin_yaw   # Forward
        local_delta[1] = -global_delta[0] * sin_yaw + global_delta[1] * cos_yaw  # Right  
        local_delta[2] = global_delta[2]  # Up (unchanged)
        
        return local_delta
    
