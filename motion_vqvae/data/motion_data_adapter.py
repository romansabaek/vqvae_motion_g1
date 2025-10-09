"""
Motion data adapter for loading and converting motion data to MVQ format for VQ-VAE training.
Based on motion_lib_mvq.py reference implementation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from scipy.spatial.transform import Rotation as R

from ..utils.motion_lib import MotionLib
from ..utils.motion_lib_npy import MotionLibNpy

logger = logging.getLogger(__name__)


class MotionDataAdapter:
    """
    Adapter class to load motion data from PKL/NPY files and convert to MVQ format for VQ-VAE training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Data processing options (simplified for MVQ format)
        
        # Motion library
        self.motion_lib = None
        self.mocap_data = None
        self.end_indices = None
        self.frame_size = None
        
    
    def get_motion_lib(self):
        """Get the underlying motion library."""
        return self.motion_lib
    
    def get_motion_name(self):
        """Get the name of the loaded motion."""
        if self.motion_lib:
            return self.motion_lib.get_motion_name()
        return None
    
    def load_motion_data(self, motion_file: str, motion_ids: Optional[list] = None) -> Tuple[torch.Tensor, list, int]:
        """
        Load motion data and convert to MVQ format based on motion_lib_mvq.py reference.
        
        Args:
            motion_file: Path to motion file (PKL or NPY)
            motion_ids: Optional list of motion IDs to load (for multi-motion files)
            
        Returns:
            mocap_data: Tensor of shape [total_frames, frame_size] in MVQ format
            end_indices: List of end indices for each motion sequence
            frame_size: Number of features per frame (364 for SMPL)
        """
        logger.info(f"Loading motion data in MVQ format from: {motion_file}")
        
        # Determine file type and load appropriate motion library
        file_ext = Path(motion_file).suffix.lower()
        
        if file_ext == '.npy':
            self.motion_lib = MotionLibNpy(motion_file, self.device, motion_ids)
        elif file_ext == '.pkl':
            # For PKL files, we need to handle multiple motions differently
            # Store motion_ids for later use in feature extraction
            self.motion_ids = motion_ids
            # Load the first motion to get the motion library structure
            motion_id = motion_ids[0] if motion_ids and len(motion_ids) > 0 else None
            self.motion_lib = MotionLib(motion_file, self.device, motion_id)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .pkl, .npy")
        
        # Extract motion data in MVQ format
        self.mocap_data, self.end_indices, self.frame_size = self._extract_motion_features()
        
        logger.info(f"Loaded motion data in MVQ format: {self.mocap_data.shape[0]} frames, {self.frame_size} features")
        logger.info(f"Number of motion sequences: {len(self.end_indices)}")
        
        return self.mocap_data, self.end_indices, self.frame_size
    
    def _extract_motion_features(self) -> Tuple[torch.Tensor, list, int]:
        """
        Extract motion features in MVQ format from the loaded motion library.
        Based on motion_lib_mvq.py reference implementation.
        """
        # For PKL files with multiple motions, we need to load each motion separately
        if hasattr(self, 'motion_ids') and self.motion_ids is not None:
            return self._extract_motion_features_pkl()
        else:
            # For NPY files or single motion PKL files
            return self._extract_motion_features_single()
    
    def _extract_motion_features_pkl(self) -> Tuple[torch.Tensor, list, int]:
        """
        Extract motion features from PKL file with multiple motions.
        """
        import joblib
        
        # Load the PKL file directly to access all motions
        motion_data_dict = joblib.load(self.motion_lib._motion_files[0])
        motion_data_keys = list(motion_data_dict.keys())
        
        # Collect all motion features in MVQ format
        all_features = []
        end_indices = []
        current_end = 0
        
        for motion_id in self.motion_ids:
            if motion_id >= len(motion_data_keys):
                logger.warning(f"Motion ID {motion_id} is out of bounds. Skipping.")
                continue
                
            motion_key = motion_data_keys[motion_id]
            motion_data = motion_data_dict[motion_key]
            
            # Extract features for this motion in MVQ format
            motion_features = self._extract_single_motion_features_pkl(motion_data)
            all_features.append(motion_features)
            
            # Update end index
            current_end += motion_features.shape[0]
            end_indices.append(current_end - 1)  # 0-indexed
        
        # Concatenate all motion features
        mocap_data = torch.cat(all_features, dim=0)
        frame_size = mocap_data.shape[1]
        
        return mocap_data, end_indices, frame_size
    
    def _extract_motion_features_single(self) -> Tuple[torch.Tensor, list, int]:
        """
        Extract motion features from single motion library (NPY or single PKL).
        """
        # Get motion data from the library
        num_motions = self.motion_lib.num_motions()
        motion_lengths = self.motion_lib._motion_num_frames
        
        # Collect all motion features in MVQ format
        all_features = []
        end_indices = []
        current_end = 0
        
        for motion_id in range(num_motions):
            # Get motion length
            motion_length = motion_lengths[motion_id].item()
            
            # Extract features for this motion in MVQ format
            motion_features = self._extract_single_motion_features(motion_id, motion_length)
            all_features.append(motion_features)
            
            # Update end index
            current_end += motion_length
            end_indices.append(current_end - 1)  # 0-indexed
        
        # Concatenate all motion features
        mocap_data = torch.cat(all_features, dim=0)
        frame_size = mocap_data.shape[1]
        
        return mocap_data, end_indices, frame_size
    
    def _extract_single_motion_features(self, motion_id: int, motion_length: int) -> torch.Tensor:
        """
        Extract features for a single motion sequence in MVQ format.
        Based on motion_lib_mvq.py reference implementation.
        """
        # Create time points for the motion
        motion_times = torch.linspace(0, self.motion_lib.get_motion_length(motion_id), motion_length, device=self.device)
        motion_ids = torch.full((motion_length,), motion_id, device=self.device, dtype=torch.long)
        
        # Get motion frame data
        if hasattr(self.motion_lib, 'calc_motion_frame_rt'):
            # Use real-time version if available
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self.motion_lib.calc_motion_frame_rt(
                motion_ids, motion_times, 1.0/30.0  # 30 FPS
            )
        else:
            # Use regular version
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel = self.motion_lib.calc_motion_frame(
                motion_ids, motion_times
            )
        
        # Get SMPL joint positions if available (preferred over DOF positions)
        if hasattr(self.motion_lib, 'get_smpl_joint_desired'):
            smpl_joints = self.motion_lib.get_smpl_joint_desired()
            if smpl_joints is not None:
                # Use SMPL joint positions instead of DOF positions
                dof_pos = smpl_joints.reshape(motion_length, -1)  # Reshape to [T, 72] for 24 joints * 3
        
        # Convert to MVQ format using consecutive frames
        mvq_features = self._convert_to_mvq_format_consecutive(
            motion_ids, motion_times, root_pos, root_rot, dof_pos
        )
        
        return mvq_features
    
    def _extract_single_motion_features_pkl(self, motion_data) -> torch.Tensor:
        """
        Extract features for a single motion from PKL data directly.
        """
        # Extract motion data from PKL structure
        root_pos = torch.tensor(motion_data["root_trans_offset"], dtype=torch.float32, device=self.device)
        root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float32, device=self.device)
        dof_pos = torch.tensor(motion_data["dof"], dtype=torch.float32, device=self.device)
        
        # Get motion length
        motion_length = root_pos.shape[0]
        
        # Create motion_ids and motion_times for consistency
        motion_ids = torch.zeros(motion_length, dtype=torch.long, device=self.device)
        motion_times = torch.linspace(0, motion_length / 30.0, motion_length, device=self.device)  # 30 FPS
        
        # Convert to MVQ format using consecutive frames
        mvq_features = self._convert_to_mvq_format_consecutive(
            motion_ids, motion_times, root_pos, root_rot, dof_pos
        )
        
        return mvq_features
    
    def _convert_to_mvq_format_consecutive(self, motion_ids, motion_times, root_pos, root_rot, dof_pos):
        """
        Convert motion data to MVQ format using consecutive frames for proper delta/velocity computation.
        Based on motion_lib_mvq.py reference implementation.
        """
        # Get dimensions
        B = motion_ids.shape[0]
        num_joints = 24  # SMPL has 24 joints
        
        # Create MVQ frame tensor (364 dimensions for SMPL)
        mvq_frames = torch.zeros(B, 364, dtype=torch.float32, device=self.device)
        
        # Process consecutive frames for proper delta and velocity computation
        for i in range(B):
            if i == 0:
                # First frame - use zeros for deltas and velocities
                mvq_frames[i] = self._convert_single_frame_to_mvq(
                    root_pos[i], root_rot[i], dof_pos[i], 
                    root_pos[i], root_rot[i], dof_pos[i],  # Same frame for first
                    is_first_frame=True
                )
            else:
                # Consecutive frames - compute deltas and velocities
                mvq_frames[i] = self._convert_single_frame_to_mvq(
                    root_pos[i-1], root_rot[i-1], dof_pos[i-1],  # Previous frame
                    root_pos[i], root_rot[i], dof_pos[i],        # Current frame
                    is_first_frame=False
                )
        
        return mvq_frames
    
    def _convert_single_frame_to_mvq(self, root_pos_t, root_rot_t, dof_pos_t, 
                                   root_pos_tp1, root_rot_tp1, dof_pos_tp1, 
                                   is_first_frame=False):
        """
        Convert a single frame pair (t, t+1) to MVQ format.
        """
        num_joints = 24
        mvq_frame = torch.zeros(364, dtype=torch.float32, device=self.device)
        
        if not is_first_frame:
            # 1. Root deltas (dx, dy, dz, dyaw)
            delta_pos = root_pos_tp1 - root_pos_t
            delta_yaw = self._compute_yaw_delta(root_rot_t, root_rot_tp1)
            
            mvq_frame[0] = delta_pos[0]  # dx
            mvq_frame[1] = delta_pos[1]  # dy  
            mvq_frame[2] = delta_pos[2]  # dz
            mvq_frame[3] = delta_yaw     # dyaw
        
        # 2. Joint positions (use DOF positions as joint positions)
        # Note: In PKL data, dof_pos might be SMPL joint positions if available
        if dof_pos_t.shape[0] >= num_joints * 3:
            joint_pos = dof_pos_t[:num_joints * 3]
            mvq_frame[4:4+num_joints*3] = joint_pos
        else:
            # Pad with zeros if not enough dimensions
            available_dims = min(dof_pos_t.shape[0], num_joints * 3)
            mvq_frame[4:4+available_dims] = dof_pos_t[:available_dims]
        
        if not is_first_frame:
            # 3. Joint velocities (compute from consecutive frames)
            if dof_pos_tp1.shape[0] >= num_joints * 3 and dof_pos_t.shape[0] >= num_joints * 3:
                joint_vel = (dof_pos_tp1[:num_joints * 3] - dof_pos_t[:num_joints * 3]) / (1.0/30.0)  # 30 FPS
                mvq_frame[4+num_joints*3:4+num_joints*6] = joint_vel
            else:
                # Use available dimensions
                available_dims = min(dof_pos_tp1.shape[0], dof_pos_t.shape[0], num_joints * 3)
                joint_vel = (dof_pos_tp1[:available_dims] - dof_pos_t[:available_dims]) / (1.0/30.0)
                mvq_frame[4+num_joints*3:4+num_joints*3+available_dims] = joint_vel
        
        # 4. Joint orientations (convert quaternion to 9-D rotation matrix)
        if root_rot_t.shape[0] >= 4:  # Quaternion
            # Convert quaternion to rotation matrix (simplified)
            rot_9d = self._quat_to_rot_matrix_9d_single(root_rot_t)
            mvq_frame[4+num_joints*6:4+num_joints*6+rot_9d.shape[0]] = rot_9d
        
        return mvq_frame
    
    def _compute_yaw_delta(self, quat_t, quat_tp1):
        """
        Compute yaw delta between two quaternions.
        Based on the PKL structure where quaternions are [x, y, z, w] format.
        """
        # Convert to scipy Rotation objects for proper quaternion operations
        from scipy.spatial.transform import Rotation as R
        
        # Ensure quaternions are in [x, y, z, w] format (as per PKL structure)
        if quat_t.shape[0] >= 4 and quat_tp1.shape[0] >= 4:
            # Convert to numpy for scipy
            quat_t_np = quat_t.cpu().numpy() if quat_t.is_cuda else quat_t.numpy()
            quat_tp1_np = quat_tp1.cpu().numpy() if quat_tp1.is_cuda else quat_tp1.numpy()
            
            # Create Rotation objects
            rot_t = R.from_quat(quat_t_np[:4])  # [x, y, z, w]
            rot_tp1 = R.from_quat(quat_tp1_np[:4])
            
            # Compute relative rotation
            delta_rot = rot_tp1 * rot_t.inv()
            
            # Extract yaw (rotation around Z-axis)
            euler = delta_rot.as_euler('xyz', degrees=False)
            yaw_delta = euler[2]  # Z-axis rotation (yaw)
            
            # Normalize to [-π, π]
            yaw_delta = ((yaw_delta + np.pi) % (2 * np.pi)) - np.pi
            
            return float(yaw_delta)
        else:
            return 0.0
    
    def _quat_to_rot_matrix_9d_single(self, quat):
        """
        Convert single quaternion to 9-D rotation matrix representation.
        Based on PKL structure where quaternions are [x, y, z, w] format.
        """
        if quat.shape[0] >= 4:
            # Convert to numpy for scipy
            quat_np = quat.cpu().numpy() if quat.is_cuda else quat.numpy()
            
            # Create Rotation object from quaternion [x, y, z, w]
            from scipy.spatial.transform import Rotation as R
            rot = R.from_quat(quat_np[:4])
            
            # Convert to rotation matrix
            rot_matrix = rot.as_matrix()  # [3, 3]
            
            # Flatten to 9-D vector
            rot_9d = torch.from_numpy(rot_matrix.flatten()).to(self.device)
            
            return rot_9d
        else:
            # Fallback to identity matrix
            return torch.eye(3, device=self.device).flatten()
    
    def _convert_to_mvq_format(self, motion_ids, motion_times, root_pos, root_rot, dof_pos):
        """
        Convert motion data to MVQ format based on motion_lib_mvq.py reference.
        This is a simplified implementation that produces the correct frame structure.
        """
        # Get dimensions
        B = motion_ids.shape[0]
        num_joints = 24  # SMPL has 24 joints
        
        # Create MVQ frame tensor (364 dimensions for SMPL)
        mvq_frames = torch.zeros(B, 364, dtype=torch.float32, device=self.device)
        
        # MVQ Frame Structure (based on motion_lib_mvq.py):
        # 0-3: Global root velocity (dx, dy, dz, dyaw)
        # 4-(4+3N-1): Local joint positions (N=24 joints) -> 4-75
        # 4+3N-(4+6N-1): Local joint velocities -> 76-147  
        # 4+6N-(4+15N-1): Joint orientations (9-D per joint) -> 148-363
        
        # For now, we'll create a simplified version that matches the structure
        # TODO: Implement proper consecutive frame processing and rotation transformations
        
        # 1. Root deltas (simplified - use zeros for now)
        # mvq_frames[:, 0:4] = 0  # Already initialized to zero
        
        # 2. Joint positions (use DOF positions as joint positions)
        if dof_pos.shape[1] >= num_joints * 3:
            # Reshape DOF positions to joint positions
            joint_pos = dof_pos[:, :num_joints * 3]  # [B, 72]
            mvq_frames[:, 4:4+num_joints*3] = joint_pos
        else:
            # Pad with zeros if not enough dimensions
            available_dims = min(dof_pos.shape[1], num_joints * 3)
            mvq_frames[:, 4:4+available_dims] = dof_pos[:, :available_dims]
        
        # 3. Joint velocities (simplified - use zeros for now)
        # mvq_frames[:, 4+num_joints*3:4+num_joints*6] = 0  # Already initialized to zero
        
        # 4. Joint orientations (simplified - use root rotation repeated)
        if root_rot.shape[1] >= 4:  # Quaternion
            # Convert quaternion to rotation matrix (simplified)
            # For now, just repeat the root rotation for all joints
            root_rot_expanded = root_rot.unsqueeze(1).expand(-1, num_joints, -1)  # [B, 24, 4]
            # Convert to 9-D rotation matrix (simplified)
            rot_9d = self._quat_to_rot_matrix_9d(root_rot_expanded)  # [B, 24, 9]
            rot_9d_flat = rot_9d.reshape(B, -1)  # [B, 216]
            mvq_frames[:, 4+num_joints*6:4+num_joints*6+rot_9d_flat.shape[1]] = rot_9d_flat
        
        return mvq_frames
    
    def _quat_to_rot_matrix_9d(self, quat):
        """
        Convert quaternion to 9-D rotation matrix representation (simplified).
        """
        B, num_joints, _ = quat.shape
        
        # Simplified conversion - create identity-like matrices
        # In a real implementation, this would use proper quaternion to rotation matrix conversion
        rot_matrices = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, num_joints, -1, -1)
        rot_9d = rot_matrices.reshape(B, num_joints, 9)
        
        return rot_9d


def create_motion_data_adapter(config: Dict[str, Any]) -> MotionDataAdapter:
    """
    Factory function to create a motion data adapter.
    """
    return MotionDataAdapter(config)
