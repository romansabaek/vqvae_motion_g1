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
from .torch_utils import quat_diff, quat_to_exp_map, slerp, normalize_angle

logger = logging.getLogger(__name__)

class MotionLib_G1():
    
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
    
    def FRAME_SIZE(self):
        return self.TOTAL_FRAME_SIZE

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Motion data storage
        self.mocap_data = None
        self.end_indices = None
        self.frame_size = None

        self._mvq_cache = {}
        

    def get_motion_state(
            self, motion_ids, motion_times, joint_3d_format="exp_map"
    ):
        motion_len = self.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds

        num_frames = self.motion_num_frames[motion_ids]
        dt = self.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel0 = self.grvs[f0l]
        root_vel1 = self.grvs[f1l]

        root_ang_vel0 = self.gravs[f0l]
        root_ang_vel1 = self.gravs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        key_body_pos0 = self.gts[f0l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]
        key_body_pos1 = self.gts[f1l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        rigid_body_pos0 = self.gts[f0l]
        rigid_body_pos1 = self.gts[f1l]

        rigid_body_rot0 = self.grs[f0l]
        rigid_body_rot1 = self.grs[f1l]

        vals = [
            root_pos0,
            root_pos1,
            local_rot0,
            local_rot1,
            root_vel0,
            root_vel1,
            root_ang_vel0,
            root_ang_vel1,
            global_vel0,
            global_vel1,
            global_ang_vel0,
            global_ang_vel1,
            dof_vel0,
            dof_vel1,
            key_body_pos0,
            key_body_pos1,
            rigid_body_pos0,
            rigid_body_pos1,
            rigid_body_rot0,
            rigid_body_rot1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos: Tensor = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust

        root_rot: Tensor = slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_body_pos = (1.0 - blend_exp) * key_body_pos0 + blend_exp * key_body_pos1
        key_body_pos[:, :, 2] += self.ref_height_adjust

        # Always compute local joint rotations by slerping, for return/debug consistency
        local_rot = slerp(
            local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
        )

        # DOF positions follow MotionLib behavior: interpolate joint configuration
        if hasattr(self, "dof_pos"):  # H1 G1 joints
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:
            dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)

        # Match reference_code/motion_lib.py: use velocities at frame_idx0 (no blending)
        root_vel = root_vel0
        root_ang_vel = root_ang_vel0
        dof_vel = dof_vel0
        rigid_body_pos = (1.0 - blend_exp) * rigid_body_pos0 + blend_exp * rigid_body_pos1
        rigid_body_pos[:, :, 2] += self.ref_height_adjust
        rigid_body_rot = slerp(rigid_body_rot0, rigid_body_rot1, blend_exp)
        global_vel = global_vel0
        global_ang_vel = global_ang_vel0

        # Return only tensors (no RobotState), sufficient for downstream usage
        return (
            root_pos,
            root_rot,
            dof_pos,
            dof_vel,
            root_vel,
            root_ang_vel,
            local_rot,
        )

    def _calc_frame_blend(self, motion_times, motion_len, num_frames, dt):
        """
        Compute two neighboring frame indices and the interpolation blend factor.
        Mirrors reference_code/motion_lib.py but operates on provided per-motion scalars.
        Args:
            motion_times: (B,) tensor of times in seconds
            motion_len:   (B,) tensor of total motion lengths in seconds
            num_frames:   (B,) tensor of frame counts per motion
            dt:           (B,) tensor of frame time deltas (unused here but kept for parity)
        Returns:
            frame_idx0, frame_idx1, blend
        """
        # phase in [0, 1]
        phase = motion_times / motion_len
        phase = torch.clip(phase, 0.0, 1.0)

        # integer base frame and next frame within the motion
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)

        # fractional blend between 0 and 1
        blend = phase * (num_frames - 1) - frame_idx0.float()
        return frame_idx0, frame_idx1, blend


    def get_mvq_batch(self, motion_ids: torch.Tensor, motion_times: torch.Tensor):
        """
        Build a batch of VQVAE frames for the requested motion/time pairs **without**
        looking at the previous frame. All information required is contained in the
        current `RobotState` returned by :py:meth:`get_motion_state`.

        Args:
            motion_ids: (B,) tensor with motion indices
            motion_times: (B,) tensor with times (seconds)

        Returns
        -------
        mvq_frames : torch.Tensor
            (B, FRAME_SIZE) tensor with the new VQVAE frame representation.
        root_rot : torch.Tensor
            (B,4) quaternion (w-last) of the root at the sampled time – returned
            for any downstream usage (unchanged from the previous API).
        """
        root_pos_curr, root_rot_curr, dof_pos_curr, _, _, _, curr_local_rot = self.get_motion_state(motion_ids, motion_times)

        # Previous time step to compute finite-difference deltas
        prev_times = torch.clamp(motion_times - (1.0 / 30.0), min=0.0)  # Assume 30 FPS
        root_pos_prev, root_rot_prev, dof_pos_prev, _, _, _, prev_local_rot = self.get_motion_state(motion_ids, prev_times)

        # For retargeted G1, we only have root pose + DOF positions/velocities
        dt = torch.tensor(1/30, dtype=torch.float32, device=self.device)
        mvq_frames = self._convert_robot_states_to_mvq(
            motion_ids,
            motion_times,
            root_pos_prev,
            root_rot_prev,
            root_pos_curr,
            root_rot_curr,
            dof_pos_prev,
            dof_pos_curr,
            dt,
        )

        return mvq_frames
    
    def _convert_robot_states_to_mvq(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        root_pos_t: torch.Tensor,           # [B,3]
        root_rot_t: torch.Tensor,           # [B,4] (xyzw)
        root_pos_tp1: torch.Tensor,         # [B,3]
        root_rot_tp1: torch.Tensor,         # [B,4]
        dof_pos_t: torch.Tensor,            # [B,NUM_DOF]
        dof_pos_tp1: torch.Tensor,          # [B,NUM_DOF]
        dt: torch.Tensor,                   # scalar tensor seconds
    ) -> torch.Tensor:
        """
        Convert consecutive frames to MVQ format for G1-retargeted data.
        We only use root planar delta + yaw delta and DOF pos/vel.
        Output layout (TOTAL_FRAME_SIZE = 50):
          [dx_local, dy_local, dz_local, d_yaw, 23 dof_pos, 23 dof_vel]
        """
        B = motion_ids.shape[0]

        is_first_frame = (motion_times <= 1e-6)
        non_first_mask = ~is_first_frame

        mvq_frames = torch.zeros(B, self.FRAME_SIZE, dtype=torch.float32, device=self.device)

        # Heading-aligned root delta in local frame
        # Compute heading yaw from root quaternion: project onto yaw
        # Assume root_rot is xyzw here; our utils are xyzw-based
        # heading quaternion (rotate world into heading frame)
        # Extract yaw from quats via atan2-like approach on forward vector
        # Simpler: use z-rotation from quaternion by converting to exp-map around z approximated by axis

        # Use finite-difference in world frame then rotate into heading frame at time t
        delta_world = root_pos_tp1 - root_pos_t                                # [B,3]
        # Construct heading rotation from root_rot_t: only yaw around z
        # Compute yaw from quaternion
        x, y, z, w = root_rot_t.unbind(-1)
        # yaw from quaternion assuming xyzw
        yaw_t = torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
        x2, y2, z2, w2 = root_rot_tp1.unbind(-1)
        yaw_tp1 = torch.atan2(2.0*(w2*z2 + x2*y2), 1.0 - 2.0*(y2*y2 + z2*z2))
        d_yaw = normalize_angle(yaw_tp1 - yaw_t)

        # Rotate delta into local heading frame at t
        cos_y = torch.cos(-yaw_t)
        sin_y = torch.sin(-yaw_t)
        dx = cos_y * delta_world[:, 0] - sin_y * delta_world[:, 1]
        dy = sin_y * delta_world[:, 0] + cos_y * delta_world[:, 1]
        dz = delta_world[:, 2]

        mvq_frames[non_first_mask, 0] = dx[non_first_mask]
        mvq_frames[non_first_mask, 1] = dy[non_first_mask]
        mvq_frames[non_first_mask, 2] = dz[non_first_mask]
        mvq_frames[non_first_mask, 3] = d_yaw[non_first_mask]

        # DOF pos (current frame) and DOF vel (FD)
        mvq_frames[:, 4:4+self.NUM_DOF] = dof_pos_tp1
        dof_vel = (dof_pos_tp1 - dof_pos_t) / dt
        mvq_frames[non_first_mask, 4+self.NUM_DOF:4+self.NUM_DOF*2] = dof_vel[non_first_mask]

        return mvq_frames

    ## for MVQ
    def get_mvq_data(self) -> torch.Tensor:
        """
        Get all motion data converted to Motion VAE format for training.
        This matches the mocap_data format expected by EA's train_mvae.py
        
        Returns:
            Tensor of shape [total_frames, frame_size] containing all motion data
        """
        if hasattr(self, '_cached_mvae_data') and self._cached_mvae_data is not None:
            return self._cached_mvae_data
            
        log.info("Converting all motion data to MVQ format...")
        
        # Get all motion data
        all_mvq_frames = []

        for motion_id in range(self.num_motions()):
            motion_length = self.get_motion_length(torch.tensor([motion_id], device=self.device))
            num_frames = int(motion_length * 30.0)  # Assume 30 FPS
            
            # Generate time samples for this motion
            times = torch.linspace(0, motion_length[0] - (1.0/30.0), num_frames, device=self.device)
            motion_ids = torch.full((num_frames,), motion_id, device=self.device)
            
            # Convert to MVAE format
            mvq_frames = self.get_mvq_batch(motion_ids, times)

            all_mvq_frames.append(mvq_frames)
        # Concatenate all motion data
        self._cached_mvq_data = torch.cat(all_mvq_frames, dim=0)
        log.info(f"Converted {self._cached_mvq_data.shape[0]} frames to MVQ format")
        
        return self._cached_mvq_data
    
    def get_mvq_end_indices(self) -> np.ndarray:
        """
        Get the end indices of each motion sequence in the MVQ data.
        This is used to avoid sampling bad indices during training.
        
        Returns:
            numpy array of end indices for each motion
        """
        if hasattr(self, '_cached_end_indices') and self._cached_end_indices is not None:
            return self._cached_end_indices
            
        end_indices = []
        current_idx = 0
        
        for motion_id in range(self.num_motions()):
            motion_length = self.get_motion_length(torch.tensor([motion_id], device=self.device))
            num_frames = int(motion_length * 30.0)  # Assume 30 FPS
            current_idx += num_frames
            end_indices.append(current_idx - 1)  # Store last valid index
        
        self._cached_end_indices = np.array(end_indices)
        return self._cached_end_indices
    
    def clear_cache(self):
        """Clear cached data to save memory or force recomputation."""
        if hasattr(self, '_cached_mvq_data'):
            del self._cached_mvq_data
            self._cached_mvq_data = None
        if hasattr(self, '_cached_end_indices'):
            del self._cached_end_indices
            self._cached_end_indices = None



    # ======================================================
    # for playing
    # ======================================================

    def mvq_frame_step(
        self,
        pose_vec: torch.Tensor,          # [B,364] – one MVQ frame
        prev_root_pos: torch.Tensor,     # [B,3]
        prev_heading: torch.Tensor,      # [B]   – yaw angle at t-1
    ):
        """
        Reconstruct root & joint rotations from the 291-D MVQ pose
        and pack them into a RobotState so that `Simulator.set_state()`
        can display the result.

        `prev_heading` lets us integrate Δyaw; return value is the
        *new* heading so the caller can carry it to the next frame.
        """
        B = pose_vec.shape[0]
        num_joints = self.num_joints
        # ---------- split vector ----------
        root_dxdypsi   = pose_vec[:, 0:4]          # [B,4]
        jp_local       = pose_vec[:, 4:4+(num_joints*4)].reshape(B, num_joints, 3)
        jv_local       = pose_vec[:, 4+(num_joints*3):4+(self.num_joints*6)]
        rot9d          = pose_vec[:, 4+(self.num_joints*6):]         # [B,24*9]

        # ---------- 9-D → quaternion ----------
        rot_mats_full = rot9d.reshape(B, num_joints, 3, 3) # when using full cols # smpl
        
        local_quat = rotations.matrix_to_quaternion(
            rot_mats_full, w_last=True
        )                                                    # [B,24,4]
        local_quat = F.normalize(local_quat, dim=-1)

        dxy_local = root_dxdypsi[:, :3]
        new_heading = prev_heading + root_dxdypsi[:, 3]

        root_rot = local_quat[:, 0].clone() # [B,4]
        delta_heading = prev_heading - torch_utils.calc_heading(root_rot, w_last=True)
        q_correction = rotations.heading_to_quat(delta_heading, w_last=True)
        root_rot_aligned = rotations.quat_mul(q_correction, root_rot, w_last=True)

        dxy_world   = rotations.quat_apply(root_rot_aligned, dxy_local, w_last=True)
        new_root_pos = prev_root_pos.clone() + dxy_world

        # ---------- FK to global joint positions (optional) ----------
        jp_world = rotations.quat_apply(
            root_rot_aligned.unsqueeze(1), jp_local, w_last=True
        ) + new_root_pos.unsqueeze(1)

        # ---------- Convert to DOF angles ----------
        dof_pos = self._local_rotation_to_dof(local_quat, joint_3d_format="exp_map")
        
        # print("===============================")
        # Return plain tensors (no RobotState):
        # new_root_pos, new_heading, root_rot_aligned, dof_pos, local_quat
        return new_root_pos.detach(), new_heading.detach(), root_rot_aligned.detach(), dof_pos.detach(), local_quat.detach()