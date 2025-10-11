import copy
from typing import Any
from easydict import EasyDict

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from pathlib import Path

from protomotions.utils.motion_lib import MotionLib
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.simulator.base_simulator.config import RobotConfig

from protomotions.envs.mimic.mimic_utils import dof_to_local
from isaac_utils import torch_utils, rotations

import logging
log = logging.getLogger(__name__)

class MotionLibMVQ(MotionLib):
    
    # SMPL has exactly 24 joints - verify this matches the config
    @property
    def FRAME_SIZE(self):
        # New frame definition for Motion-VQVAE
        #   0-1-2   : global root velocity in x / y / z directions
	#   3   : global root angular velocity (wz) -> only wz is used
        #   4-(4+3N−1)                  : local joint positions  (N = num_joints)
        #   4+3N-(4+6N−1)               : local joint velocities
        #   4+6N-(4+15N−1)              : joint orientations (9-D per joint)
        if self.num_joints == 24:
            return 4 + (self.num_joints * 3) + (self.num_joints * 3) + (self.num_joints * 9)  # 4 + 72 + 72 + 216 = 364 for SMPL

        
    ## smpl
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mvq_cache = {}
        self.num_joints = len(self.robot_config.body_names)

        
    
    """
    ============================================================================
    ============================================================================
    """

    def get_motion_state(
            self, motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> RobotState:
        motion_len = self.state.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds

        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

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

        root_rot: Tensor = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_body_pos = (1.0 - blend_exp) * key_body_pos0 + blend_exp * key_body_pos1
        key_body_pos[:, :, 2] += self.ref_height_adjust

        if hasattr(self, "dof_pos"):  # H1 G1 joints
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
            # local_rot = torch_utils.slerp(
            #     local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
            # )
        else:
            local_rot = torch_utils.slerp(
                local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
            )
            dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)

        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        rigid_body_pos = (1.0 - blend_exp) * rigid_body_pos0 + blend_exp * rigid_body_pos1
        rigid_body_pos[:, :, 2] += self.ref_height_adjust
        rigid_body_rot = torch_utils.slerp(rigid_body_rot0, rigid_body_rot1, blend_exp)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1

        motion_state = RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            key_body_pos=key_body_pos,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=global_vel,
            rigid_body_ang_vel=global_ang_vel,
        )

        return motion_state, local_rot


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
        current_states, curr_local_rot = self.get_motion_state(motion_ids, motion_times)

        # Get previous frame states for velocity/delta computation
        prev_times = torch.clamp(motion_times - (1.0 / 30.0), min=0.0)  # Assume 30 FPS
        prev_states, prev_local_rot = self.get_motion_state(motion_ids, prev_times)

        mvq_frames = self._convert_robot_states_to_mvq(
            motion_ids, 
            motion_times,
            prev_states.root_pos, 
            prev_states.root_rot,
            current_states.root_pos, 
            current_states.root_rot, 
            prev_local_rot,
            curr_local_rot,
            prev_states.rigid_body_pos,
            current_states.rigid_body_pos,
            torch.tensor(1/30, dtype=torch.float32, device=self.device),
            w_last=True
        )

        return mvq_frames
    
    def _convert_robot_states_to_mvq(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        root_pos_t: torch.Tensor,           # [B,3]
        root_rot_t: torch.Tensor,           # [B,4]  w-last
        root_pos_tp1: torch.Tensor,         # [B,3]
        root_rot_tp1: torch.Tensor,         # [B,4]
        local_rot_t: torch.Tensor,          # [B,24,4]
        local_rot_tp1: torch.Tensor,        # [B,24,4]
        joint_pos_t: torch.Tensor,          # [B,24,3] global
        joint_pos_tp1: torch.Tensor,        # [B,24,3] global
        dt: torch.Tensor,                   # [B]  seconds
        w_last,
    ) -> torch.Tensor:
        """
        Convert a pair of consecutive frames (t, t+1) into a
        single 1-D vector compatible with the Motion-VAE paper.
        """
        # B = root_pos_t.shape[0]
        B = motion_ids.shape[0]

        is_first_frame = (motion_times <= 1e-6)
        non_first_mask = ~is_first_frame

        mvq_frames = torch.zeros(B, self.FRAME_SIZE, dtype=torch.float32, device=self.device)

        # ------------------------------------------------------------------
        # 1. Δ root (x, y, yaw)  – rotate displacement into heading frame
        # ------------------------------------------------------------------
        heading_q_t = torch_utils.calc_heading_quat(
            root_rot_t, w_last=w_last
        )                       # [B,4]
        heading_inv_t = rotations.quat_conjugate(heading_q_t, w_last=w_last)
        delta_xy_world = root_pos_tp1[:, :3] - root_pos_t[:, :3]       # [B,3]
        delta_xy_world = torch.cat([delta_xy_world, delta_xy_world.new_zeros(B, 1)], dim=-1)
        delta_xy_local = rotations.quat_apply(heading_inv_t, delta_xy_world, w_last=w_last)[..., :3]

        yaw_t   = torch_utils.calc_heading(root_rot_t,  w_last=w_last)   # [B]
        yaw_tp1 = torch_utils.calc_heading(root_rot_tp1, w_last=w_last)
        delta_yaw = torch_utils.normalize_angle(yaw_tp1 - yaw_t)       # wrap to [-π,π]

        # root_block = torch.stack([delta_xy_local[:,0], delta_xy_local[:,1], delta_yaw], dim=-1)  # [B,3]
        mvq_frames[non_first_mask, 0] = delta_xy_local[non_first_mask,0]
        mvq_frames[non_first_mask, 1] = delta_xy_local[non_first_mask,1]
	mvq_frames[non_first_mask, 2] = delta_xy_local[non_first_mask,2]
        mvq_frames[non_first_mask, 3] = delta_yaw[non_first_mask]

        # ------------------------------------------------------------------
        # 2. joint positions (root space)
        # ------------------------------------------------------------------
        joint_pos_rel = joint_pos_t - root_pos_t.unsqueeze(1)          # [B,24,3]
        heading_inv_joint_rep = heading_inv_t[:, None, :].expand(-1, len(self.robot_config.body_names), -1)
        joint_pos_local = rotations.quat_apply(heading_inv_joint_rep,
                                               joint_pos_rel,
                                               w_last=w_last)            # [B,24,3]
        jp_block = joint_pos_local.reshape(B, -1)                      # [B,72]
        mvq_frames[:, 4:4+(self.num_joints*3)] = jp_block

        # ------------------------------------------------------------------
        # 3. joint velocities (root space)
        # ------------------------------------------------------------------
        joint_pos_rel_tp1 = joint_pos_tp1 - root_pos_t.unsqueeze(1)    # note: use
        joint_pos_local_tp1 = rotations.quat_apply(heading_inv_joint_rep,
                                                   joint_pos_rel_tp1,
                                                   w_last=w_last)        # [B,24,3]
        joint_vel_local = (joint_pos_local_tp1 - joint_pos_local) / dt.view(-1,1,1)
        jv_block = joint_vel_local.reshape(B, -1)                      # [B,72]
        mvq_frames[non_first_mask, 4+(self.num_joints*3):4+(self.num_joints*6)] = jv_block[non_first_mask]

        # ------------------------------------------------------------------
        # 4. joint 9-D orientation
        # ------------------------------------------------------------------
        # Flatten 9-D columns for every joint at *t*
        rot_mats = rotations.quaternion_to_matrix(local_rot_t, w_last=w_last)  # [B,24,3,3]
        rot_6d = rot_mats.reshape(B, -1) #9
        mvq_frames[:, 4+(self.num_joints*6):] = rot_6d

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

    def mvq_frame_to_robot_state(
        self,
        pose_vec: torch.Tensor,          # [B,364] – one MVQ frame
        prev_root_pos: torch.Tensor,     # [B,3]
        prev_heading: torch.Tensor,      # [B]   – yaw angle at t-1
    ) -> Tuple[RobotState, torch.Tensor]:  # returns state, new_heading
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
        # ---------- Pack RobotState ----------
        state = RobotState(
            root_pos        = new_root_pos,
            root_rot        = root_rot_aligned,
            root_vel        = torch.zeros_like(new_root_pos),
            root_ang_vel    = torch.zeros_like(new_heading).unsqueeze(-1).repeat(1,3),
            key_body_pos    = torch.zeros_like(jp_world[:, self.key_body_ids]), #jp_world[:, self.key_body_ids],
            dof_pos         = dof_pos,
            dof_vel         = torch.zeros_like(dof_pos),
            rigid_body_pos  = torch.zeros_like(jp_world), #jp_world,
            rigid_body_rot  = torch.zeros((B, num_joints, 4), device=self.device), # local_quat
            rigid_body_vel  = torch.zeros_like(jp_world),
            rigid_body_ang_vel = torch.zeros_like(jp_world),
        )
        return state, new_root_pos.detach(), new_heading.detach(), local_quat.detach()
