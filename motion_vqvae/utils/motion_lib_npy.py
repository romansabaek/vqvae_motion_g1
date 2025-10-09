import os
import logging

import torch
import numpy as np
from tqdm import tqdm

from .torch_utils import quat_diff, quat_to_exp_map, slerp

logger = logging.getLogger(__name__)

def smooth(x, box_pts, device):
    """Applies a 1D moving average filter to smooth the data."""
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T


def ensure_quaternion_continuity(quats: torch.Tensor) -> torch.Tensor:
    """
    Enforces continuity on a sequence of quaternions by flipping signs where needed.
    Expects and returns XYZW format.
    """
    continuous_quats = quats.clone()
    for i in range(1, continuous_quats.shape[0]):
        # Dot product between current and previous quaternion
        dot_product = torch.dot(continuous_quats[i], continuous_quats[i-1])
        # If the dot product is negative, the quaternions are on opposite hemispheres.
        # Flipping the current quaternion will result in a shorter rotational path.
        # if dot_product < 0.0:
            # continuous_quats[i] *= -1.0
    return continuous_quats



class MotionLibNpy:
    def __init__(self, motion_file, device, motion_id=None):
        self._device = device
        # motion_id is used to select which clip from the sequence to treat as the primary one
        self._motion_id_to_load = motion_id
        self._loaded_motion_name = None
        self._load_motions(motion_file)

    def _load_motions(self, motion_file):
        # Initialize lists to store data for each motion clip
        self._motion_names = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        
        self._motion_root_pos = []
        self._motion_root_rot = []
        self._motion_root_vel = []
        self._motion_root_ang_vel = []
        self._motion_dof_pos = []
        self._motion_dof_vel = []

  
        self._motion_transition_flags = [] # Stores the per-frame transition flag
        self._motion_stable_lengths = []   # Stores the duration of the stable part of the motion
        self._motion_transition_lengths = [] # Stores the duration of the transition part
      
        # 1. Load the continuous trajectory data from the .npy file
        trajectory_data = np.load(motion_file)
        
        # 2. Split the trajectory into individual motion clips based on the transition flag
        transition_flags = trajectory_data[:, -1]
        split_indices = [0] + list(np.where((transition_flags[:-1] > 0.5) & (transition_flags[1:] < 0.5))[0] + 1)

        motion_clips = []
        for i in range(len(split_indices)):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1] if i + 1 < len(split_indices) else len(trajectory_data)
            clip_data = trajectory_data[start_idx:end_idx]
            
            motion_clips.append(clip_data)
            
        # If a specific motion_id is requested, only process that one
        if self._motion_id_to_load is not None:
            if not (0 <= self._motion_id_to_load < len(motion_clips)):
                raise IndexError(f"motion_id {self._motion_id_to_load} is out of bounds for the {len(motion_clips)} clips in the file.")
            
            motions_to_process = {f"motion_{self._motion_id_to_load}": motion_clips[self._motion_id_to_load]}
            self._loaded_motion_name = f"motion_{self._motion_id_to_load}"
        else:
            motions_to_process = {f"motion_{i}": clip for i, clip in enumerate(motion_clips)}

        # 3. Process each segmented motion clip
        for name, motion_data_raw in tqdm(motions_to_process.items(), desc="[MotionLib] Processing motions"):
            try:
                self._motion_names.append(name)
                
                time_stamps = motion_data_raw[:, 0]
                dt = np.mean(np.diff(time_stamps)) if len(time_stamps) > 1 else 1.0 / 30.0
                fps = 1.0 / dt
                # dt = 1.0 / fps

                root_pos = torch.tensor(motion_data_raw[:, 1:4], dtype=torch.float, device=self._device)
                dof_pos = torch.tensor(motion_data_raw[:, 8:-1], dtype=torch.float, device=self._device)
                
                root_rot_wxyz = torch.tensor(motion_data_raw[:, 4:8], dtype=torch.float, device=self._device)
                root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
                root_rot = ensure_quaternion_continuity(root_rot_xyzw)

                num_frames = root_pos.shape[0]
                curr_len = dt * (num_frames - 1)

                transition_flags_clip = torch.tensor(motion_data_raw[:, -1], dtype=torch.float, device=self._device)
                
                num_stable_frames = torch.sum(transition_flags_clip < 0.5).item()
                num_trans_frames = num_frames - num_stable_frames

                stable_len = dt * (num_stable_frames - 1) if num_stable_frames > 1 else 0.0
                trans_len = dt * num_trans_frames
                
                self._motion_transition_flags.append(transition_flags_clip)
                self._motion_stable_lengths.append(stable_len)
                self._motion_transition_lengths.append(trans_len)

                # Calculate and smooth velocities
                root_vel = torch.zeros_like(root_pos) 
                root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :]) 
                root_vel[-1, :] = root_vel[-2, :] 
                root_vel = smooth(root_vel, 19, device=self._device)

                root_ang_vel = torch.zeros_like(root_pos) 
                root_drot = quat_diff(root_rot[:-1], root_rot[1:]) 
                root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot) 
                root_ang_vel[-1, :] = root_ang_vel[-2, :]
                root_ang_vel = smooth(root_ang_vel, 19, device=self._device)

                dof_vel = torch.zeros_like(dof_pos); 
                dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :]); 
                dof_vel[-1, :] = dof_vel[-2, :]; 
                dof_vel = smooth(dof_vel, 19, device=self._device)
                
                self._motion_weights.append(1.0)
                self._motion_fps.append(fps)
                self._motion_dt.append(dt)
                self._motion_num_frames.append(num_frames)
                self._motion_lengths.append(curr_len) # This is now the total length (stable + transition)
                
                self._motion_root_pos.append(root_pos); 
                self._motion_root_rot.append(root_rot); 
                self._motion_root_vel.append(root_vel); 
                self._motion_root_ang_vel.append(root_ang_vel); 
                self._motion_dof_pos.append(dof_pos); 
                self._motion_dof_vel.append(dof_vel)

            except Exception as e:
                logger.error(f"Error processing motion '{name}': {e}")
                continue
        
        # 4. Finalize by concatenating all data into large tensors
        if not self._motion_root_pos:
            raise ValueError("No valid motions were loaded from the trajectory file.")
            
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float, device=self._device); self._motion_weights /= torch.sum(self._motion_weights)
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float, device=self._device); self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float, device=self._device)
        
        self._motion_stable_lengths = torch.tensor(self._motion_stable_lengths, dtype=torch.float, device=self._device)
        self._motion_transition_lengths = torch.tensor(self._motion_transition_lengths, dtype=torch.float, device=self._device)
        self._motion_transition_flags = torch.cat(self._motion_transition_flags, dim=0)

        self._motion_root_pos = torch.cat(self._motion_root_pos, dim=0); 
        self._motion_root_rot = torch.cat(self._motion_root_rot, dim=0)
        self._motion_root_vel = torch.cat(self._motion_root_vel, dim=0); 
        self._motion_root_ang_vel = torch.cat(self._motion_root_ang_vel, dim=0)
        self._motion_dof_pos = torch.cat(self._motion_dof_pos, dim=0); 
        self._motion_dof_vel = torch.cat(self._motion_dof_vel, dim=0)

        lengths_shifted = self._motion_num_frames.roll(1); lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        self._motion_ids = torch.arange(self.num_motions(), dtype=torch.long, device=self._device)
        
        total_len = self.get_total_length()
        print(f"Loaded and segmented {self.num_motions()} motions with a total length of {total_len:.3f}s.")

    def get_stable_length(self, motion_ids):
        """Returns the duration of the stable part of the motion."""
        return self._motion_stable_lengths[motion_ids]
    
    def get_transition_length(self, motion_ids):
        """Returns the duration of the transition part of the motion."""
        return self._motion_transition_lengths[motion_ids]


    def get_motion_length(self, motion_ids):
        """Returns the total length (stable + transition) of the motion."""
        return self._motion_lengths[motion_ids]
        
    def num_motions(self):
        return self._motion_weights.shape[0]
    
    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()
    
    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]

        phase = times / self._motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx

        return frame_idx0, frame_idx1, blend
        
    def calc_motion_frame(self, motion_ids, motion_times):
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._motion_root_pos[frame_idx0]; root_pos1 = self._motion_root_pos[frame_idx1]
        root_rot0 = self._motion_root_rot[frame_idx0]; root_rot1 = self._motion_root_rot[frame_idx1]
        root_vel = self._motion_root_vel[frame_idx0]; root_ang_vel = self._motion_root_ang_vel[frame_idx0]
        dof_pos0 = self._motion_dof_pos[frame_idx0]; dof_pos1 = self._motion_dof_pos[frame_idx1]
        dof_vel = self._motion_dof_vel[frame_idx0]
        
        # --- START OF MODIFICATION: Sample transition flag ---
        trans_flag0 = self._motion_transition_flags[frame_idx0]
        trans_flag1 = self._motion_transition_flags[frame_idx1]
        # --- END OF MODIFICATION ---

        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        root_rot = slerp(root_rot0, root_rot1, blend)
        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        transition_flag = (1.0 - blend) * trans_flag0 + blend * trans_flag1
        
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, transition_flag   

    
    @torch.no_grad()
    def calc_motion_frame_rt(self, motion_ids, motion_times, fd_dt: float):
        """
        Strict RT: states at t, velocities from central diff at t ± fd_dt.
        motion_ids: (T,) long
        motion_times: (T,) float
        fd_dt: small positive float (e.g., control_dt or 1/fps)
        """
        # Clamp fd_dt to be <= half a period of the motion to avoid wrap artifacts
        fd_dt = float(fd_dt)

        # t, t-Δ, t+Δ  (handle loop internally, same as calc_motion_frame)
        t0 = motion_times
        tm = torch.clamp(t0 - fd_dt, min=0.0)             # safe clamp before wrapping
        tp = t0 + fd_dt

        # States at t
        root_pos_t, root_rot_t, _, _, dof_pos_t, _, transition_flag_t  = self.calc_motion_frame(motion_ids, t0)

        # States at t - Δ and t + Δ
        root_pos_m, root_rot_m, _, _, dof_pos_m, _, transition_flag_t = self.calc_motion_frame(motion_ids, tm)
        root_pos_p, root_rot_p, _, _, dof_pos_p, _, transition_flag_t = self.calc_motion_frame(motion_ids, tp)

        # Linear velocity & joint velocity by central difference
        root_vel_t = (root_pos_p - root_pos_m) / (2.0 * fd_dt)
        dof_vel_t  = (dof_pos_p - dof_pos_m) / (2.0 * fd_dt)

        # Angular velocity via quaternion logarithmic map
        # Requires quat_diff(q1, q2) = q1^{-1} * q2   and quat_to_exp_map(q) -> rotation vector
        
        dq = quat_diff(root_rot_m, root_rot_p)                    # rotation from t-Δ to t+Δ
        rotvec = quat_to_exp_map(dq)                              # axis-angle vector (rad)
        root_ang_vel_t = rotvec / (2.0 * fd_dt)                   # rad/s around t

        return root_pos_t, root_rot_t, root_vel_t, root_ang_vel_t, dof_pos_t, dof_vel_t, transition_flag_t


    def get_smpl_joint_desired(self):
        return None

    def get_motion_name(self):
        return self._loaded_motion_name