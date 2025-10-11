import os, pickle
import logging

import torch

from .torch_utils import quat_diff, quat_to_exp_map, slerp
from tqdm import tqdm
logger = logging.getLogger(__name__)


import joblib
import numpy as np
from deploy_common.utils.torch_utils import quat_diff, quat_to_exp_map

def smooth(x, box_pts, device):
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


class MotionLib:    
    def __init__(self, motion_file, device, motion_id=None):
        self._device = device
        self._motion_id_to_load = motion_id 
        self._loaded_motion_name = None 
        self._load_motions(motion_file)
        
    def _load_motions(self, motion_file):
        self._motion_names = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_files = []
        
        self._motion_root_pos = []
        self._motion_root_rot = []
        self._motion_root_vel = []
        self._motion_root_ang_vel = []
        self._motion_dof_pos = []
        self._motion_dof_vel = []
        self._smpl_joints = []

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        
        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            curr_file = motion_files[i]
            self._motion_names.append(os.path.basename(curr_file))
            try:
                
                if self._motion_id_to_load is not None:
                    # If an ID is specified, load the multi-motion file
                    # and select the specific motion.
                    data_list = joblib.load(curr_file)
                    motion_data_list = np.array(list(data_list.values()))
                    motion_data_keys = np.array(list(data_list.keys()))

                    if self._motion_id_to_load >= len(motion_data_list):
                        raise IndexError(f"motion_id {self._motion_id_to_load} is out of bounds for file with {len(motion_data_list)} motions.")

                    print(f"Selected motion ID: {self._motion_id_to_load} from file.")
                    self._loaded_motion_name = motion_data_keys[self._motion_id_to_load]
                    
                    motion_data_raw = motion_data_list[self._motion_id_to_load]
                    # Reformat into the expected dictionary structure
                    motion_data = {
                        "fps": motion_data_raw.get("fps", 30), # Default fps if not present
                        "root_pos": motion_data_raw["root_trans_offset"],   
                        "root_rot": motion_data_raw["root_rot"],
                        "dof_pos": motion_data_raw["dof"],
                        "smpl_joints": motion_data_raw.get("smpl_joints") 
                    }
                    if motion_data["smpl_joints"] is None:
                        del motion_data["smpl_joints"]
                else:
                    # Original behavior: load a single-motion pickle file.
                    with open(curr_file, "rb") as f:
                        motion_data = pickle.load(f)

                # The rest of the processing is IDENTICAL to the original code.
                fps = motion_data["fps"]
                curr_weight = motion_weights[i]
                dt = 1.0 / fps
                
                root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float, device=self._device)
                root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float, device=self._device)
                dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float, device=self._device)
                
                if "smpl_joints" in motion_data: 
                    smpl_joints = torch.tensor(motion_data["smpl_joints"], dtype=torch.float, device=self._device)
                    self._smpl_joints.append(smpl_joints)

                num_frames = root_pos.shape[0]
                curr_len = dt * (num_frames - 1)
                
                root_vel = torch.zeros_like(root_pos)
                root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
                root_vel[-1, :] = root_vel[-2, :]
                root_vel = smooth(root_vel, 19, device=self._device)
                
                root_ang_vel = torch.zeros_like(root_pos)
                root_drot = quat_diff(root_rot[:-1], root_rot[1:])
                root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot)
                root_ang_vel[-1, :] = root_ang_vel[-2, :]
                root_ang_vel = smooth(root_ang_vel, 19, device=self._device)
                
                dof_vel = torch.zeros_like(dof_pos)
                dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                dof_vel[-1, :] = dof_vel[-2, :]
                dof_vel = smooth(dof_vel, 19, device=self._device)
                
                self._motion_weights.append(curr_weight)
                self._motion_fps.append(fps)
                self._motion_dt.append(dt)
                self._motion_num_frames.append(num_frames)
                self._motion_lengths.append(curr_len)
                self._motion_files.append(curr_file)
                
                self._motion_root_pos.append(root_pos)
                self._motion_root_rot.append(root_rot)
                self._motion_root_vel.append(root_vel)
                self._motion_root_ang_vel.append(root_ang_vel)
                self._motion_dof_pos.append(dof_pos)
                self._motion_dof_vel.append(dof_vel)

            except Exception as e:
                logger.error(f"Error loading motion file {curr_file}: {e}")
                continue
        
        
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights /= torch.sum(self._motion_weights)
        
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float, device=self._device)
        
        self._motion_root_pos = torch.cat(self._motion_root_pos, dim=0)
        self._motion_root_rot = torch.cat(self._motion_root_rot, dim=0)
        self._motion_root_vel = torch.cat(self._motion_root_vel, dim=0)
        self._motion_root_ang_vel = torch.cat(self._motion_root_ang_vel, dim=0)
        self._motion_dof_pos = torch.cat(self._motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.cat(self._motion_dof_vel, dim=0)
        if self._smpl_joints:
            self._smpl_joints = torch.cat(self._smpl_joints, dim=0)


        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        
        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))


    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
        
    def num_motions(self):
        return self._motion_weights.shape[0]
    
    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()
                
    def _fetch_motion_files(self, motion_file: str):
        motion_files = [motion_file]
        motion_weights = [1.0]
        
        return motion_files, motion_weights
    
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
        
        root_pos0 = self._motion_root_pos[frame_idx0]
        root_pos1 = self._motion_root_pos[frame_idx1]
        
        root_rot0 = self._motion_root_rot[frame_idx0]
        root_rot1 = self._motion_root_rot[frame_idx1]
        
        root_vel = self._motion_root_vel[frame_idx0]
        root_ang_vel = self._motion_root_ang_vel[frame_idx0]
        
        dof_pos0 = self._motion_dof_pos[frame_idx0]
        dof_pos1 = self._motion_dof_pos[frame_idx1]
        
        dof_vel = self._motion_dof_vel[frame_idx0]
        
        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        root_rot = slerp(root_rot0, root_rot1, blend)
        
        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
    
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
        root_pos_t, root_rot_t, _, _, dof_pos_t, _ = self.calc_motion_frame(motion_ids, t0)

        # States at t - Δ and t + Δ
        root_pos_m, root_rot_m, _, _, dof_pos_m, _ = self.calc_motion_frame(motion_ids, tm)
        root_pos_p, root_rot_p, _, _, dof_pos_p, _ = self.calc_motion_frame(motion_ids, tp)

        # Linear velocity & joint velocity by central difference
        root_vel_t = (root_pos_p - root_pos_m) / (2.0 * fd_dt)
        dof_vel_t  = (dof_pos_p - dof_pos_m) / (2.0 * fd_dt)

        # Angular velocity via quaternion logarithmic map
        # Requires quat_diff(q1, q2) = q1^{-1} * q2   and quat_to_exp_map(q) -> rotation vector
        
        dq = quat_diff(root_rot_m, root_rot_p)                    # rotation from t-Δ to t+Δ
        rotvec = quat_to_exp_map(dq)                              # axis-angle vector (rad)
        root_ang_vel_t = rotvec / (2.0 * fd_dt)                   # rad/s around t

        return root_pos_t, root_rot_t, root_vel_t, root_ang_vel_t, dof_pos_t, dof_vel_t



    def get_smpl_joint_desired(self):
        # This now correctly returns the stored data as a numpy array.
        if hasattr(self, '_smpl_joints') and self._smpl_joints is not None:
            return self._smpl_joints.cpu().numpy()
        return None


    def get_motion_name(self):
        return self._loaded_motion_name
