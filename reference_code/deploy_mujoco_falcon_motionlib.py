# Modified script to run MuJoCo simulation using MotionLib for upper-body references.
# This version abstracts the command generation into a dedicated function for better modularity.

import sys
import os
import time
import yaml
import mujoco
import mujoco.viewer
import numpy as np
import torch
import argparse
import imageio
from mujoco import Renderer
from datetime import datetime
from typing import Dict

# Append deploy directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deploy_common.utils.indexing import sim_to_real_joint_values, real_to_sim_joint_values
from utils.buffer import ObsBuffer
from utils.observations import get_projected_gravity
from deploy_common.utils.motion_lib import MotionLib 
from scipy.spatial.transform import Rotation as R

# Helper function from GMT script to rotate vectors by the inverse of a quaternion
@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def is_standing(lin_vel_raw, threshold=0.05):
    """Returns True if motion is considered standing."""
    horizontal_speed = np.linalg.norm(lin_vel_raw[:2])
    return horizontal_speed < threshold

def get_falcon_motion_command(motion_lib: MotionLib, motion_id: int, time_step: float, control_dt: float, device: str) -> Dict[str, np.ndarray]:
    """
    Generates the reference command dictionary for the Falcon policy from MotionLib.
    """
    fd_dt = control_dt
    motion_time = torch.tensor([time_step], device=device)
    motion_id_tensor = torch.tensor([motion_id], device=device, dtype=torch.long)
    
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = motion_lib.calc_motion_frame_rt(motion_id_tensor, motion_time, fd_dt)
    
    # Convert to local frame velocities
    lin_vel_local = quat_rotate_inverse(root_rot, root_vel).cpu().numpy().squeeze()
    ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel).cpu().numpy().squeeze()

    # Extract command components
    height = root_pos.cpu().numpy().squeeze()[2]
    dof_pos_np = dof_pos.cpu().numpy().squeeze()
    
    walk_on = 0 if is_standing(lin_vel_local) else 1
    
    lin_vel_cmd = np.array([lin_vel_local[0], lin_vel_local[1]])
    ang_vel_cmd = np.array([ang_vel_local[2]])
    
    # Define DOF indices
    waist_dof_indices = [12, 13, 14]
    upper_dof_indices = [15, 16, 17, 18, 19, 20, 21, 22]
    
    waist_cmd = dof_pos_np[waist_dof_indices]
    
    first_half = upper_dof_indices[:4]
    second_half = upper_dof_indices[4:8]
    extended_upper_dof_pos = np.concatenate([
        dof_pos_np[first_half],
        np.zeros(3),
        dof_pos_np[second_half],
        np.zeros(3)
    ])
    
    # print("waist_dofs comd", waist_cmd)
    # Assemble and return the command dictionary
    command = {
        "ang_vel": ang_vel_cmd,
        "base_height": np.array([height]) * 2.0,
        "lin_vel": lin_vel_cmd,
        "stand": np.array([walk_on]),
        "waist_dofs": waist_cmd,
        "ref_upper_dof_pos": extended_upper_dof_pos,
    }
    return command


def compute_observation_falcon(config, omega, projected_gravity, qj, dqj, action, obs_buffer: ObsBuffer, command: dict):
    obs_history = obs_buffer._get_obs_history_old_first()
    obs_history_flat = obs_history[:, -115:].reshape(-1)
    obs_curr = np.concatenate([
        action, omega, command['ang_vel'], command['base_height'],
        command['lin_vel'], command['stand'], command['waist_dofs'],
        qj, dqj, projected_gravity, command['ref_upper_dof_pos']
    ])
    return np.concatenate([obs_history_flat, obs_curr]).squeeze()


def get_joint_dof_mapping(model):
    joint_to_dof = {}
    for joint_id in range(model.njnt):
        name = model.names[model.name_jntadr[joint_id]:].split(b'\x00', 1)[0]
        dof_start = model.jnt_dofadr[joint_id]
        dof_end = model.jnt_dofadr[joint_id + 1] if joint_id < model.njnt - 1 else model.nv
        joint_to_dof[name] = list(range(dof_start, dof_end))
    return joint_to_dof

def projected_gravity_wrt_body(q_wxyz):
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    R_wb = R.from_quat(q_xyzw).as_matrix()
    g_world = np.array([0.0, 0.0, -1.0], dtype=float)
    return R_wb.T @ g_world


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file name in the config folder")
    parser.add_argument("motion_path", type=str, help="path to the motion file (.pkl)")
    args = parser.parse_args()

    with open(f"configs/{args.config}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Unpack config
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps, kds = np.array(config["kps"], dtype=np.float32), np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    sim_dof, real_dof = config["sim_dof"], config["real_dof"]
    sim_to_real_idx = config["sim_to_real_idx"]
    real_to_sim_idx = config["real_to_sim_idx"]
    num_obs = config["num_obs"]
    
    # Initialize variables
    obs_sim = np.zeros(num_obs, dtype=np.float32)
    action_sim = np.zeros(sim_dof, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    counter = 0


    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy
    if '.onnx' in policy_path:
        import onnxruntime as ort
        policy_onnx = ort.InferenceSession(policy_path)
        input_name = policy_onnx.get_inputs()[0].name
        policy = lambda obs: torch.from_numpy(policy_onnx.run(None, {
            input_name: obs.detach().cpu().numpy().astype("float32")
        })[0])
    else:
        policy = torch.jit.load(policy_path)

    obs_buffer = ObsBuffer(config)

    motion_id_selected = 3670
    motion_lib_idx = 0   
    device = "cpu" # Use CPU for motion lib as torch usage is minimal
    motion_lib = MotionLib(motion_file=args.motion_path, device=device, motion_id=motion_id_selected)

    # Setup logging and video recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    video_path = f"logs/{timestamp}_recorded_mujoco.mp4"
    os.makedirs("logs", exist_ok=True)
    renderer = Renderer(m, width=640, height=480)
    video_writer = imageio.get_writer(video_path, fps=30, codec='libx264', quality=10)
    sim_fps = int(1 / simulation_dt)
    frame_skip = sim_fps // 30

    b_smpl_joint = True
    time_step = 0.0
    motion_len = motion_lib.get_motion_length(torch.tensor([motion_lib_idx], device=device))
    print("motion_len:", motion_len)
    
    last_command = {} # Initialize empty command dictionary
    smpl_offset_pos = None # For SMPL visualization alignment
    control_dt = simulation_dt * control_decimation

    with mujoco.viewer.launch_passive(m, d) as viewer:
        try:
            for _ in range(50):
                add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        except Exception as e:
            print("Warning: add_visual_capsule failed due to OpenGL/Mesa:", e)

        start = time.time()
        obs_log, torque_log, data_log  = [], [], []

        while viewer.is_running() and time_step < motion_len:
            step_start = time.time()
            counter += 1

            if counter % control_decimation == 0:
                # Get motion command from the new dedicated function
                command = get_falcon_motion_command(motion_lib, motion_lib_idx, time_step, control_dt, device)
                last_command = command

                # Get robot state
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                # Scale observations
                qj_scaled = (qj - default_angles) * dof_pos_scale
                dqj_scaled = dqj * dof_vel_scale
                projected_gravity = get_projected_gravity(quat)
                omega_scaled = omega * config["ang_vel_scale"]
                
                qj_sim = real_to_sim_joint_values(qj_scaled, real_to_sim_idx, sim_dof)
                dqj_sim = real_to_sim_joint_values(dqj_scaled, real_to_sim_idx, sim_dof)

                # Compute observation and get action from policy
                obs_sim = compute_observation_falcon(
                    config, omega_scaled, projected_gravity, qj_sim, dqj_sim, action_sim, obs_buffer, command
                )
                obs_tensor = torch.from_numpy(obs_sim).unsqueeze(0)
                action_sim = policy(obs_tensor.float()).detach().numpy().squeeze()
                obs_buffer._update_obs_buffer(obs_sim)

                # Map action to target joint positions
                action = sim_to_real_joint_values(action_sim, sim_to_real_idx, real_dof)
                target_dof_pos = action * action_scale + default_angles
                target_dof_pos[-14:] += command["ref_upper_dof_pos"] - default_angles[-14:]

            # Apply PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            # (SMPL visualization, viewer sync, logging, and video recording logic remains unchanged)
            if b_smpl_joint:
                motion_dt = motion_lib._motion_dt[0].item()
                num_frames = motion_lib._motion_num_frames[0].item()
                curr_frame_idx = int(time_step / motion_dt) % num_frames
                joint_gt = motion_lib.get_smpl_joint_desired()
                if smpl_offset_pos is None:
                    smpl_root_pos = joint_gt[0, 0]
                    robot_root_pos = d.qpos[:3].copy()
                    smpl_offset_pos = robot_root_pos - smpl_root_pos
                for i in range(joint_gt.shape[1]):
                    aligned_pos = joint_gt[curr_frame_idx, i] + smpl_offset_pos
                    viewer.user_scn.geoms[i].pos = aligned_pos
            
            viewer.sync()
            viewer.cam.lookat = d.qpos.astype(np.float32)[:3]
            time_step += simulation_dt  

            if last_command:
                data_log.append(np.concatenate([
                    [time_step], last_command["lin_vel"], last_command["ang_vel"], d.qvel[:3]
                ]))
            obs_log.append(obs_sim.copy())
            torque_log.append(tau.copy())


            # g_body = projected_gravity_wrt_body(d.qpos[3:7])
            # gx, gy, gz = g_body
            # theta = np.arctan2(np.sqrt(gx*gx + gy*gy), -gz)   # radians
            # if theta > np.deg2rad(45):
            #     print("TILT", np.degrees(theta))
            
            # # Rule 2: Absolute Height
            # height = d.qpos[2]
            # if height < 0.2:
            #     print("HEIGHT", height)

            if counter % frame_skip == 0:
                renderer.update_scene(d, camera=viewer.cam)
                frame = renderer.render()
                video_writer.append_data(frame)

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Save logs and video
    np.savez_compressed(f"logs/log_data_{timestamp}.npz", obs=obs_log, torques=torque_log)
    video_writer.close()
    print(f"Video saved to {video_path}")
    data_log = np.array(data_log)
    header = "time, des_lin_x, des_lin_y, des_ang_z, act_lin_x, act_lin_y, act_lin_z"
    np.savetxt(f"logs/falcon_log_{timestamp}.csv", data_log, delimiter=",", header=header, comments='')
    print(f"Command log saved to logs/falcon_log_{timestamp}.csv")


# python deploy_mujoco_falcon_motionlib.py falcon/g1_29to29_falcon.yaml /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl
