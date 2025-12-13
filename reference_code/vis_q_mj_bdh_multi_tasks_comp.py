import os
import sys
import time
import joblib
import numpy as np
from copy import deepcopy
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R, Slerp
import argparse
from pathlib import Path
import yaml

import imageio
from mujoco import Renderer
from datetime import datetime

import re

# Add project root to path
sys.path.append(os.getcwd())

# Global state
motion_id, time_step, dt, paused = 0, 0, 1 / 30, False
motion_data_all = []
motion_lengths = []
motion_data_keys = []


def key_callback(keycode):
    global motion_id, time_step, paused
    if chr(keycode) == " ":
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif chr(keycode) == "R":
        time_step = 0
        motion_id = 0
        print("Reset motion sequence")



def blend_quat_mujoco(q1_wxyz, q2_wxyz, alpha):
    q1_xyzw = [q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]]
    q2_xyzw = [q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]]
    slerp = Slerp([0, 1], R.from_quat([q1_xyzw, q2_xyzw]))
    r_interp = slerp([alpha])[0]
    q_interp = r_interp.as_quat()
    return np.array([q_interp[3], q_interp[0], q_interp[1], q_interp[2]])  # back to wxyz



def extract_euler_xyz_from_wxyz(q_wxyz):
    """Convert quaternion wxyz to Euler angles [yaw, pitch, roll] (ZYX order)."""
    r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return r.as_euler('zyx')  # [yaw, pitch, roll]


def ensure_quaternion_continuity(quat_sequence):
    for i in range(1, len(quat_sequence)):
        if np.dot(quat_sequence[i], quat_sequence[i - 1]) < 0:
            quat_sequence[i] = -quat_sequence[i]
    return quat_sequence








def get_motions_by_index_list(motion_file: str, motion_indices: list):
    """
    Load motions directly by providing a list of motion indices (0-based).
    
    Args:
        motion_file (str): Path to the motion pkl file
        motion_indices (list): List of motion indices to load (0-based)
        
    Returns:
        (motion_indices, motion_keys) - returns the same indices and corresponding motion keys
    """
    motion_dict = joblib.load(motion_file)
    
    # Get all available motion keys
    available_keys = list(motion_dict.keys())
    
    motion_keys = []
    valid_motion_indices = []
    
    print("--- Selecting Motions by Index List ---")
    print(f"Total available motions: {len(available_keys)}")
    
    for motion_index in motion_indices:
        if 0 <= motion_index < len(available_keys):
            motion_keys.append(available_keys[motion_index])
            valid_motion_indices.append(motion_index)
            print(f"Found motion index {motion_index}: {available_keys[motion_index]}")
        else:
            print(f"Warning: Motion index {motion_index} out of range (0-{len(available_keys)-1})")
    
    return valid_motion_indices, motion_keys

def load_robot_config(robot_config_name: str):
    """Load robot configuration from assets YAML (matches VQVAE viewer)."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "assets" / f"{robot_config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Robot config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_motion_indices(indices_arg: str):
    """Parse comma/space-separated indices string into list of ints."""
    if not indices_arg:
        return []
    return [int(x) for x in re.split(r"[,\s]+", indices_arg) if x.strip() != ""]


def main():
    global motion_id, time_step, dt, paused, motion_data_all, motion_lengths, motion_data_keys

    parser = argparse.ArgumentParser(description="Visualize AMASS motions in MuJoCo (no Hydra).")
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1_kungfu_23dof_bdh",
        help="Robot configuration name (assets/{name}.yaml)",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default="/home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl",
        help="Path to AMASS pkl file",
    )
    parser.add_argument(
        "--motion-indices",
        type=str,
        default="8,286",
        help="Comma/space separated motion indices to load (e.g., '8,286')",
    )
    args = parser.parse_args()

    motion_file = args.motion_file
    motion_indices = parse_motion_indices(args.motion_indices)
    if not motion_indices:
        raise ValueError("No motion indices provided.")
    print("Motion indices:", motion_indices)
    
    # Get motion keys directly from motion indices
    motion_indices, motion_keys = get_motions_by_index_list(motion_file, motion_indices)
    
    # Load motions directly using motion keys
    motion_dict = joblib.load(motion_file)
    motion_data_all = [motion_dict[key] for key in motion_keys]
    motion_data_keys = motion_keys
    motion_lengths = [m['dof'].shape[0] for m in motion_data_all]

    print("Loaded motion keys:", motion_data_keys)
    print("Lengths:", motion_lengths)

    # Load robot config (argparse/YAML like VQVAE viewer)
    robot_config = load_robot_config(args.robot)
    if "asset" not in robot_config or "assetFileName" not in robot_config["asset"]:
        raise ValueError(f"Could not find asset.assetFileName in robot config: {robot_config.keys()}")
    humanoid_xml = robot_config["asset"]["assetFileName"]

    # Load model
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_model.opt.timestep = dt
    mj_data = mujoco.MjData(mj_model)


    # Transition-related state
    transitioning = False
    transition_cnt = 0
    transition_frames = int(1 / dt)

    print("dt:",dt) 
    print("transition_frames:", transition_frames)

    root_pos_fixed = None
    root_rot_fixed = None
    dof_start = None

    next_root_pos = None
    next_root_rot = None
    next_dof = None
    updated_global_offset = False

    saved_states = []
    motion_name_str = "_".join(str(i) for i in motion_indices)
    
    # Global cumulative time that doesn't reset between motions
    global_time = 0.0

    ##video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    video_path = f"logs/{motion_name_str}_recorded_mujoco.mp4"
    os.makedirs("logs", exist_ok=True)

    renderer = Renderer(mj_model, width=640, height=480)
    video_writer = imageio.get_writer(video_path, fps=30)
    sim_fps = int(1 / dt)
    frame_skip = sim_fps // 30
    counter = 0

    
    print("== Raw Start and End Yaw of Each Motion BEFORE Alignment ==")
    for i, motion in enumerate(motion_data_all):
        q_start = motion['root_rot'][0]
        q_end = motion['root_rot'][-1]
        
        roll_start, pitch_start, yaw_start = R.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]]).as_euler('xyz')
        roll_end, pitch_end, yaw_end = R.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]]).as_euler('xyz')
        
        start_pos = motion['root_trans_offset'][0]
        end_pos = motion['root_trans_offset'][-1]

        print(f"Motion {i}:")
        print(f"  Start Roll = {np.degrees(roll_start):.2f}°, Pitch = {np.degrees(pitch_start):.2f}°, Yaw = {np.degrees(yaw_start):.2f}°")
        print(f"  End   Roll = {np.degrees(roll_end):.2f}°, Pitch = {np.degrees(pitch_end):.2f}°, Yaw = {np.degrees(yaw_end):.2f}°")
        print(f"  Start Pos = {start_pos}")
        print(f"  End   Pos = {end_pos}\n")
 

    print("\nApplying sequential roll, pitch, and yaw alignment between motions...")

    # Get initial orientation and position
    prev_q_end = motion_data_all[0]['root_rot'][-1]
    prev_roll, prev_pitch, prev_yaw = R.from_quat([prev_q_end[1], prev_q_end[2], prev_q_end[3], prev_q_end[0]]).as_euler('xyz')
    global_end_pos = motion_data_all[0]['root_trans_offset'][-1].copy()

    for i in range(1, len(motion_data_all)):
        motion = motion_data_all[i]

        curr_q_start = motion['root_rot'][0]
        curr_roll, curr_pitch, curr_yaw = R.from_quat([curr_q_start[1], curr_q_start[2], curr_q_start[3], curr_q_start[0]]).as_euler('xyz')

        # === Rotation Compensation ===
        delta_roll = prev_roll - curr_roll

        original_first_pos = motion['root_trans_offset'][0].copy()

        for t in range(len(motion['root_rot'])):
            # --- Orientation update ---
            q = motion['root_rot'][t]
            r = R.from_quat([q[1], q[2], q[3], q[0]])  # wxyz → xyzw
            roll, pitch, yaw = r.as_euler('xyz')  # Consistent use of xyz

            roll += delta_roll

            r_new = R.from_euler('xyz', [roll, pitch, yaw])
            q_new = r_new.as_quat()
            motion['root_rot'][t] = [q_new[3], q_new[0], q_new[1], q_new[2]]  # xyzw → wxyz

            local_vec = motion['root_trans_offset'][t] - original_first_pos
            
            rot_x = R.from_euler('z', delta_roll)
            rotated_vec = rot_x.apply(local_vec)

            motion['root_trans_offset'][t] = original_first_pos + rotated_vec

        # === Position Offset Correction ===
        first_pos = motion['root_trans_offset'][0]
        offset_vec = global_end_pos - first_pos
        for t in range(len(motion['root_trans_offset'])):
            motion['root_trans_offset'][t] += offset_vec

        # Update reference
        motion_data_all[i] = motion
        global_end_pos = motion['root_trans_offset'][-1].copy()
        new_q_end = motion['root_rot'][-1]
        prev_roll, prev_pitch, prev_yaw = R.from_quat([new_q_end[1], new_q_end[2], new_q_end[3], new_q_end[0]]).as_euler('xyz')

        print(f"Motion {i} aligned: Δroll = {np.degrees(delta_roll):.2f}°")

    # === Final Report ===
    print("\n== Start and End Orientation and Position of Each Motion ==")
    for i, motion in enumerate(motion_data_all):
        q_start = motion['root_rot'][0]
        q_end = motion['root_rot'][-1]

        roll_start, pitch_start, yaw_start = R.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]]).as_euler('xyz')
        roll_end, pitch_end, yaw_end = R.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]]).as_euler('xyz')

        start_pos = motion['root_trans_offset'][0]
        end_pos = motion['root_trans_offset'][-1]

        print(f"Motion {i}:")
        print(f"  Start Roll = {np.degrees(roll_start):.2f}°, Pitch = {np.degrees(pitch_start):.2f}°, Yaw = {np.degrees(yaw_start):.2f}°")
        print(f"  End   Roll = {np.degrees(roll_end):.2f}°, Pitch = {np.degrees(pitch_end):.2f}°, Yaw = {np.degrees(yaw_end):.2f}°")
        print(f"  Start Pos = {start_pos}")
        print(f"  End   Pos = {end_pos}\n")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            curr_motion = motion_data_all[motion_id]
            motion_len = motion_lengths[motion_id]

            curr_index = int(time_step / dt)

            if curr_index < motion_len:
                # These are already globally aligned
                global_pos = curr_motion['root_trans_offset'][curr_index]
                global_rot = curr_motion['root_rot'][curr_index]  # wxyz

                mj_data.qpos[:3] = global_pos
                mj_data.qpos[3:7] = global_rot[[3, 0, 1, 2]]
                mj_data.qpos[7:] = curr_motion['dof'][curr_index]

                # yaw_now, pitch_now, roll_now = extract_euler_xyz_from_wxyz(global_rot)
                # print(f"[Live] Motion {motion_id} | Frame {curr_index} | Yaw={np.degrees(yaw_now):.2f}°, Pitch={np.degrees(pitch_now):.2f}°, Roll={np.degrees(roll_now):.2f}°")


            elif curr_index < motion_len + transition_frames:
                # === End after last motion ===
                if motion_id == len(motion_data_all) - 1:
                    print("All motions finished.")
                    break  # Exit the viewer loop

                # === Smooth transition ===
                if not transitioning:
                    transitioning = True
                    transition_cnt = 0
                    prev_motion = curr_motion
                    next_motion = motion_data_all[(motion_id + 1) % len(motion_data_all)]

                    # Save transition start/end
                    pos_start = prev_motion['root_trans_offset'][-1]
                    rot_start = prev_motion['root_rot'][-1]
                    dof_start = prev_motion['dof'][-1]

                    pos_end = next_motion['root_trans_offset'][0]
                    rot_end = next_motion['root_rot'][0]
                    dof_end = next_motion['dof'][0]

                alpha = (curr_index - motion_len) / transition_frames
                alpha = np.clip(alpha, 0.0, 1.0)

                # Interpolate position
                pos_interp = (1 - alpha) * pos_start + alpha * pos_end

                # Interpolate orientation
                rot_interp = blend_quat_mujoco(rot_start, rot_end, alpha)

                # Interpolate joints
                dof_interp = (1 - alpha) * dof_start + alpha * dof_end

                mj_data.qpos[:3] = pos_interp
                mj_data.qpos[3:7] = rot_interp[[3, 0, 1, 2]]
                mj_data.qpos[7:] = dof_interp

                transition_cnt += 1

            else:

                # === Switch to next motion ===
                transitioning = False
                motion_id = (motion_id + 1) % len(motion_data_all)
                time_step = 0
                print("motion_id", motion_id)
                continue



            #### save globally
            motion_identifier = motion_id  # Integer ID to distinguish different motions (0, 1, 2, ...)
            state = np.concatenate([
                [global_time],                    #  1 - Global cumulative time (doesn't reset)
                mj_data.qpos[:3],           # root position 3
                mj_data.qpos[3:7],          # root rotation 4
                mj_data.qpos[7:].copy(),    # DOF 23
                [motion_identifier]         # Motion ID (0, 1, 2, ...) to distinguish different motions 1
            ])
            saved_states.append(state)


            mujoco.mj_forward(mj_model, mj_data)

            viewer.sync()
            if not paused:
                time_step += dt
                global_time += dt  # Global time accumulates continuously

            if counter % frame_skip == 0:
                root_pos = mj_data.qpos[:3].copy()  # Root world position

                # Configure camera to look at root
                viewer.cam.lookat[:] =  mj_data.qpos.astype(np.float32)[:3] #root_pos
                # viewer.cam.distance = 2.0  # Zoom level
                # viewer.cam.azimuth = -20   # Horizontal angle
                # viewer.cam.elevation = -20 # Vertical angle
                # viewer.cam.trackbodyid = -1
                # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

                # Render and save frame
                renderer.update_scene(mj_data, camera=viewer.cam)
                frame = renderer.render()
                video_writer.append_data(frame)


            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


    # Create filename based on motion sequence
    save_name = f"saved_desired_states_{motion_name_str}.npy"

    # Save all states to a single file
    saved_states = np.array(saved_states)
    np.save(save_name, saved_states)
    np.save("/home/baekdh/dh_workspace/data_deploy/deploy_pkl/motions/sequence_data/comparison/" + save_name, saved_states)
    print(f"Saved {saved_states.shape[0]} frames to '{save_name}'")

    video_writer.close()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main()




# python reference_code/vis_q_mj_bdh_multi_tasks_comp.py  --motion-indices 8,286