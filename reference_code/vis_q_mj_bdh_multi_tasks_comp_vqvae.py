import os
import sys
import time
import joblib
import numpy as np
from copy import deepcopy
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R, Slerp
import hydra
from omegaconf import DictConfig

import imageio
from mujoco import Renderer
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

# Global state
motion_id, time_step, dt, paused = 0, 0, 1 / 30, False
# dt = 0.02 ############################

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


def align_vqvae_motions(motion_sequence):
    """
    Apply motion alignment logic for VQVAE-generated motions.
    Similar to the alignment logic in the original visualization code.
    """
    print("\nApplying sequential roll, pitch, and yaw alignment between VQVAE motions...")
    
    if len(motion_sequence) <= 1:
        return motion_sequence
    
    # Get initial orientation and position from first motion
    prev_q_end = motion_sequence[0]['root_rot'][-1]
    prev_roll, prev_pitch, prev_yaw = R.from_quat([prev_q_end[1], prev_q_end[2], prev_q_end[3], prev_q_end[0]]).as_euler('xyz')
    global_end_pos = motion_sequence[0]['root_trans_offset'][-1].copy()
    
    for i in range(1, len(motion_sequence)):
        motion = motion_sequence[i]
        
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
            
            # --- Position update ---
            local_vec = motion['root_trans_offset'][t] - original_first_pos
            
            rot_x = R.from_euler('z', delta_roll)
            rotated_vec = rot_x.apply(local_vec)
            
            motion['root_trans_offset'][t] = original_first_pos + rotated_vec
        
        # === Position Offset Correction ===
        first_pos = motion['root_trans_offset'][0]
        offset_vec = global_end_pos - first_pos
        for t in range(len(motion['root_trans_offset'])):
            motion['root_trans_offset'][t] += offset_vec
        
        # Update reference for next motion
        motion_sequence[i] = motion
        global_end_pos = motion['root_trans_offset'][-1].copy()
        new_q_end = motion['root_rot'][-1]
        prev_roll, prev_pitch, prev_yaw = R.from_quat([new_q_end[1], new_q_end[2], new_q_end[3], new_q_end[0]]).as_euler('xyz')
        
        print(f"VQVAE Motion {i} aligned: Δroll = {np.degrees(delta_roll):.2f}°")
    
    return motion_sequence


def load_vqvae_motion_direct(vqvae_file_path):
    """
    Load VQVAE-generated motion directly from the generated PKL file.
    This assumes the file is already in AMASS format from generate_motion_from_vqvae_s2.py
    """
    print(f"Loading VQVAE motion directly from: {vqvae_file_path}")
    
    try:
        # Load the VQVAE-generated motion (already in AMASS format)
        vqvae_motion_dict = joblib.load(vqvae_file_path)
        vqvae_motion_key = list(vqvae_motion_dict.keys())[0]
        vqvae_motion_data = vqvae_motion_dict[vqvae_motion_key]
        
        print(f"Loaded VQVAE motion: {vqvae_motion_key}")
        print(f"Motion data keys: {list(vqvae_motion_data.keys())}")
        print(f"Motion length: {vqvae_motion_data['dof'].shape[0]} frames")
        
        return vqvae_motion_data
        
    except Exception as e:
        print(f"Error loading VQVAE motion: {e}")
        return None


@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg: DictConfig):
    global motion_id, time_step, dt, paused, motion_data_all, motion_lengths, motion_data_keys

    # Configuration for motion loading
    use_vqvae_motions = True  # Set to True to use VQVAE motions, False for original motions
    
    if use_vqvae_motions:
        # VQVAE motion configuration - load directly from generated files
        vqvae_motion_files = [
            "/home/dhbaek/dh_workspace/vqvae_motion_g1/outputs/vqvae_motions/vqvae_motion_000.pkl",
            "/home/dhbaek/dh_workspace/vqvae_motion_g1/outputs/vqvae_motions/vqvae_motion_001.pkl", 
            "/home/dhbaek/dh_workspace/vqvae_motion_g1/outputs/vqvae_motions/vqvae_motion_002.pkl"
        ]
        
        print("=== Loading VQVAE-Generated Motions ===")
        motion_data_all = []
        
        for vqvae_file in vqvae_motion_files:
            if os.path.exists(vqvae_file):
                motion_data = load_vqvae_motion_direct(vqvae_file)
                if motion_data is not None:
                    motion_data_all.append(motion_data)
                    print(f"✅ Successfully loaded: {vqvae_file}")
                else:
                    print(f"❌ Failed to load: {vqvae_file}")
            else:
                print(f"❌ File not found: {vqvae_file}")
        
        if not motion_data_all:
            print("No VQVAE motions loaded. Falling back to original motions.")
            use_vqvae_motions = False
    
    if not use_vqvae_motions:
        # Original motion loading (fallback)
        data_pkl = "/home/dhbaek/dh_workspace/vqvae_motion_g1/outputs/vqvae_motion_0.pkl"
        
        motions = joblib.load(data_pkl)
        all_keys = list(motions.keys())
        
        # Choose what you want to play (by index or by name)
        select_id = 0
        
        _selected_env = os.getenv("SELECTED_IDS", "").strip()
        if _selected_env:
            import re as _re
            selected = [int(x) for x in _re.split(r"[,\s]+", _selected_env) if x]
        else:
            selected = [select_id]
        
        if all(isinstance(x, int) for x in selected):
            motion_names = [all_keys[i] for i in selected]
        else:
            motion_names = list(selected)
        
        motion_data_all = [motions[k] for k in motion_names]
    
    motion_lengths = [m['dof'].shape[0] for m in motion_data_all]
    
    # Create descriptive motion keys
    if use_vqvae_motions:
        motion_data_keys = [f"vqvae_motion_{i:03d}" for i in range(len(motion_data_all))]
    else:
        motion_data_keys = [f"original_motion_{i}" for i in range(len(motion_data_all))]

    # Load model
    humanoid_xml = cfg.robot.asset.assetFileName
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_model.opt.timestep = dt
    mj_data = mujoco.MjData(mj_model)

    # Transition-related state
    transitioning = False
    transition_cnt = 0
    transition_frames = int(1 / dt)

    print("dt:", dt) 
    print("transition_frames:", transition_frames)

    root_pos_fixed = None
    root_rot_fixed = None
    dof_start = None

    next_root_pos = None
    next_root_rot = None
    next_dof = None
    updated_global_offset = False

    saved_states = []

    # Video setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    motion_type = "vqvae" if use_vqvae_motions else "original"
    video_path = f"/home/dhbaek/dh_workspace/data_deploy/deploy_pkl/each_motion_ref_video/{motion_data_keys[0]}_recorded_mujoco_{motion_type}.mp4"
    os.makedirs("logs", exist_ok=True)

    renderer = Renderer(mj_model, width=640, height=480)
    video_writer = imageio.get_writer(video_path, fps=30)
    sim_fps = int(1 / dt)
    frame_skip = sim_fps // 30
    counter = 0

    # Print motion information
    motion_type_str = "VQVAE-Generated" if use_vqvae_motions else "Original"
    print(f"\n== {motion_type_str} Motion Information ==")
    for i, motion in enumerate(motion_data_all):
        q_start = motion['root_rot'][0]
        q_end = motion['root_rot'][-1]
        
        roll_start, pitch_start, yaw_start = R.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]]).as_euler('xyz')
        roll_end, pitch_end, yaw_end = R.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]]).as_euler('xyz')
        
        start_pos = motion['root_trans_offset'][0]
        end_pos = motion['root_trans_offset'][-1]

        print(f"{motion_type_str} Motion {i}:")
        print(f"  Start Roll = {np.degrees(roll_start):.2f}°, Pitch = {np.degrees(pitch_start):.2f}°, Yaw = {np.degrees(yaw_start):.2f}°")
        print(f"  End   Roll = {np.degrees(roll_end):.2f}°, Pitch = {np.degrees(pitch_end):.2f}°, Yaw = {np.degrees(yaw_end):.2f}°")
        print(f"  Start Pos = {start_pos}")
        print(f"  End   Pos = {end_pos}")
        print(f"  Frames = {motion['dof'].shape[0]}\n")

    # Apply motion alignment for VQVAE motions
    if use_vqvae_motions and len(motion_data_all) > 1:
        motion_data_all = align_vqvae_motions(motion_data_all)

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
                mj_data.qpos[3:7] = global_rot[[3, 0, 1, 2]]  # Convert wxyz to xyzw for Mujoco
                mj_data.qpos[7:] = curr_motion['dof'][curr_index]

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
                mj_data.qpos[3:7] = rot_interp[[3, 0, 1, 2]]  # Convert wxyz to xyzw for Mujoco
                mj_data.qpos[7:] = dof_interp

                transition_cnt += 1

            else:
                # === Switch to next motion ===
                transitioning = False
                motion_id = (motion_id + 1) % len(motion_data_all)
                time_step = 0
                print("motion_id", motion_id)
                continue

            # Save state globally
            transition_flag = 1.0 if transitioning else 0.0
            state = np.concatenate([
                [time_step],                    #  1
                mj_data.qpos[:3],           # root position 3
                mj_data.qpos[3:7],          # root rotation 4
                mj_data.qpos[7:].copy(),    # DOF 23
                [transition_flag]           # Transition flag (1.0 during transition, 0.0 otherwise) 1
            ])
            saved_states.append(state)

            mujoco.mj_forward(mj_model, mj_data)

            viewer.sync()
            if not paused:
                time_step += dt

            if counter % frame_skip == 0:
                root_pos = mj_data.qpos[:3].copy()  # Root world position

                # Configure camera to look at root
                viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[:3]  # root_pos

                # Render and save frame
                renderer.update_scene(mj_data, camera=viewer.cam)
                frame = renderer.render()
                video_writer.append_data(frame)

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # Save results
    motion_name_str = "_".join(motion_data_keys)
    save_name = f"saved_desired_states_{motion_data_keys[0]}_vqvae.npy"

    video_writer.close()
    print(f"Video saved to {video_path}")
    print(f"Motion type: {'VQVAE-Generated' if use_vqvae_motions else 'Original'}")
    print(f"Total motions processed: {len(motion_data_all)}")
    print(f"Total frames: {sum(motion_lengths)}")


if __name__ == "__main__":
    main()


# Usage examples:
# python scripts/vis/vis_q_mj_bdh_multi_tasks_comp_vqvae.py robot=unitree_g1_kungfu_23dof_bdh
