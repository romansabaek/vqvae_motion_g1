import os
import sys
import time
import re
import joblib
import numpy as np
from copy import deepcopy
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R, Slerp
import yaml
import argparse
from pathlib import Path

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
    # root_rot is stored in XYZW format (matches convert_to_amass_global_from_local)
    prev_q_end = motion_sequence[0]['root_rot'][-1]
    prev_roll, prev_pitch, prev_yaw = R.from_quat(prev_q_end).as_euler('xyz')  # XYZW format, use directly
    global_end_pos = motion_sequence[0]['root_trans_offset'][-1].copy()
    
    for i in range(1, len(motion_sequence)):
        motion = motion_sequence[i]
        
        curr_q_start = motion['root_rot'][0]
        curr_roll, curr_pitch, curr_yaw = R.from_quat(curr_q_start).as_euler('xyz')  # XYZW format, use directly
        
        # === Rotation Compensation ===
        delta_roll = prev_roll - curr_roll
        
        original_first_pos = motion['root_trans_offset'][0].copy()
        
        for t in range(len(motion['root_rot'])):
            # --- Orientation update ---
            q = motion['root_rot'][t]  # XYZW format
            r = R.from_quat(q)  # XYZW format, use directly
            roll, pitch, yaw = r.as_euler('xyz')  # Consistent use of xyz
            
            roll += delta_roll
            
            r_new = R.from_euler('xyz', [roll, pitch, yaw])
            q_new = r_new.as_quat()  # Returns XYZW format
            motion['root_rot'][t] = q_new  # Store in XYZW format
            
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
        prev_roll, prev_pitch, prev_yaw = R.from_quat(new_q_end).as_euler('xyz')  # XYZW format, use directly
        
        print(f"VQVAE Motion {i} aligned: Δroll = {np.degrees(delta_roll):.2f}°")
    
    return motion_sequence


def load_vqvae_motion_direct(vqvae_file_path):
    """
    Load VQVAE-generated motion directly from the generated PKL or NPY file.
    - PKL files: Assumes AMASS format from generate_motion_from_vqvae_s2.py
    - NPY files: Assumes format from generate_motion_from_vqvae_s2_npy.py
      Format: [time, root_pos(3), root_rot(4) WXYZ, dof_pos(23), motion_id] = 32 cols
    """
    print(f"Loading VQVAE motion directly from: {vqvae_file_path}")
    
    try:
        file_path = Path(vqvae_file_path)
        
        # Check if it's an NPY file
        if file_path.suffix.lower() == '.npy':
            # Load NPY file format: [time, root_pos(3), root_rot(4) WXYZ, dof_pos(23), motion_id] = 32 cols
            trajectory_data = np.load(vqvae_file_path)
            
            if trajectory_data.shape[1] != 32:
                raise ValueError(f"Expected 32 columns in NPY file, got {trajectory_data.shape[1]}")
            
            num_frames = trajectory_data.shape[0]
            
            # Extract data
            time_stamps = trajectory_data[:, 0]
            root_pos = trajectory_data[:, 1:4]  # [T, 3]
            root_rot_wxyz = trajectory_data[:, 4:8]  # [T, 4] in WXYZ format
            dof_pos = trajectory_data[:, 8:31]  # [T, 23]
            
            # Convert root rotation from WXYZ to XYZW format
            # The visualization code expects XYZW format (it converts to WXYZ for MuJoCo with [3,0,1,2] indexing)
            # This matches the format returned by convert_to_amass_global_from_local
            root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]  # [x, y, z, w] = [1, 2, 3, 0]
            
            # Calculate fps from time stamps
            if len(time_stamps) > 1:
                dt = np.mean(np.diff(time_stamps))
                fps = 1.0 / dt if dt > 0 else 30.0
            else:
                fps = 30.0
            
            # Create AMASS-like motion dict
            # Note: root_rot is stored in XYZW format to match convert_to_amass_global_from_local
            # and what the visualization code expects (despite misleading comments)
            vqvae_motion_data = {
                "root_trans_offset": root_pos.astype(np.float32),
                "root_rot": root_rot_xyzw.astype(np.float32),  # XYZW format (matches convert_to_amass_global_from_local)
                "dof": dof_pos.astype(np.float32),
                "fps": fps,
                "pose_aa": np.zeros((num_frames, 72), dtype=np.float32),  # Placeholder for compatibility
                "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),  # Placeholder for compatibility
            }
            
            print(f"Loaded VQVAE motion from NPY: {file_path.name}")
            print(f"Motion data keys: {list(vqvae_motion_data.keys())}")
            print(f"Motion length: {vqvae_motion_data['dof'].shape[0]} frames")
            print(f"FPS: {fps:.2f}")
            
            return vqvae_motion_data
        
        else:
            # Load PKL file (original behavior)
            vqvae_motion_dict = joblib.load(vqvae_file_path)
            vqvae_motion_key = list(vqvae_motion_dict.keys())[0]
            vqvae_motion_data = vqvae_motion_dict[vqvae_motion_key]
            
            print(f"Loaded VQVAE motion from PKL: {vqvae_motion_key}")
            print(f"Motion data keys: {list(vqvae_motion_data.keys())}")
            print(f"Motion length: {vqvae_motion_data['dof'].shape[0]} frames")
            
            return vqvae_motion_data
        
    except Exception as e:
        print(f"Error loading VQVAE motion: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_motion_blocks_npy(blocks_dir, block_ids=None):
    """
    Load motion blocks from NPY files in the motion_blocks_npy directory.
    Converts raw local features to AMASS format for visualization.
    
    Args:
        blocks_dir: Directory containing motion_block_*.npy files
        block_ids: List of block IDs to load (e.g., [0, 1, 2]). If None, loads all.
    
    Returns:
        List of motion data dictionaries in AMASS format
    """
    print(f"Loading motion blocks from: {blocks_dir}")
    
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from scripts.vqvae_gen_init import convert_to_amass_global_from_local
    except ImportError:
        print("Warning: Could not import convert_to_amass_global_from_local. Using simple conversion.")
        convert_to_amass_global_from_local = None
    
    blocks_path = Path(blocks_dir)
    if not blocks_path.exists():
        raise FileNotFoundError(f"Motion blocks directory not found: {blocks_dir}")
    
    # Find all motion block files
    block_files = sorted(blocks_path.glob("motion_block_*.npy"))
    
    if not block_files:
        raise ValueError(f"No motion_block_*.npy files found in {blocks_dir}")
    
    # Filter by block_ids if provided and extract block IDs
    block_file_map = {}  # Map block_id -> file_path
    if block_ids is not None:
        block_ids_set = set(block_ids)
        print(f"Filtering blocks by IDs: {sorted(block_ids_set)}")
        for f in block_files:
            match = re.search(r'motion_block_(\d+)\.npy', f.name)
            if match:
                file_block_id = int(match.group(1))
                if file_block_id in block_ids_set:
                    block_file_map[file_block_id] = f
        print(f"Found {len(block_file_map)} matching file(s):")
        for bid, f in sorted(block_file_map.items()):
            print(f"  - {f.name} (block_id: {bid})")
    else:
        # Load all files and extract their block IDs
        for f in block_files:
            match = re.search(r'motion_block_(\d+)\.npy', f.name)
            if match:
                file_block_id = int(match.group(1))
                block_file_map[file_block_id] = f
        print(f"Loading all {len(block_file_map)} motion block file(s)")
    
    if not block_file_map:
        raise ValueError(f"No motion block files match the requested block_ids")
    
    motion_data_all = []
    
    # Process files in sorted order by block_id to ensure consistency
    for block_id, block_file in sorted(block_file_map.items()):
        
        print(f"\nLoading block {block_id}: {block_file.name}")
        
        # Load raw motion features [T, 50]
        raw_features = np.load(block_file)
        num_frames = raw_features.shape[0]
        
        print(f"  Raw features shape: {raw_features.shape}")
        print(f"  Feature stats - min: {raw_features.min():.4f}, max: {raw_features.max():.4f}, mean: {raw_features.mean():.4f}, std: {raw_features.std():.4f}")
        print(f"  First frame sample (first 5 features): {raw_features[0, :5]}")
        
        # Create a neutral original motion for conversion
        # This provides initial conditions (starting pose/orientation)
        neutral_motion = {
            "root_trans_offset": np.zeros((num_frames, 3), dtype=np.float32),
            "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_frames, 1)),  # Neutral quaternion [x,y,z,w]
            "dof": np.zeros((num_frames, 23), dtype=np.float32),  # Neutral joint positions
            "fps": 30.0,
            "pose_aa": np.zeros((num_frames, 72), dtype=np.float32),
            "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
        }
        
        # Convert local features to AMASS global format
        if convert_to_amass_global_from_local is not None:
            try:
                # Use motion_id parameter to ensure each block is processed uniquely
                amass_motion = convert_to_amass_global_from_local(
                    local_features=raw_features,
                    original_motion=neutral_motion,
                    motion_id=block_id,  # Use block_id instead of 0 to ensure uniqueness
                )
                motion_data_all.append(amass_motion)
                print(f"  Converted to AMASS format: {amass_motion['dof'].shape[0]} frames")
                print(f"  Final position: {amass_motion['root_trans_offset'][-1]}")
                print(f"  Final DOF sample (first 3): {amass_motion['dof'][-1, :3]}")
            except Exception as e:
                print(f"  Error converting to AMASS format: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Using simple conversion instead...")
                # Fallback: simple conversion
                amass_motion = {
                    "root_trans_offset": np.zeros((num_frames, 3), dtype=np.float32),
                    "root_rot": neutral_motion["root_rot"],
                    "dof": raw_features[:, 4:27].astype(np.float32),  # Extract DOF positions
                    "fps": 30.0,
                    "pose_aa": neutral_motion["pose_aa"],
                    "smpl_joints": neutral_motion["smpl_joints"],
                }
                motion_data_all.append(amass_motion)
        else:
            # Simple fallback conversion
            amass_motion = {
                "root_trans_offset": np.zeros((num_frames, 3), dtype=np.float32),
                "root_rot": neutral_motion["root_rot"],
                "dof": raw_features[:, 4:27].astype(np.float32),  # Extract DOF positions
                "fps": 30.0,
                "pose_aa": neutral_motion["pose_aa"],
                "smpl_joints": neutral_motion["smpl_joints"],
            }
            motion_data_all.append(amass_motion)
    
    print(f"\nSuccessfully loaded {len(motion_data_all)} motion block(s)")
    
    # Verify that different blocks have different data
    if len(motion_data_all) > 1:
        print("\nVerifying block differences:")
        for i in range(len(motion_data_all) - 1):
            dof_diff = np.abs(motion_data_all[i]['dof'] - motion_data_all[i+1]['dof']).mean()
            pos_diff = np.abs(motion_data_all[i]['root_trans_offset'] - motion_data_all[i+1]['root_trans_offset']).mean()
            print(f"  Block {i} vs Block {i+1}: DOF diff={dof_diff:.6f}, Pos diff={pos_diff:.6f}")
            if dof_diff < 1e-6 and pos_diff < 1e-6:
                print(f"  ⚠️  WARNING: Blocks {i} and {i+1} appear to be identical!")
    
    return motion_data_all


def load_robot_config(robot_config_name: str):
    """
    Load robot configuration from YAML file in assets directory.
    
    Args:
        robot_config_name: Name of the robot config file (e.g., 'unitree_g1_kungfu_23dof_bdh')
    
    Returns:
        dict: Robot configuration dictionary
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent
    config_path = project_root / "assets" / f"{robot_config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Robot config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    global motion_id, time_step, dt, paused, motion_data_all, motion_lengths, motion_data_keys

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize VQVAE-generated motions in MuJoCo")
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1_kungfu_23dof_bdh",
        help="Robot configuration name (without .yaml extension)"
    )
    parser.add_argument(
        "--motion-files",
        type=str,
        nargs="+",
        default=None,
        help="Paths to VQVAE motion PKL files (if not provided, uses default paths)"
    )
    parser.add_argument(
        "--motion-blocks-dir",
        type=str,
        default=None,
        help="Directory containing motion_block_*.npy files (e.g., outputs/motion_blocks_npy)"
    )
    parser.add_argument(
        "--block-ids",
        type=str,
        default=None,
        help="Comma-separated block IDs to visualize (e.g., '0,1,2' or '0-5'). If not provided, loads all blocks."
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original motions instead of VQVAE motions"
    )
    
    args = parser.parse_args()

    # Load robot configuration
    print(f"Loading robot configuration: {args.robot}")
    robot_config = load_robot_config(args.robot)
    
    # Extract robot XML path from config
    if 'asset' in robot_config and 'assetFileName' in robot_config['asset']:
        humanoid_xml = robot_config['asset']['assetFileName']
    else:
        raise ValueError(f"Could not find asset.assetFileName in robot config: {robot_config.keys()}")

    # Configuration for motion loading
    use_vqvae_motions = not args.use_original
    use_motion_blocks = args.motion_blocks_dir is not None
    
    if use_motion_blocks:
        # Load motion blocks from NPY files
        print("=== Loading Motion Blocks from NPY Files ===")
        
        # Parse block_ids if provided
        block_ids = None
        if args.block_ids:
            block_ids = []
            for x in args.block_ids.split(','):
                x = x.strip()
                if '-' in x:
                    start, end = map(int, x.split('-'))
                    block_ids.extend(range(start, end + 1))
                else:
                    block_ids.append(int(x))
            print(f"Requested block IDs: {sorted(block_ids)}")
        
        motion_data_all = load_motion_blocks_npy(args.motion_blocks_dir, block_ids)
        use_vqvae_motions = True  # Treat motion blocks as VQVAE motions
        
    elif use_vqvae_motions:
        # VQVAE motion configuration - load directly from generated files
        if args.motion_files:
            vqvae_motion_files = args.motion_files
        else:
            # Default motion files
            project_root = Path(__file__).parent.parent
            vqvae_motion_files = [
                str(project_root / "outputs" / "vqvae_motions" / "vqvae_motion_083.pkl"),
                str(project_root / "outputs" / "vqvae_motions" / "vqvae_motion_201.pkl") 
            ]
        
        print("=== Loading VQVAE-Generated Motions ===")
        motion_data_all = []
        
        for vqvae_file in vqvae_motion_files:
            if os.path.exists(vqvae_file):
                motion_data = load_vqvae_motion_direct(vqvae_file)
                if motion_data is not None:
                    motion_data_all.append(motion_data)
                    print(f" Successfully loaded: {vqvae_file}")
                else:
                    print(f" Failed to load: {vqvae_file}")
            else:
                print(f" File not found: {vqvae_file}")
        
        if not motion_data_all:
            print("No VQVAE motions loaded. Falling back to original motions.")
            use_vqvae_motions = False
    
    if not use_vqvae_motions:
        # Original motion loading (fallback)
        project_root = Path(__file__).parent.parent
        data_pkl = str(project_root / "outputs" / "vqvae_motion_0.pkl")
        
        if not os.path.exists(data_pkl):
            raise FileNotFoundError(f"Original motion file not found: {data_pkl}")
        
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
    if use_motion_blocks:
        motion_data_keys = [f"motion_block_{i:03d}" for i in range(len(motion_data_all))]
    elif use_vqvae_motions:
        motion_data_keys = [f"vqvae_motion_{i:03d}" for i in range(len(motion_data_all))]
    else:
        motion_data_keys = [f"original_motion_{i}" for i in range(len(motion_data_all))]

    # Load model
    print(f"Loading MuJoCo model from: {humanoid_xml}")
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
    if use_motion_blocks:
        motion_type = "motion_blocks"
    else:
        motion_type = "vqvae" if use_vqvae_motions else "original"
    video_path = f"/home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_ref_video/{motion_data_keys[0]}_recorded_mujoco_{motion_type}.mp4"
    os.makedirs("logs", exist_ok=True)

    renderer = Renderer(mj_model, width=640, height=480)
    video_writer = imageio.get_writer(video_path, fps=30)
    sim_fps = int(1 / dt)
    frame_skip = sim_fps // 30
    counter = 0

    # Print motion information
    if use_motion_blocks:
        motion_type_str = "Motion Blocks (Codebook)"
    else:
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

    # Apply motion alignment for VQVAE motions and motion blocks
    if (use_vqvae_motions or use_motion_blocks) and len(motion_data_all) > 1:
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
                global_rot = curr_motion['root_rot'][curr_index]  # XYZW format

                mj_data.qpos[:3] = global_pos
                mj_data.qpos[3:7] = global_rot[[3, 0, 1, 2]]  # Convert XYZW to WXYZ for MuJoCo
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
                # Convert XYZW to WXYZ for blend_quat_mujoco (which expects WXYZ)
                rot_start_wxyz = rot_start[[3, 0, 1, 2]]  # XYZW -> WXYZ: [x,y,z,w] -> [w,x,y,z]
                rot_end_wxyz = rot_end[[3, 0, 1, 2]]  # XYZW -> WXYZ: [x,y,z,w] -> [w,x,y,z]
                rot_interp_wxyz = blend_quat_mujoco(rot_start_wxyz, rot_end_wxyz, alpha)  # Returns WXYZ

                # Interpolate joints
                dof_interp = (1 - alpha) * dof_start + alpha * dof_end

                mj_data.qpos[:3] = pos_interp
                mj_data.qpos[3:7] = rot_interp_wxyz[[1, 2, 3, 0]]  # Convert WXYZ to XYZW for MuJoCo: [w,x,y,z] -> [x,y,z,w]
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
    if use_motion_blocks:
        print(f"Motion type: Motion Blocks (Codebook)")
    else:
        print(f"Motion type: {'VQVAE-Generated' if use_vqvae_motions else 'Original'}")
    print(f"Total motions processed: {len(motion_data_all)}")
    print(f"Total frames: {sum(motion_lengths)}")


if __name__ == "__main__":
    main()


# Usage examples:
# python reference_code/vis_q_mj_bdh_multi_tasks_comp_vqvae_npy.py --robot unitree_g1_kungfu_23dof_bdh --motion-blocks-dir outputs/motion_blocks_npy --block-ids "0"

# python reference_code/vis_q_mj_bdh_multi_tasks_comp_vqvae_npy.py --robot unitree_g1_kungfu_23dof_bdh --motion-files outputs/vqvae_amass_motions_npy/vqvae_motion_000.npy