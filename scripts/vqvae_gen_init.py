import joblib
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import sys

# Local repo imports
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter


def load_config_and_agent(config_path: str, checkpoint_path: str) -> Tuple[Dict, MVQVAEAgent, MotionDataAdapter]:
    config = ConfigLoader().load_config(config_path)
    agent = MVQVAEAgent(config=config)
    adapter = MotionDataAdapter(config)
    if hasattr(adapter, "device"):
        adapter.device = agent.device
    agent.checkpoint_path = checkpoint_path
    return config, agent, adapter


def load_original_pkl(input_pkl_file: str) -> Tuple[Dict, List[str]]:
    motions = joblib.load(input_pkl_file)
    keys = list(motions.keys())
    return motions, keys


def infer_frame_size(adapter: MotionDataAdapter, input_pkl_file: str, motion_ids: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
    mocap_data, end_indices, frame_size = adapter.load_motion_data(input_pkl_file, motion_ids or [0])
    return mocap_data, end_indices, int(frame_size)


def initialize_model(agent: MVQVAEAgent, config: Dict, frame_size: int, checkpoint_path: str) -> None:
    from motion_vqvae.models.models import MotionVQVAE
    from motion_vqvae.models.policy_classifier import PolicyIDClassifier
    from motion_vqvae.models.latent_predictor import LatentPredictor
    
    agent.config["frame_size"] = int(frame_size)
    agent.model = MotionVQVAE(
        agent,
        config["nb_code"],
        config["code_dim"],
        config["output_emb_width"],
        config["down_t"],
        config["stride_t"],
        config["width"],
        config["depth"],
        config["dilation_growth_rate"],
        config["vq_act"],
        config["vq_norm"],
    ).to(agent.device)

    ckpt = torch.load(checkpoint_path, map_location=agent.device)
    agent.model.load_state_dict(ckpt["model"])
    agent.model.eval()

    mean_ckpt = ckpt.get("mean", None)
    std_ckpt = ckpt.get("std", None)
    if mean_ckpt is not None and std_ckpt is not None:
        mean_t = torch.as_tensor(mean_ckpt, dtype=torch.float32, device=agent.device)
        std_t = torch.as_tensor(std_ckpt, dtype=torch.float32, device=agent.device)
        if mean_t.numel() == frame_size and std_t.numel() == frame_size:
            agent.mean = mean_t
            agent.std = std_t
    
    # Load policy classifier if available in checkpoint and config enables it
    if "policy_classifier" in ckpt and config.get("use_policy_loss", False):
        policy_state = ckpt["policy_classifier"]
        # Get num_policies from checkpoint metadata (saved during training)
        num_policies = ckpt.get("num_policies", None)
        if num_policies is None:
            # Fallback: try to infer from state dict (output layer shape)
            layer_keys = [k for k in policy_state.keys() if "classifier" in k and "weight" in k]
            if layer_keys:
                # Find the output layer (smallest out_features among 2D weights)
                candidates = [(k, policy_state[k]) for k in layer_keys if policy_state[k].ndim == 2]
                if candidates:
                    candidates_sorted = sorted(candidates, key=lambda kv: kv[1].shape[0])
                    num_policies = int(candidates_sorted[0][1].shape[0])
            if num_policies is None:
                print("Warning: Could not determine num_policies. Skipping policy classifier loading.")
                num_policies = None
        
        if num_policies is not None:
            # Helper function to filter state dict by shape compatibility
            def filter_state_dict_by_shape(model, state_dict):
                """Filter state dict to only include keys with matching shapes."""
                model_dict = model.state_dict()
                filtered = {}
                for k, v in state_dict.items():
                    if k in model_dict:
                        if model_dict[k].shape == v.shape:
                            filtered[k] = v
                return filtered
            
            # Check if using sequence-wise or chunk-wise classifier
            use_sequence_wise = config.get("policy_use_sequence_wise", False)
            
            if use_sequence_wise:
                # Sequence-wise classifier
                from motion_vqvae.models.sequence_policy_classifier import SequencePolicyIDClassifier
                agent.policy_classifier = SequencePolicyIDClassifier(
                    num_codebooks=config["nb_code"],
                    num_policies=num_policies,
                    code_dim=config["code_dim"],
                    hidden_dim=config.get("policy_classifier_hidden_dim", 256),
                    num_layers=config.get("policy_classifier_layers", 2),
                    dropout=config.get("policy_classifier_dropout", 0.1),
                    architecture=config.get("policy_classifier_architecture", "lstm"),  # lstm, transformer, or cnn1d
                    num_heads=config.get("policy_classifier_num_heads", 8),
                    kernel_size=config.get("policy_classifier_kernel_size", 3),
                ).to(agent.device)
                agent.use_sequence_policy = True
            else:
                # Chunk-wise classifier (original)
                agent.policy_classifier = PolicyIDClassifier(
                    num_codebooks=config["nb_code"],
                    num_policies=num_policies,
                    code_dim=config["code_dim"],
                    hidden_dim=config.get("policy_classifier_hidden_dim", 256),
                    num_layers=config.get("policy_classifier_layers", 2),
                    dropout=config.get("policy_classifier_dropout", 0.1),
                    architecture=config.get("policy_classifier_architecture", "cnn1d"),
                    num_heads=config.get("policy_classifier_num_heads", 8),
                    kernel_size=config.get("policy_classifier_kernel_size", 3),
                ).to(agent.device)
                agent.use_sequence_policy = False
            
            # Try strict loading first
            try:
                agent.policy_classifier.load_state_dict(policy_state, strict=True)
                agent.policy_classifier.eval()
                agent.use_policy_loss = True
                print(f"[Load] Policy classifier loaded successfully (strict mode)")
            except RuntimeError as e:
                # Fallback: try partial loading with shape filtering
                print(f"[Load] Strict loading failed: {e}")
                print(f"[Load] Attempting partial loading (filtering by shape compatibility)...")
                filtered_state = filter_state_dict_by_shape(agent.policy_classifier, policy_state)
                if len(filtered_state) > 0:
                    try:
                        agent.policy_classifier.load_state_dict(filtered_state, strict=False)
                        agent.policy_classifier.eval()
                        agent.use_policy_loss = True
                        print(f"[Load] Policy classifier loaded partially: {len(filtered_state)}/{len(policy_state)} keys matched")
                    except Exception as e2:
                        print(f"[Load] Partial loading also failed: {e2}")
                        print(f"[Load] Skipping policy classifier loading. Model will continue without policy prediction.")
                        agent.policy_classifier = None
                        agent.use_policy_loss = False
                else:
                    print(f"[Load] No compatible keys found. Skipping policy classifier loading.")
                    agent.policy_classifier = None
                    agent.use_policy_loss = False
            
            # Load policy_id_to_class if available
            if "policy_id_to_class" in ckpt and agent.policy_classifier is not None:
                agent.policy_id_to_class = ckpt["policy_id_to_class"]
    
    # Load latent predictor if available in checkpoint and config enables it
    if "latent_predictor" in ckpt and config.get("use_latent_predictor", False):
        # Helper function to filter state dict by shape compatibility
        def filter_state_dict_by_shape(model, state_dict):
            """Filter state dict to only include keys with matching shapes."""
            model_dict = model.state_dict()
            filtered = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        filtered[k] = v
            return filtered
        
        agent.latent_predictor = LatentPredictor(
            num_codebooks=config["nb_code"],
            code_dim=config["code_dim"],
            hidden_dim=config.get("latent_predictor_hidden_dim", 256),
            num_layers=config.get("latent_predictor_layers", 2),
            dropout=config.get("latent_predictor_dropout", 0.1),
            pred_len=config.get("latent_predictor_pred_len", 1),
            architecture=config.get("latent_predictor_architecture", "mlp"),
            num_heads=config.get("latent_predictor_num_heads", 8),
            kernel_size=config.get("latent_predictor_kernel_size", 3),
        ).to(agent.device)
        
        # Try strict loading first
        try:
            agent.latent_predictor.load_state_dict(ckpt["latent_predictor"], strict=True)
            agent.latent_predictor.eval()
            agent.use_latent_predictor = True
            print(f"[Load] Latent predictor loaded successfully (strict mode)")
        except RuntimeError as e:
            # Fallback: try partial loading with shape filtering
            print(f"[Load] Strict loading failed for latent predictor: {e}")
            print(f"[Load] Attempting partial loading (filtering by shape compatibility)...")
            filtered_state = filter_state_dict_by_shape(agent.latent_predictor, ckpt["latent_predictor"])
            if len(filtered_state) > 0:
                try:
                    agent.latent_predictor.load_state_dict(filtered_state, strict=False)
                    agent.latent_predictor.eval()
                    agent.use_latent_predictor = True
                    print(f"[Load] Latent predictor loaded partially: {len(filtered_state)}/{len(ckpt['latent_predictor'])} keys matched")
                except Exception as e2:
                    print(f"[Load] Partial loading also failed: {e2}")
                    print(f"[Load] Skipping latent predictor loading. Model will continue without latent prediction.")
                    agent.latent_predictor = None
                    agent.use_latent_predictor = False
            else:
                print(f"[Load] No compatible keys found. Skipping latent predictor loading.")
                agent.latent_predictor = None
                agent.use_latent_predictor = False


def ensure_stats(agent: MVQVAEAgent, mocap_data_for_stats: torch.Tensor) -> None:
    if getattr(agent, "mean", None) is not None and getattr(agent, "std", None) is not None:
        return
    mocap = mocap_data_for_stats.to(agent.device) if isinstance(mocap_data_for_stats, torch.Tensor) else torch.as_tensor(mocap_data_for_stats, dtype=torch.float32, device=agent.device)
    mean = mocap.mean(dim=0)
    std = mocap.std(dim=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    agent.mean = mean
    agent.std = std


def prepare_single_motion_context(agent: MVQVAEAgent, mocap_features: torch.Tensor) -> None:
    agent.mocap_data = mocap_features.to(agent.device)
    last_idx = mocap_features.shape[0] - 1
    agent.end_indices = torch.as_tensor([last_idx], device=agent.device)
    agent.frame_size = int(mocap_features.shape[1])


# ===== SHARED UTILITY FUNCTIONS =====

def quat_to_euler_xyzw(q):
    """Convert quaternion (XYZW) to Euler angles (roll, pitch, yaw) in radians."""
    x, y, z, w = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def euler_to_quat_xyzw(roll, pitch, yaw):
    """Convert Euler angles (roll, pitch, yaw) in radians to quaternion (XYZW)."""
    # Roll (x-axis rotation)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    # Pitch (y-axis rotation)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    
    # Yaw (z-axis rotation)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    # Quaternion (XYZW format)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    
    return np.array([x, y, z, w], dtype=np.float32)


def parse_motion_ids(spec: str) -> List[int]:
    """Parse '0,3,7-10' into a list of ints."""
    ids: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            ids.extend(list(range(a, b + step, step)))
        else:
            ids.append(int(token))
    return ids


# ===== QUATERNION MATH UTILITIES (XYZW convention) =====

def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply q1*q2 in XYZW convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
    y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
    z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
    return np.array([x, y, z, w], dtype=np.float32)


def quat_rotate_xyzw_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate 3D vector v by quaternion q (XYZW).
    Implements v' = v + 2*q_vec x (q_vec x v + w*v)
    (equivalent to q * [v,0] * q_conj, optimized)
    """
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    # t = 2 * cross(qv, v)
    t = 2.0 * np.cross(qv, v)
    # v' = v + w*t + cross(qv, t)
    vp = v + w * t + np.cross(qv, t)
    return vp.astype(np.float32)


# ===== AMASS FORMAT CONVERSION =====

def convert_to_amass_global_from_local(
    local_features: np.ndarray,
    original_motion: Dict,
    motion_id: int,
) -> Dict:
    """
    Natural local-to-global conversion using proper quaternion integration.
    
    The key insight: local deltas represent velocities in the robot's body frame,
    and we need to integrate them properly using the robot's current orientation.
    """
    mvq = local_features.astype(np.float32)
    T = mvq.shape[0]
    fps = int(original_motion.get("fps", 30))

    # decoder features (G1 format assumption)
    root_deltas_local = mvq[:, 0:4]   # [dx, dy, dz, d_yaw] local per-frame deltas
    dof_positions     = mvq[:, 4:27]  # 23
    dof_velocities    = mvq[:, 27:50] # 23
    contact_mask = np.zeros((T, 2))
    
    # Seed globals from original first frame (AMASS-like compatibility)
    root_trans_offset = np.zeros((T, 3), dtype=np.float32)
    root_rot = np.zeros((T, 4), dtype=np.float32)
    root_trans_offset[0] = original_motion["root_trans_offset"][0].astype(np.float32)
    
    # FIXED: Ensure robot starts upright by normalizing initial orientation
    # Extract only the yaw component from the original orientation to preserve heading
    original_init_quat = original_motion["root_rot"][0].astype(np.float32)
    original_yaw = np.arctan2(2 * (original_init_quat[3] * original_init_quat[2] + original_init_quat[0] * original_init_quat[1]),
                             1 - 2 * (original_init_quat[1] * original_init_quat[1] + original_init_quat[2] * original_init_quat[2]))
    
    # Create upright quaternion with only the original yaw (roll=0, pitch=0)
    root_rot[0] = np.array([
        0.0,  # x component (roll = 0)
        0.0,  # y component (pitch = 0)
        np.sin(original_yaw / 2),  # z component (preserve original yaw)
        np.cos(original_yaw / 2)   # w component (preserve original yaw)
    ], dtype=np.float32)
    
    # Preserve ground offset to prevent ground penetration
    original_z_values = original_motion["root_trans_offset"][:, 2]
    ground_offset = np.min(original_z_values) if len(original_z_values) > 0 else 0.0

    # Track cumulative yaw to avoid discontinuities
    # Use the extracted original yaw as the starting point
    cumulative_yaw = original_yaw

    # Natural integration: treat local deltas as body-frame velocities
    for i in range(1, T):
        # Get current robot orientation
        q_current = root_rot[i - 1]
        
        # Extract local frame deltas (these are per-frame displacements in body frame)
        local_dx = root_deltas_local[i, 0]
        local_dy = root_deltas_local[i, 1] 
        local_dz = root_deltas_local[i, 2]
        local_yaw_vel = root_deltas_local[i, 3]  # Angular velocity around local Z
        
        # Convert local displacement to global displacement
        local_displacement = np.array([local_dx, local_dy, local_dz], dtype=np.float32)
        global_displacement = quat_rotate_xyzw_numpy(q_current, local_displacement)
        
        # Update position
        root_trans_offset[i] = root_trans_offset[i - 1] + global_displacement
        
        # Apply ground offset correction to prevent ground penetration
        if root_trans_offset[i, 2] < ground_offset:
            root_trans_offset[i, 2] = ground_offset
        
        # Handle orientation: apply yaw delta while maintaining natural motion characteristics
        # Use a more conservative approach that preserves original motion better
        if abs(local_yaw_vel) > 1e-8:  # Only apply if there's actual rotation
            # Accumulate yaw continuously to avoid discontinuities
            cumulative_yaw += local_yaw_vel
            
            # Get original orientation for this frame to preserve natural characteristics
            original_quat = original_motion["root_rot"][i].astype(np.float32)
            orig_roll, orig_pitch, orig_yaw = quat_to_euler_xyzw(original_quat)
            
            # Only correct excessive tilts (>15 degrees) to preserve natural motion
            corrected_roll = orig_roll if abs(orig_roll) < np.radians(15) else np.sign(orig_roll) * np.radians(15)
            corrected_pitch = orig_pitch if abs(orig_pitch) < np.radians(15) else np.sign(orig_pitch) * np.radians(15)
            
            # Create quaternion with corrected roll/pitch and updated yaw
            q_new = euler_to_quat_xyzw(corrected_roll, corrected_pitch, cumulative_yaw)
            
            root_rot[i] = q_new
        else:
            # No rotation, apply conservative tilt correction to current orientation
            q_current = root_rot[i - 1]
            curr_roll, curr_pitch, curr_yaw = quat_to_euler_xyzw(q_current)
            
            # Only correct excessive tilts
            corrected_roll = curr_roll if abs(curr_roll) < np.radians(15) else np.sign(curr_roll) * np.radians(15)
            corrected_pitch = curr_pitch if abs(curr_pitch) < np.radians(15) else np.sign(curr_pitch) * np.radians(15)
            
            q_new = euler_to_quat_xyzw(corrected_roll, corrected_pitch, curr_yaw)
            root_rot[i] = q_new

    return {
        "root_trans_offset": root_trans_offset,
        "root_rot": root_rot,
        "pose_aa": original_motion["pose_aa"].astype(np.float32),       # left as-is for compatibility
        "smpl_joints": original_motion["smpl_joints"].astype(np.float32),
        "contact_mask": contact_mask,
        "dof": dof_positions.astype(np.float32),
        "fps": fps,
    }


# ===== GENERIC PKL GENERATION =====

def generate_motion_pkl_files(
    motion_ids: List[int],
    original_keys: List[str],
    motion_generator_func: Callable[[int], Optional[Dict]],
    output_dir: str,
    filename_prefix: str,
    generation_type: str = "Motion"
) -> Tuple[List[str], str]:
    """
    Generic function to generate PKL files for a list of motion IDs.
    
    Args:
        motion_ids: List of motion IDs to process
        original_keys: List of original motion keys for consistency
        motion_generator_func: Function that takes motion_id and returns motion dict or None
        output_dir: Output directory path
        filename_prefix: Prefix for output filenames (e.g., "vqvae_motion_", "amass_gt_motion_")
        generation_type: Type of generation for logging (e.g., "AMASS Format", "AMASS GT Format")
    
    Returns:
        Tuple of (generated_files, output_directory)
    """
    print(f"\n=== Generating {generation_type} PKL Files ===")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_files: List[str] = []

    for motion_id in motion_ids:
        if motion_id < 0 or motion_id >= len(original_keys):
            print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
            continue

        print(f"Processing motion ID {motion_id}...")
        amass_motion = motion_generator_func(motion_id)

        if amass_motion is None:
            print(f"  ❌ Failed to generate motion {motion_id}")
            continue

        # Save with original key for consistency
        original_key = original_keys[motion_id]
        single_motion_dict = {original_key: amass_motion}

        out_path = out_dir / f"{filename_prefix}{motion_id:03d}.pkl"
        joblib.dump(single_motion_dict, out_path)
        generated_files.append(str(out_path))
        print(f"  ✅ Saved: {original_key} -> {out_path}")

    print(f"\nSaved {len(generated_files)} files to: {out_dir}")
    return generated_files, str(out_dir)


