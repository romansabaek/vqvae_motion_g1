#!/usr/bin/env python3
"""
Evaluate LSTM-based policy ID prediction from codebook sequences extracted from .npy motion files.
Uses VQVAE to extract codebook sequences, then LSTM for policy prediction.
"""

import os
import sys
import argparse
import json
import re
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib

# Add parent dir so imports work similarly to your training script
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from motion_vqvae.data.torch_utils import quat_diff, quat_to_exp_map
from motion_vqvae.models.policy_lstm_codebook import PolicyLSTMCodebook
from motion_vqvae.agent import MVQVAEAgent
from scripts.vqvae_gen_init import ensure_stats, initialize_model


# -----------------------------
# Utils (same as eval_policy_id_prediction_vqvae.py)
# -----------------------------
def unique_in_order(arr: np.ndarray) -> List[int]:
    """Unique values in first-occurrence order (NOT sorted)."""
    seen = set()
    out = []
    for x in arr.tolist():
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


def extract_motion_ids_from_filename(file_path: Path) -> Optional[List[int]]:
    """Extract motion IDs from filename: saved_desired_states_8.npy -> [8], saved_desired_states_108_8_fg.npy -> [108, 8]"""
    m = re.search(r'saved_desired_states_((?:\d+_)*\d+)', file_path.stem)
    return [int(x) for x in m.group(1).split('_') if x.isdigit()] if m else None


def parse_policy_ids_from_filename(file_path: Path) -> List[int]:
    """Parse policy IDs from filename: _fg -> [0,1], _gf -> [1,0], else extract f/g chars or default [0,1]"""
    fn = file_path.stem.lower()
    if fn.endswith('_fg'): return [0, 1]
    if fn.endswith('_gf'): return [1, 0]
    methods = [{'f': 0, 'g': 1}[ch] for ch in fn if ch in 'fg']
    return methods if methods else [0, 1]


def unique_in_order(arr: np.ndarray) -> List[int]:
    """Unique values in first-occurrence order (NOT sorted)."""
    seen = set()
    out = []
    for x in arr.tolist():
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


def extract_motion_ids_from_filename(file_path: Path) -> Optional[List[int]]:
    """Extract motion IDs from filename: saved_desired_states_8.npy -> [8], saved_desired_states_108_8_fg.npy -> [108, 8]"""
    m = re.search(r'saved_desired_states_((?:\d+_)*\d+)', file_path.stem)
    return [int(x) for x in m.group(1).split('_') if x.isdigit()] if m else None


def parse_policy_ids_from_filename(file_path: Path) -> List[int]:
    """Parse policy IDs from filename: _fg -> [0,1], _gf -> [1,0], else extract f/g chars or default [0,1]"""
    fn = file_path.stem.lower()
    if fn.endswith('_fg'): return [0, 1]
    if fn.endswith('_gf'): return [1, 0]
    methods = [{'f': 0, 'g': 1}[ch] for ch in fn if ch in 'fg']
    return methods if methods else [0, 1]


def convert_motion_ids_to_policy_ids(motion_ids: np.ndarray, method_ids: List[int]) -> np.ndarray:
    """Map each unique motion_id (in first-occurrence order) to a policy id from method_ids."""
    uniq = unique_in_order(motion_ids)
    motion_id_to_idx = {mid: i for i, mid in enumerate(uniq)}
    policy_ids = np.zeros_like(motion_ids, dtype=np.int64)
    for mid in uniq:
        seq_idx = motion_id_to_idx[mid]
        if 0 <= seq_idx < len(method_ids):
            policy_ids[motion_ids == mid] = int(method_ids[seq_idx])
        else:
            policy_ids[motion_ids == mid] = 0
    return policy_ids


def _smooth_tensor(x: torch.Tensor, box_pts: int) -> torch.Tensor:
    """Moving average smoothing (same logic as your adapter)."""
    box = torch.ones(box_pts, device=x.device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)  # (1, C, T)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q (q is xyzw)."""
    shape0 = q.shape[0]
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape0, 1, 3), v.view(shape0, 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def find_first_switch_index(motion_ids: np.ndarray) -> Optional[int]:
    """Return first index i where motion_ids[i] != motion_ids[i-1]."""
    if len(motion_ids) < 2:
        return None
    idxs = np.where(motion_ids[1:] != motion_ids[:-1])[0]
    return int(idxs[0] + 1) if len(idxs) > 0 else None


# -----------------------------
# Motion feature loader (same as test_baseline_risk_predictor.py)
# -----------------------------
def load_npy_motion_data(
    npy_path: Path,
    device: torch.device = None,
    pkl_policy_ids_per_motion: Optional[List] = None,
    motion_ids_from_filename: Optional[List[int]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Returns:
        motion_features_np: (T, 50)
        policy_ids: (T,) or None
        motion_ids_raw: (T,) raw motion_id column from file (useful for switch diagnostics)
    
    Args:
        motion_ids_from_filename: List of motion IDs extracted from filename (e.g., [108, 8] for sequence files)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.load(npy_path, allow_pickle=True)

    times = data[:, 0]
    root_pos_np = data[:, 1:4]
    root_rot_wxyz_np = data[:, 4:8]
    dof_pos_np = data[:, 8:31]
    motion_ids_raw = data[:, -1].astype(np.int64)

    T = len(times)

    # Try PKL policy IDs for both single-motion and sequence files
    policy_ids = None
    if pkl_policy_ids_per_motion is not None and motion_ids_from_filename is not None:
        unique_motion_ids = unique_in_order(motion_ids_raw)
        raw_to_pkl = {}
        for idx, raw_id in enumerate(unique_motion_ids):
            if idx < len(motion_ids_from_filename):
                pkl_id = motion_ids_from_filename[idx]
                if 0 <= pkl_id < len(pkl_policy_ids_per_motion) and pkl_policy_ids_per_motion[pkl_id] is not None:
                    raw_to_pkl[raw_id] = np.asarray(pkl_policy_ids_per_motion[pkl_id], dtype=np.int64)
        
        if raw_to_pkl:
            seg_starts = {raw_id: next(i for i, rid in enumerate(motion_ids_raw) if rid == raw_id) for raw_id in raw_to_pkl.keys()}
            policy_ids_list = []
            for frame_idx, raw_id in enumerate(motion_ids_raw):
                if raw_id in raw_to_pkl:
                    seq = raw_to_pkl[raw_id]
                    rel_idx = frame_idx - seg_starts.get(raw_id, 0)
                    policy_ids_list.append(int(seq[rel_idx % len(seq)]) if len(seq) > 0 else 0)
                else:
                    policy_ids_list.append(0)
            policy_ids = np.array(policy_ids_list[:T], dtype=np.int64)
            if len(policy_ids) < T:
                policy_ids = np.pad(policy_ids, (0, T - len(policy_ids)), constant_values=int(policy_ids[-1]) if len(policy_ids) > 0 else 0)

    # Fallback for sequence files / when PKL missing
    if policy_ids is None:
        method_ids = parse_policy_ids_from_filename(npy_path)
        policy_ids = convert_motion_ids_to_policy_ids(motion_ids_raw, method_ids)

    fps = 1.0 / np.mean(np.diff(times)) if T > 1 and np.mean(np.diff(times)) > 1e-9 else 30.0
    root_pos = torch.tensor(root_pos_np, dtype=torch.float32, device=device)
    root_rot = torch.tensor(root_rot_wxyz_np[:, [1, 2, 3, 0]], dtype=torch.float32, device=device)  # wxyz -> xyzw
    dof_pos = torch.tensor(dof_pos_np, dtype=torch.float32, device=device)

    def compute_vel(pos, fps_val):
        vel = torch.zeros_like(pos)
        if T > 1:
            vel[:-1] = fps_val * (pos[1:] - pos[:-1])
            vel[-1] = vel[-2]
        return _smooth_tensor(vel, 19)

    root_vel = compute_vel(root_pos, fps)
    root_ang_vel = torch.zeros_like(root_pos)
    if T > 1:
        root_drot = quat_diff(root_rot[:-1], root_rot[1:])
        root_ang_vel[:-1] = fps * quat_to_exp_map(root_drot)
        root_ang_vel[-1] = root_ang_vel[-2]
    root_ang_vel = _smooth_tensor(root_ang_vel, 19)
    dof_vel = compute_vel(dof_pos, fps)

    # local frame velocities
    lin_vel_local = quat_rotate_inverse(root_rot, root_vel)
    ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)

    # assemble features
    motion_features = torch.zeros((T, 50), dtype=torch.float32, device=device)
    motion_features[:, 0:3] = lin_vel_local / fps
    motion_features[:, 3] = ang_vel_local[:, 2] / fps  # approx delta-yaw
    motion_features[:, 4:27] = dof_pos
    motion_features[:, 27:50] = dof_vel

    return motion_features.detach().cpu().numpy(), policy_ids, motion_ids_raw


# -----------------------------
# Codebook extraction from motion features
# -----------------------------
def extract_codebook_from_motion(
    motion_features: np.ndarray,
    vqvae_agent: MVQVAEAgent,
    window_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    """Extract codebook sequence from motion features using VQVAE encoder."""
    device = vqvae_agent.device
    
    # Normalize motion features
    motion_tensor = torch.tensor(motion_features, dtype=torch.float32, device=device)
    motion_norm = (motion_tensor - mean.to(device)) / std.to(device)
    
    T = motion_norm.shape[0]
    if T < window_size:
        return np.array([], dtype=np.int64)
    
    # Create windows
    windows = []
    for i in range(T - window_size + 1):
        windows.append(motion_norm[i:i + window_size])
    
    if not windows:
        return np.array([], dtype=np.int64)
    
    windows_tensor = torch.stack(windows, dim=0)  # (N, W, D)
    
    # Extract codebook indices using VQVAE encoder
    vqvae_agent.model.eval()
    with torch.no_grad():
        code_indices = vqvae_agent.model.encode(windows_tensor)  # (N, num_codebooks_per_window)
    
    # Flatten to 1D sequence: take the first codebook index per window (or use majority vote)
    # For simplicity, we'll use the first codebook index of each window
    if code_indices.dim() > 1:
        code_indices = code_indices[:, 0]  # Take first codebook index per window
    
    return code_indices.detach().cpu().numpy().astype(np.int64)
    """
    Load codebook sequence and policy IDs from CSV file.
    
    Returns:
        codebook_seq: (T,) codebook indices
        policy_ids: (T,) policy IDs
        times: (T,) timesteps (if available)
    """
    df = pd.read_csv(csv_path)
    
    # Find codebook column
    codebook_col = None
    for col in ['codebook_id', 'codebook_idx', 'codebook', 'codebook_index']:
        if col in df.columns:
            codebook_col = col
            break
    
    if codebook_col is None:
        raise ValueError(f"No codebook column found in {csv_path.name}")
    
    # Find policy_id column
    policy_col = None
    for col in ['policy_id', 'gt_policy_id', 'policy']:
        if col in df.columns:
            policy_col = col
            break
    
    if policy_col is None:
        raise ValueError(f"No policy_id column found in {csv_path.name}")
    
    # Extract sequences
    codebook_seq = df[codebook_col].values.astype(np.int64)
    policy_ids = df[policy_col].values.astype(np.int64)
    
    # Extract times if available
    if 'timestep' in df.columns or 'time' in df.columns:
        time_col = 'timestep' if 'timestep' in df.columns else 'time'
        times = df[time_col].values
    else:
        times = np.arange(len(codebook_seq)) / 30.0  # Default 30 FPS
    
    return codebook_seq, policy_ids, times


# -----------------------------
# Evaluator
# -----------------------------
class LSTMPolicyIDEvaluator:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        vqvae_checkpoint_path: Optional[str],
        npy_data_dir: str,
        eval_stride: Optional[int] = None,
        input_pkl: Optional[str] = None,
        boundary_radius_frames: int = 32,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.npy_data_dir = Path(npy_data_dir)
        self.eval_stride = eval_stride
        self.input_pkl = input_pkl
        self.boundary_radius_frames = int(boundary_radius_frames)

        # Load config
        config_loader = ConfigLoader(config_path)
        cfg = config_loader.to_dict()

        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Load LSTM checkpoint to get model config
        self._ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model config from checkpoint
        checkpoint_config = self._ckpt.get('config', {})
        num_policies = self._ckpt.get('num_policies', 2)
        codebook_size = self._ckpt.get('codebook_size', 512)
        window_size = checkpoint_config.get('window_size', cfg.get('window_size', 32))
        self.window_size = int(window_size)

        W = self.window_size
        self.eval_stride = max(1, W // 2) if self.eval_stride is None else self.eval_stride
        print(f"[Eval] device={self.device}, window_size={W}, eval_stride={self.eval_stride}")

        # Initialize VQVAE agent for codebook extraction
        self.vqvae_agent = MVQVAEAgent(cfg)
        self.vqvae_agent.device = self.device
        self.vqvae_agent.config['frame_size'] = 50  # Set frame_size before creating model
        
        # Initialize VQVAE model structure
        from motion_vqvae.models.models import MotionVQVAE
        self.vqvae_agent.model = MotionVQVAE(
            self.vqvae_agent,
            cfg['nb_code'],
            cfg['code_dim'],
            cfg['output_emb_width'],
            cfg['down_t'],
            cfg['stride_t'],
            cfg['width'],
            cfg['depth'],
            cfg['dilation_growth_rate'],
            cfg.get('vq_act', 'relu'),
            cfg.get('vq_norm', None)
        ).to(self.device)
        
        # Determine VQVAE checkpoint path
        if vqvae_checkpoint_path and Path(vqvae_checkpoint_path).exists():
            vqvae_ckpt = vqvae_checkpoint_path
        else:
            # Try to find VQVAE checkpoint in common locations
            possible_paths = [
                checkpoint_path,  # Same checkpoint might contain VQVAE weights
                Path(checkpoint_path).parent.parent / "vqvae" / "best_model.ckpt",
                Path("./checkpoints/vqvae/best_model.ckpt"),
                Path("./outputs/vqvae/best_model.ckpt"),
            ]
            vqvae_ckpt = None
            for path in possible_paths:
                if Path(path).exists():
                    vqvae_ckpt = str(path)
                    break
            
            if vqvae_ckpt is None:
                # Use the same checkpoint path (might contain VQVAE weights)
                vqvae_ckpt = checkpoint_path
                print(f"[Eval] Warning: VQVAE checkpoint not found in common locations, trying: {vqvae_ckpt}")
        
        # Load VQVAE checkpoint
        print(f"[Eval] Loading VQVAE from: {vqvae_ckpt}")
        try:
            if Path(vqvae_ckpt).exists():
                self.vqvae_agent.load(vqvae_ckpt)
                print(f"[Eval] Successfully loaded VQVAE from {vqvae_ckpt}")
            else:
                raise FileNotFoundError(f"VQVAE checkpoint not found: {vqvae_ckpt}")
        except Exception as e:
            print(f"[Eval] Error: Failed to load VQVAE from {vqvae_ckpt}: {e}")
            print(f"[Eval] Please provide a valid VQVAE checkpoint with --vqvae_checkpoint")
            raise
        
        # Load normalization stats from PKL if available
        self._cached_policy_ids_per_motion = None
        self._pkl_normalization_stats = None
        if self.input_pkl:
            self._load_policy_ids_from_pkl()

        # Create LSTM model
        self.model = PolicyLSTMCodebook(
            codebook_size=codebook_size,
            embedding_dim=checkpoint_config.get('lstm_embedding_dim', cfg.get('lstm_embedding_dim', 64)),
            hidden_dim=checkpoint_config.get('lstm_hidden_dim', cfg.get('lstm_hidden_dim', 256)),
            num_layers=checkpoint_config.get('lstm_num_layers', cfg.get('lstm_num_layers', 2)),
            num_policies=num_policies,
            dropout=checkpoint_config.get('lstm_dropout', cfg.get('lstm_dropout', 0.1)),
            bidirectional=checkpoint_config.get('lstm_bidirectional', cfg.get('lstm_bidirectional', False)),
        ).to(self.device)

        # Load LSTM model weights
        self.model.load_state_dict(self._ckpt['model_state_dict'])
        self.model.eval()
        print(f"[Eval][LSTM] Loaded model. num_policies={num_policies}, codebook_size={codebook_size}")

    def _load_policy_ids_from_pkl(self, max_motions_for_stats: int = 1000):
        try:
            print(f"[Eval] Loading PKL: {self.input_pkl}")
            motion_data_dict = joblib.load(self.input_pkl)
            self._cached_policy_ids_per_motion = [
                np.asarray(md["policy_id"], dtype=np.int64) if md.get("policy_id") is not None else None
                for md in motion_data_dict.values()
            ]
            num_with = sum(1 for x in self._cached_policy_ids_per_motion if x is not None)
            print(f"[Eval] Loaded policy IDs for {num_with}/{len(self._cached_policy_ids_per_motion)} motions")

            subset_motion_ids = list(range(min(max_motions_for_stats, len(motion_data_dict))))
            temp_adapter = MotionDataAdapter({'device': str(self.device)})
            mocap_data, _, _ = temp_adapter.load_motion_data(self.input_pkl, subset_motion_ids)
            subset_t = mocap_data.to(self.device) if isinstance(mocap_data, torch.Tensor) else torch.tensor(mocap_data, device=self.device)
            self._pkl_normalization_stats = (subset_t.mean(dim=0), subset_t.std(dim=0).clamp_min(1e-8))
            print(f"[Eval] PKL stats: frames={subset_t.shape[0]}, feat_dim={subset_t.shape[-1]}")
        except Exception as e:
            print(f"[Eval] Warning: PKL load failed: {e}")
            import traceback
            traceback.print_exc()
            self._cached_policy_ids_per_motion = None
            self._pkl_normalization_stats = None

    def _make_overlapped_windows_full_only(self, codebook_seq: np.ndarray) -> Tuple[torch.Tensor, List[int]]:
        """Returns (N, W) codebook windows and list of start indices."""
        W = self.window_size
        S = int(self.eval_stride)
        T = len(codebook_seq)
        if T < W:
            return torch.empty((0, W), device=self.device, dtype=torch.long), []
        starts = list(range(0, T - W + 1, S))
        windows = torch.stack([torch.from_numpy(codebook_seq[s:s + W]).long() for s in starts], dim=0).to(self.device)
        return windows, starts
    
    def _map_codebook_to_frames(self, codebook_seq: np.ndarray, num_frames: int) -> np.ndarray:
        """Map window-level codebook sequence to frame-level (for voting)."""
        # Each codebook index corresponds to a window, so we need to map it to frames
        # Simple approach: each codebook index covers window_size frames
        codebook_per_frame = np.zeros(num_frames, dtype=np.int64)
        if len(codebook_seq) == 0:
            return codebook_per_frame
        
        # Each codebook index covers window_size frames
        frames_per_codebook = max(1, num_frames // len(codebook_seq))
        for i, code_idx in enumerate(codebook_seq):
            start_frame = i * frames_per_codebook
            end_frame = min((i + 1) * frames_per_codebook, num_frames)
            codebook_per_frame[start_frame:end_frame] = code_idx
        
        # Fill remaining frames with last codebook index
        if codebook_per_frame[-1] == 0 and len(codebook_seq) > 0:
            codebook_per_frame[codebook_per_frame == 0] = codebook_seq[-1]
        
        return codebook_per_frame

    def _vote_windows_to_frames(self, pred_per_window: np.ndarray, starts: List[int], T: int) -> np.ndarray:
        """Simple uniform vote of each window label over its W frames."""
        W = self.window_size
        offset = -int(pred_per_window.min()) if pred_per_window.min() < 0 else 0
        num_policies = int(pred_per_window.max()) + 1 + offset
        vote_counts = torch.zeros((T, num_policies), device=self.device, dtype=torch.long)
        for i, s in enumerate(starts):
            vote_counts[s:s + W, int(pred_per_window[i]) + offset] += 1
        pred_per_frame = torch.argmax(vote_counts, dim=1).detach().cpu().numpy().astype(np.int64) - offset
        no_vote = (vote_counts.sum(dim=1) == 0).detach().cpu().numpy()
        if no_vote.any():
            pred_per_frame[no_vote] = int(pred_per_window[-1])
        return pred_per_frame

    def _boundary_breakdown(self, gt: np.ndarray, pred: np.ndarray, motion_ids_raw: np.ndarray, radius: int) -> Dict[str, float]:
        """Compute accuracy near/outside switch boundary."""
        switch_idx = find_first_switch_index(motion_ids_raw)
        if switch_idx is None:
            return {}
        a, b = max(0, switch_idx - radius), min(len(gt), switch_idx + radius + 1)
        near_mask = np.zeros(len(gt), dtype=bool)
        near_mask[a:b] = True
        return {
            "switch_idx": float(switch_idx),
            "near_acc": float((pred[near_mask] == gt[near_mask]).mean()) if near_mask.any() else float("nan"),
            "out_acc": float((pred[~near_mask] == gt[~near_mask]).mean()) if (~near_mask).any() else float("nan"),
            "near_count": float(near_mask.sum()),
            "out_count": float((~near_mask).sum()),
        }

    def _plot_policy_id_tracking(self, gt: np.ndarray, pred: np.ndarray, filename: str, out_dir: Path, acc: float):
        T = len(gt)
        frames = np.arange(T)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        all_pids = np.unique(np.concatenate([gt, pred]))
        ax1 = axes[0]
        ax1.scatter(frames, gt, s=12, alpha=0.8)
        ax1.set_title(f"GT policy IDs ({filename})")
        ax1.set_ylabel("Policy ID")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        correct = (gt == pred)
        ax2.scatter(frames[correct], pred[correct], s=12, alpha=0.8, marker='o')
        ax2.scatter(frames[~correct], pred[~correct], s=18, alpha=0.9, marker='x')
        ax2.set_title(f"Pred policy IDs (Acc={acc:.2%})")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Policy ID")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = out_dir / f"{filename}_policy_id_tracking.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def evaluate_file(self, npy_file: Path, output_dir: Optional[Path] = None) -> Optional[Dict[str, object]]:
        print("\n" + "=" * 80)
        print(f"Evaluating: {npy_file.name}")
        print("=" * 80)

        motion_ids_from_filename = extract_motion_ids_from_filename(npy_file)
        try:
            feats_np, policy_ids, motion_ids_raw = load_npy_motion_data(
                npy_file, device=self.device,
                pkl_policy_ids_per_motion=self._cached_policy_ids_per_motion,
                motion_ids_from_filename=motion_ids_from_filename
            )
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            import traceback
            traceback.print_exc()
            return None

        if policy_ids is None:
            print("No GT policy IDs available -> skipping.")
            return None

        T = feats_np.shape[0]
        print(f"Loaded {T} frames, policy IDs unique: {np.unique(policy_ids).tolist()}")

        # Get normalization stats
        motion_tensor = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        mean, std = self._pkl_normalization_stats if self._pkl_normalization_stats else (motion_tensor.mean(dim=0), motion_tensor.std(dim=0).clamp_min(1e-8))
        
        # Ensure VQVAE stats are set
        if self._pkl_normalization_stats:
            self.vqvae_agent.mean = mean.to(self.device)
            self.vqvae_agent.std = std.to(self.device)
        else:
            ensure_stats(self.vqvae_agent, motion_tensor.to(self.device))
            mean = self.vqvae_agent.mean
            std = self.vqvae_agent.std

        # Extract codebook sequence from motion features
        codebook_seq = extract_codebook_from_motion(
            feats_np, self.vqvae_agent, self.window_size, mean, std
        )
        
        if len(codebook_seq) == 0:
            print("Failed to extract codebook sequence -> skipping.")
            return None
        
        print(f"Extracted {len(codebook_seq)} codebook indices, range: [{codebook_seq.min()}, {codebook_seq.max()}]")

        # Create windows from codebook sequence
        windows, starts = self._make_overlapped_windows_full_only(codebook_seq)
        if windows.shape[0] == 0:
            print("Codebook sequence shorter than window_size -> skipping.")
            return None

        # LSTM forward pass
        self.model.eval()
        with torch.no_grad():
            logits = self.model(windows)  # (N, num_policies)
            pred_per_window = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)

        # Map window predictions to frames
        pred_per_frame = self._vote_windows_to_frames(pred_per_window, starts, len(codebook_seq))
        
        # Map codebook-level predictions to motion frame level
        # Since codebook_seq is window-level, we need to map it to frame-level policy predictions
        # Simple approach: use the codebook sequence length to map predictions
        if len(pred_per_frame) < T:
            # Interpolate/extend predictions to match T frames
            pred_per_frame_full = np.zeros(T, dtype=np.int64)
            frames_per_codebook = max(1, T // len(pred_per_frame))
            for i, pred in enumerate(pred_per_frame):
                start_frame = i * frames_per_codebook
                end_frame = min((i + 1) * frames_per_codebook, T)
                pred_per_frame_full[start_frame:end_frame] = pred
            if pred_per_frame_full[-1] == 0 and len(pred_per_frame) > 0:
                pred_per_frame_full[pred_per_frame_full == 0] = pred_per_frame[-1]
            pred_per_frame = pred_per_frame_full
        
        gt = policy_ids[:T].astype(np.int64)
        acc = float((pred_per_frame == gt).mean())

        conf = {}
        for g, p in zip(gt, pred_per_frame):
            conf[(int(g), int(p))] = conf.get((int(g), int(p)), 0) + 1
        confusion = [(k[0], k[1], v) for k, v in sorted(conf.items(), key=lambda x: -x[1])]
        bd = self._boundary_breakdown(gt, pred_per_frame, motion_ids_raw, self.boundary_radius_frames)

        print(f"  - Policy ID Accuracy: {acc:.2%}")
        if bd:
            print(f"  - Switch idx: {int(bd['switch_idx'])}, near(+/-{self.boundary_radius_frames}) acc={bd['near_acc']:.2%}, outside acc={bd['out_acc']:.2%}")
        print("  - Confusion (top10):")
        for (tg, tp, c) in confusion[:10]:
            print(f"    True={tg}, Pred={tp}: {c}")

        return {"policy_acc": acc, "confusion": confusion, "gt_policy_ids_per_frame": gt,
                "pred_policy_ids_per_frame": pred_per_frame, "boundary": bd}

    def evaluate_all_files(self, output_dir: Optional[str] = None) -> Dict[str, Dict[str, object]]:
        npy_files = sorted(self.npy_data_dir.glob("*.npy"))
        if not npy_files:
            print(f"No .npy found in {self.npy_data_dir}")
            return {}

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        all_results = {f.name: r for f in npy_files if (r := self.evaluate_file(f, out_path)) is not None}

        if all_results:
            accs = [v["policy_acc"] for v in all_results.values()]
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Total files: {len(all_results)}")
            print(f"Mean acc: {np.mean(accs):.2%}, Std: {np.std(accs):.2%}, Min: {np.min(accs):.2%}, Max: {np.max(accs):.2%}")

            if out_path:
                # Save summary CSV
                csv_path = out_path / "policy_id_prediction_results.csv"
                with open(csv_path, "w", newline="") as fp:
                    w = csv.writer(fp)
                    w.writerow(["filename", "acc", "switch_idx", f"near_acc(+/-{self.boundary_radius_frames})", "out_acc"])
                    for fn, res in sorted(all_results.items()):
                        bd = res.get("boundary", {}) or {}
                        w.writerow([fn, f"{res['policy_acc']:.6f}",
                                   "" if not bd else int(bd["switch_idx"]),
                                   "" if not bd else f"{bd['near_acc']:.6f}",
                                   "" if not bd else f"{bd['out_acc']:.6f}"])
                print(f"Saved summary CSV: {csv_path}")

        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="LSTM checkpoint path")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None, help="VQVAE checkpoint path for codebook extraction (optional, will search common locations if not provided)")
    parser.add_argument("--npy_data_dir", type=str, required=True, help="Directory containing .npy motion files")
    parser.add_argument("--input_pkl", type=str, default=None, help="PKL file for policy IDs (optional)")
    parser.add_argument("--file", type=str, default=None, help="Specific .npy file to evaluate (relative to npy_data_dir)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_lstm_policy_id_codebook")
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--boundary_radius", type=int, default=32, help="frames around first switch to compute near/out accuracy")
    args = parser.parse_args()
    
    evaluator = LSTMPolicyIDEvaluator(
        config_path=args.config, checkpoint_path=args.checkpoint, vqvae_checkpoint_path=args.vqvae_checkpoint,
        npy_data_dir=args.npy_data_dir, eval_stride=args.eval_stride, input_pkl=args.input_pkl,
        boundary_radius_frames=args.boundary_radius
    )

    if args.file:
        npy_file = Path(args.npy_data_dir) / args.file
        if not npy_file.exists():
            print(f"File not found: {npy_file}")
            return
        evaluator.evaluate_file(npy_file, Path(args.output_dir))
    else:
        evaluator.evaluate_all_files(args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()


'''
# Example usage for LSTM policy ID predictor evaluation (codebook-based):

# Evaluate all .npy files in directory
python scripts/test_baseline_risk_predictor_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint ./checkpoints/lstm_policy_predictor_codebook/best_model.ckpt \
  --vqvae_checkpoint ./checkpoints/vqvae/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_training \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_lstm_policy_id_codebook

# Evaluate single .npy file
python scripts/test_baseline_risk_predictor_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint ./checkpoints/lstm_policy_predictor_codebook/best_model.ckpt \
  --vqvae_checkpoint ./checkpoints/vqvae/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_testing \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_lstm_policy_id_codebook_testing

'''
