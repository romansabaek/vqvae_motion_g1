#!/usr/bin/env python3
"""
Evaluate ONLY policy ID prediction from .npy motion files.

Key fixes vs your modified version:
1) Windowing bug fix: create ONLY full windows (no partial windows), so prediction/windows indices align.
2) Motion-id mapping fix: map by first-occurrence order (NOT np.unique sorted), which is safer for sequences.
3) Adds boundary diagnostics for sequence files: accuracy near switch vs outside.
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

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from motion_vqvae.data.torch_utils import quat_diff, quat_to_exp_map
from motion_vqvae.models.policy_classifier import PolicyIDClassifier
from motion_vqvae.models.models import MotionVQVAE


# -----------------------------
# Utils
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


def convert_motion_ids_to_policy_ids(motion_ids: np.ndarray, method_ids: List[int]) -> np.ndarray:
    """
    Map each unique motion_id (in first-occurrence order) to a policy id from method_ids.
    This is safer than np.unique(sorted) for sequences.
    """
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
# Feature loader (aligned with training)
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

    # 2) Fallback for sequence files / when PKL missing
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
# Evaluator
# -----------------------------
class SimplePolicyIDEvaluator:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
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

        # Create agent
        self.agent = MVQVAEAgent(config=cfg, device=self.device)
        self.agent.config['frame_size'] = MotionDataAdapter.TOTAL_FRAME_SIZE  # 50

        # Init model + load checkpoint
        self.agent.model = MotionVQVAE(
            self.agent,
            self.agent.config['nb_code'],
            self.agent.config['code_dim'],
            self.agent.config['output_emb_width'],
            self.agent.config['down_t'],
            self.agent.config['stride_t'],
            self.agent.config['width'],
            self.agent.config['depth'],
            self.agent.config['dilation_growth_rate'],
            self.agent.config['vq_act'],
            self.agent.config['vq_norm']
        ).to(self.device)

        self._ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.agent.load(checkpoint_path)

        W = int(self.agent.config.get("window_size", 32))
        self.eval_stride = max(1, W // 2) if self.eval_stride is None else self.eval_stride
        self.agent.config["eval_stride"] = int(self.eval_stride)
        print(f"[Eval] device={self.device}, window_size={W}, eval_stride={self.eval_stride}")

        self._cached_policy_ids_per_motion = None
        self._pkl_normalization_stats = None
        if self.input_pkl:
            self._load_policy_ids_from_pkl()
        self._try_load_policy_classifier_from_ckpt()

    def _extract_policy_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(self._ckpt, dict) and "policy_classifier" in self._ckpt:
            sd = self._ckpt["policy_classifier"]
            if isinstance(sd, dict) and len(sd) > 0:
                return sd
        if isinstance(self._ckpt, dict) and isinstance(self._ckpt.get("state_dict"), dict):
            pol_keys = [k for k in self._ckpt["state_dict"].keys() if k.startswith("policy_classifier.")]
            return {k[len("policy_classifier."):]: self._ckpt["state_dict"][k] for k in pol_keys} if pol_keys else None
        return None

    def _infer_num_policies_from_state_dict(self, pol_sd: Dict[str, torch.Tensor]) -> Optional[int]:
        candidates = [(k, v) for k, v in pol_sd.items() if k.endswith(".weight") and v.ndim == 2]
        return int(sorted(candidates, key=lambda kv: kv[1].shape[0])[0][1].shape[0]) if candidates else None

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
            temp_adapter = MotionDataAdapter(self.agent.config)
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

    def _try_load_policy_classifier_from_ckpt(self):
        pol_sd = self._extract_policy_state_dict()
        if pol_sd is None:
            print("[Eval][Policy] No policy classifier weights found.")
            self.agent.policy_classifier = None
            return

        num_policies = self._infer_num_policies_from_state_dict(pol_sd)
        if num_policies is None:
            print("[Eval][Policy] Could not infer num_policies.")
            self.agent.policy_classifier = None
            return

        cfg = self.agent.config
        model = PolicyIDClassifier(
            num_codebooks=int(cfg["nb_code"]), num_policies=int(num_policies), code_dim=int(cfg["code_dim"]),
            hidden_dim=int(cfg.get("policy_classifier_hidden_dim", 256)), num_layers=int(cfg.get("policy_classifier_layers", 2)),
            dropout=float(cfg.get("policy_classifier_dropout", 0.1)), architecture=str(cfg.get("policy_classifier_architecture", "cnn1d")),
            num_heads=int(cfg.get("policy_classifier_num_heads", 8)), kernel_size=int(cfg.get("policy_classifier_kernel_size", 3)),
        ).to(self.device)

        try:
            model.load_state_dict(pol_sd, strict=True)
            model.eval()
            self.agent.policy_classifier = model
            print(f"[Eval][Policy] Loaded classifier. num_policies={num_policies}")
        except Exception as e:
            print(f"[Eval][Policy] Load failed: {e}")
            self.agent.policy_classifier = None

    # -------- FIXED windowing: ONLY full windows --------
    def _make_overlapped_windows_full_only(self, seq: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Returns (N, W, D) full windows only and list of start indices."""
        W = int(self.agent.config.get("window_size", 32))
        S = int(self.agent.config.get("eval_stride", max(1, W // 2)))
        seq_norm = (seq.to(self.device) - mean.to(self.device)) / std.to(self.device)
        T = int(seq_norm.shape[0])
        if T < W:
            return torch.empty((0, W, seq_norm.shape[1]), device=self.device, dtype=seq_norm.dtype), []
        starts = list(range(0, T - W + 1, S))
        return torch.stack([seq_norm[s:s + W] for s in starts], dim=0), starts

    def _vote_windows_to_frames(self, pred_per_window: np.ndarray, starts: List[int], T: int) -> np.ndarray:
        """Simple uniform vote of each window label over its W frames."""
        W = int(self.agent.config.get("window_size", 32))
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

        if policy_ids is None or self.agent.policy_classifier is None:
            print("No GT policy IDs or classifier available -> skipping.")
            return None

        T = feats_np.shape[0]
        print(f"Loaded {T} frames, policy IDs unique: {np.unique(policy_ids).tolist()}")

        motion_tensor = torch.tensor(feats_np, dtype=torch.float32, device=self.device)
        mean, std = self._pkl_normalization_stats if self._pkl_normalization_stats else (motion_tensor.mean(dim=0), motion_tensor.std(dim=0).clamp_min(1e-8))

        windows, starts = self._make_overlapped_windows_full_only(motion_tensor, mean, std)
        if windows.shape[0] == 0:
            print("Sequence shorter than window_size -> skipping.")
            return None

        self.agent.model.eval()
        with torch.no_grad():
            codebook_windows = self.agent.model.encode(windows).long()
            pred_per_window = torch.argmax(self.agent.policy_classifier(codebook_windows), dim=1).detach().cpu().numpy().astype(np.int64)

        pred_per_frame = self._vote_windows_to_frames(pred_per_window, starts, T)
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

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create separate folders for images and CSV
            images_dir = output_dir / "images"
            csv_dir = output_dir / "csv_data"
            images_dir.mkdir(parents=True, exist_ok=True)
            csv_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plot in images folder
            plot_path = self._plot_policy_id_tracking(gt, pred_per_frame, npy_file.stem, images_dir, acc)
            print(f"Saved plot: {plot_path}")
            
            # Save CSV with timestep, GT policy ID, and predicted policy ID in csv_data folder
            csv_path = csv_dir / f"{npy_file.stem}_policy_ids.csv"
            
            # Load timesteps from the npy file
            data = np.load(npy_file, allow_pickle=True)
            times = data[:T, 0]  # Only take T frames
            
            with open(csv_path, "w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["timestep", "policy_id", "gt_policy_id"])
                for t, gt_pid, pred_pid in zip(times, gt, pred_per_frame):
                    w.writerow([f"{t:.6f}", int(pred_pid), int(gt_pid)])
            print(f"Saved CSV: {csv_path}")

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
                # Save summary CSV in csv_data folder
                csv_dir = out_path / "csv_data"
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_path = csv_dir / "policy_id_prediction_results.csv"
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
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npy_data_dir", type=str, required=True)
    parser.add_argument("--input_pkl", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./evaluation_policy_id")
    parser.add_argument("--eval_stride", type=int, default=None)
    parser.add_argument("--boundary_radius", type=int, default=32, help="frames around first switch to compute near/out accuracy")
    args = parser.parse_args()

    evaluator = SimplePolicyIDEvaluator(
        config_path=args.config, checkpoint_path=args.checkpoint, npy_data_dir=args.npy_data_dir,
        eval_stride=args.eval_stride, input_pkl=args.input_pkl, boundary_radius_frames=args.boundary_radius
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

# Evaluate all files in directory
python scripts/eval_policy_id_prediction_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_testing\
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_policy_id_sequence_testing


python scripts/eval_policy_id_prediction_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_training \
  --file saved_desired_states_108_8_fg.npy \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_policy_id_amass


# Evaluate single file (with PKL for GT policy IDs)
python scripts/eval_policy_id_prediction_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy \
  --file saved_desired_states_8.npy \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_policy_id

# Evaluate all files with motion IDs 0-100 (with PKL for GT policy IDs)
python scripts/eval_policy_id_prediction_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_policy_id \
  --min_motion_id 0 \
  --max_motion_id 10


# Evaluate all files in each_motion_npy (with PKL for GT policy IDs)
python scripts/eval_policy_id_prediction_vqvae.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --npy_data_dir /home/baekdh/dh_workspace/data_deploy/deploy_pkl/each_motion_npy \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --output_dir ./evaluation_policy_id

  '''