#!/usr/bin/env python3
"""
Real-time (streaming) Policy-ID prediction evaluation using VQ-VAE encoder + PolicyIDClassifier.

Adds:
- total accuracy vs GT (GT from PKL if provided; else from filename policy ids + raw motion id segments)
- boundary diagnostics (near/outside first switch)
- supports single file (--demo_npy) and directory mode (--npy_data_dir)
- optional CSV export (--output_dir)

References:
- VQ-VAE: https://arxiv.org/abs/1711.00937
"""

import os
import sys
import re
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
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
# Helpers (GT mapping) - aligned to your offline evaluator logic
# -----------------------------
def unique_in_order(arr: np.ndarray) -> List[int]:
    seen = set()
    out = []
    for x in arr.tolist():
        xi = int(x)
        if xi not in seen:
            seen.add(xi)
            out.append(xi)
    return out


def extract_motion_ids_from_filename(file_path: Path) -> Optional[List[int]]:
    m = re.search(r"saved_desired_states_((?:\d+_)*\d+)", file_path.stem)
    return [int(x) for x in m.group(1).split("_") if x.isdigit()] if m else None


def parse_policy_ids_from_filename(file_path: Path) -> List[int]:
    fn = file_path.stem.lower()
    if fn.endswith("_fg"):
        return [0, 1]
    if fn.endswith("_gf"):
        return [1, 0]
    methods = [{"f": 0, "g": 1}[ch] for ch in fn if ch in "fg"]
    return methods if methods else [0, 1]


def convert_motion_ids_to_policy_ids(motion_ids: np.ndarray, method_ids: List[int]) -> np.ndarray:
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


def find_first_switch_index(motion_ids: np.ndarray) -> Optional[int]:
    if len(motion_ids) < 2:
        return None
    idxs = np.where(motion_ids[1:] != motion_ids[:-1])[0]
    return int(idxs[0] + 1) if len(idxs) > 0 else None


def compute_boundary_breakdown(gt: np.ndarray, pred: np.ndarray, motion_ids_raw: np.ndarray, radius: int) -> Dict[str, float]:
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


# -----------------------------
# Real-time math utils
# -----------------------------
def quat_rotate_inverse(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q (q is xyzw). Shapes: (4,), (3,) -> (3,)"""
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * (2.0 * q_w)
    c = q_vec * (2.0 * torch.dot(q_vec, v))
    return a - b + c


@dataclass
class FrameInput:
    t: float
    root_pos: np.ndarray          # (3,)
    root_rot_wxyz: np.ndarray     # (4,)
    dof_pos: np.ndarray           # (23,)


class CausalFeatureExtractor:
    """
    Causal version of your 50D features.
    Uses backward difference + EMA smoothing (no future frames).
    """
    def __init__(self, device: torch.device, dof_dim: int = 23, ema_alpha: float = 0.2, default_fps: float = 30.0):
        self.device = device
        self.dof_dim = int(dof_dim)
        self.ema_alpha = float(ema_alpha)
        self.default_fps = float(default_fps)

        self._prev_t: Optional[float] = None
        self._prev_root_pos: Optional[torch.Tensor] = None
        self._prev_root_rot_xyzw: Optional[torch.Tensor] = None
        self._prev_dof_pos: Optional[torch.Tensor] = None

        self._ema_root_vel_world: Optional[torch.Tensor] = None
        self._ema_root_ang_vel_world: Optional[torch.Tensor] = None
        self._ema_dof_vel: Optional[torch.Tensor] = None

    def reset(self):
        self._prev_t = None
        self._prev_root_pos = None
        self._prev_root_rot_xyzw = None
        self._prev_dof_pos = None
        self._ema_root_vel_world = None
        self._ema_root_ang_vel_world = None
        self._ema_dof_vel = None

    def _ema(self, prev: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if prev is None:
            return x
        a = self.ema_alpha
        return (1.0 - a) * prev + a * x

    def step(self, fin: FrameInput) -> Tuple[torch.Tensor, float]:
        t = float(fin.t)
        root_pos = torch.tensor(fin.root_pos, dtype=torch.float32, device=self.device)

        wxyz = torch.tensor(fin.root_rot_wxyz, dtype=torch.float32, device=self.device)
        root_rot_xyzw = wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw

        dof_pos = torch.tensor(fin.dof_pos, dtype=torch.float32, device=self.device)

        if self._prev_t is None:
            dt = 1.0 / self.default_fps
        else:
            dt = max(1e-6, t - self._prev_t)
        fps = float(1.0 / dt) if dt > 0 else self.default_fps

        if self._prev_root_pos is None:
            root_vel_world = torch.zeros(3, device=self.device)
        else:
            root_vel_world = (root_pos - self._prev_root_pos) * fps

        if self._prev_root_rot_xyzw is None:
            root_ang_vel_world = torch.zeros(3, device=self.device)
        else:
            q_prev = self._prev_root_rot_xyzw.unsqueeze(0)
            q_curr = root_rot_xyzw.unsqueeze(0)
            drot = quat_diff(q_prev, q_curr)               # (1,4)
            w = quat_to_exp_map(drot).squeeze(0) * fps     # (3,)
            root_ang_vel_world = w

        if self._prev_dof_pos is None:
            dof_vel = torch.zeros(self.dof_dim, device=self.device)
        else:
            dof_vel = (dof_pos - self._prev_dof_pos) * fps

        self._ema_root_vel_world = self._ema(self._ema_root_vel_world, root_vel_world)
        self._ema_root_ang_vel_world = self._ema(self._ema_root_ang_vel_world, root_ang_vel_world)
        self._ema_dof_vel = self._ema(self._ema_dof_vel, dof_vel)

        lin_vel_local = quat_rotate_inverse(root_rot_xyzw, self._ema_root_vel_world)
        ang_vel_local = quat_rotate_inverse(root_rot_xyzw, self._ema_root_ang_vel_world)

        feat = torch.zeros(50, dtype=torch.float32, device=self.device)
        feat[0:3] = lin_vel_local / max(fps, 1e-6)
        feat[3] = ang_vel_local[2] / max(fps, 1e-6)
        feat[4:27] = dof_pos
        feat[27:50] = self._ema_dof_vel

        self._prev_t = t
        self._prev_root_pos = root_pos
        self._prev_root_rot_xyzw = root_rot_xyzw
        self._prev_dof_pos = dof_pos

        return feat, fps


class RealTimePolicyIDPredictor:
    def __init__(self, config_path: str, checkpoint_path: str, input_pkl: Optional[str], device: Optional[str], ema_alpha: float):
        config_loader = ConfigLoader(config_path)
        cfg = config_loader.to_dict()

        if device is None:
            device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.agent = MVQVAEAgent(config=cfg, device=self.device)
        self.agent.config["frame_size"] = MotionDataAdapter.TOTAL_FRAME_SIZE  # 50

        self.agent.model = MotionVQVAE(
            self.agent,
            self.agent.config["nb_code"],
            self.agent.config["code_dim"],
            self.agent.config["output_emb_width"],
            self.agent.config["down_t"],
            self.agent.config["stride_t"],
            self.agent.config["width"],
            self.agent.config["depth"],
            self.agent.config["dilation_growth_rate"],
            self.agent.config["vq_act"],
            self.agent.config["vq_norm"],
        ).to(self.device)

        self._ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.agent.load(checkpoint_path)

        self.W = int(self.agent.config.get("window_size", 32))

        self.agent.policy_classifier = self._load_policy_classifier_from_ckpt()
        if self.agent.policy_classifier is None:
            raise RuntimeError("Policy classifier weights not found or failed to load from ckpt.")

        self.agent.model.eval()
        self.agent.policy_classifier.eval()

        self.mean = None
        self.std = None
        if input_pkl:
            self._load_norm_stats_from_pkl(input_pkl)

        # ring buffer
        self._buf = torch.zeros((self.W, 50), dtype=torch.float32, device=self.device)
        self._buf_len = 0
        self._buf_idx = 0

        self.feat_extractor = CausalFeatureExtractor(self.device, dof_dim=23, ema_alpha=float(ema_alpha), default_fps=30.0)

    def reset(self):
        self._buf_len = 0
        self._buf_idx = 0
        self._buf.zero_()
        self.feat_extractor.reset()

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

    def _load_policy_classifier_from_ckpt(self) -> Optional[PolicyIDClassifier]:
        pol_sd = self._extract_policy_state_dict()
        if pol_sd is None:
            return None
        num_policies = self._infer_num_policies_from_state_dict(pol_sd)
        if num_policies is None:
            return None

        cfg = self.agent.config
        model = PolicyIDClassifier(
            num_codebooks=int(cfg["nb_code"]),
            num_policies=int(num_policies),
            code_dim=int(cfg["code_dim"]),
            hidden_dim=int(cfg.get("policy_classifier_hidden_dim", 256)),
            num_layers=int(cfg.get("policy_classifier_layers", 2)),
            dropout=float(cfg.get("policy_classifier_dropout", 0.1)),
            architecture=str(cfg.get("policy_classifier_architecture", "cnn1d")),
            num_heads=int(cfg.get("policy_classifier_num_heads", 8)),
            kernel_size=int(cfg.get("policy_classifier_kernel_size", 3)),
        ).to(self.device)

        model.load_state_dict(pol_sd, strict=True)
        model.eval()
        return model

    def _load_norm_stats_from_pkl(self, input_pkl: str, max_motions_for_stats: int = 1000):
        motion_data_dict = joblib.load(input_pkl)
        subset_motion_ids = list(range(min(max_motions_for_stats, len(motion_data_dict))))
        temp_adapter = MotionDataAdapter(self.agent.config)
        mocap_data, _, _ = temp_adapter.load_motion_data(input_pkl, subset_motion_ids)
        subset_t = mocap_data.to(self.device) if isinstance(mocap_data, torch.Tensor) else torch.tensor(mocap_data, device=self.device)
        self.mean = subset_t.mean(dim=0)
        self.std = subset_t.std(dim=0).clamp_min(1e-8)

    def _push_feat(self, feat_50: torch.Tensor):
        self._buf[self._buf_idx] = feat_50
        self._buf_idx = (self._buf_idx + 1) % self.W
        self._buf_len = min(self.W, self._buf_len + 1)

    def _get_window_ordered(self) -> torch.Tensor:
        if self._buf_len < self.W:
            return self._buf[:self._buf_len].unsqueeze(0)  # not used for inference
        old = self._buf_idx
        return torch.cat([self._buf[old:], self._buf[:old]], dim=0).unsqueeze(0)  # (1,W,50)

    @torch.no_grad()
    def step(self, fin: FrameInput) -> Optional[Dict[str, object]]:
        feat, fps = self.feat_extractor.step(fin)
        self._push_feat(feat)

        if self._buf_len < self.W:
            return None

        window = self._get_window_ordered()
        if self.mean is None or self.std is None:
            # fallback (not recommended)
            mean = window.mean(dim=1).squeeze(0)
            std = window.std(dim=1).squeeze(0).clamp_min(1e-8)
        else:
            mean = self.mean
            std = self.std

        window_norm = (window - mean.view(1, 1, -1)) / std.view(1, 1, -1)

        codebook = self.agent.model.encode(window_norm).long()
        logits = self.agent.policy_classifier(codebook)   # (1, num_policies)
        probs = torch.softmax(logits, dim=1)              # (1, num_policies)

        pred = int(torch.argmax(probs, dim=1).item())
        p = probs.squeeze(0).detach().cpu().numpy()

        eps = 1e-8
        entropy = float(-np.sum(p * np.log(p + eps)))
        confidence = float(np.max(p))

        return {
            "t": float(fin.t),
            "fps": float(fps),
            "pred_policy_id": pred,
            "probs": p,
            "entropy": entropy,
            "confidence": confidence,
        }


# -----------------------------
# GT policy_id loader for a given npy file
# -----------------------------
def build_gt_policy_ids_for_npy(
    npy_path: Path,
    input_pkl: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      times: (T,)
      gt_policy_ids: (T,)
      motion_ids_raw: (T,)
    """
    data = np.load(npy_path, allow_pickle=True)
    times = data[:, 0].astype(np.float64)
    motion_ids_raw = data[:, -1].astype(np.int64)
    T = data.shape[0]

    policy_ids = None
    motion_ids_from_filename = extract_motion_ids_from_filename(npy_path)

    cached_policy_ids_per_motion = None
    if input_pkl is not None:
        motion_data_dict = joblib.load(input_pkl)
        cached_policy_ids_per_motion = [
            np.asarray(md["policy_id"], dtype=np.int64) if md.get("policy_id") is not None else None
            for md in motion_data_dict.values()
        ]

    # PKL-based GT (preferred)
    if cached_policy_ids_per_motion is not None and motion_ids_from_filename is not None:
        unique_raw = unique_in_order(motion_ids_raw)
        raw_to_seq = {}
        for idx, raw_id in enumerate(unique_raw):
            if idx < len(motion_ids_from_filename):
                pkl_id = motion_ids_from_filename[idx]
                if 0 <= pkl_id < len(cached_policy_ids_per_motion) and cached_policy_ids_per_motion[pkl_id] is not None:
                    raw_to_seq[raw_id] = np.asarray(cached_policy_ids_per_motion[pkl_id], dtype=np.int64)

        if raw_to_seq:
            seg_starts = {raw_id: int(np.where(motion_ids_raw == raw_id)[0][0]) for raw_id in raw_to_seq.keys()}
            out = np.zeros(T, dtype=np.int64)
            for frame_idx, raw_id in enumerate(motion_ids_raw):
                if raw_id in raw_to_seq:
                    seq = raw_to_seq[raw_id]
                    rel_idx = frame_idx - seg_starts.get(raw_id, 0)
                    out[frame_idx] = int(seq[rel_idx % len(seq)]) if len(seq) > 0 else 0
                else:
                    out[frame_idx] = 0
            policy_ids = out

    # filename-based fallback
    if policy_ids is None:
        method_ids = parse_policy_ids_from_filename(npy_path)
        policy_ids = convert_motion_ids_to_policy_ids(motion_ids_raw, method_ids)

    return times, policy_ids[:T], motion_ids_raw[:T]


# -----------------------------
# Evaluation
# -----------------------------
def eval_one_npy_streaming(
    predictor: RealTimePolicyIDPredictor,
    npy_path: Path,
    input_pkl: Optional[str],
    boundary_radius: int,
    max_frames: Optional[int] = None,
    print_every: int = 0,
) -> Dict[str, object]:
    predictor.reset()

    data = np.load(npy_path, allow_pickle=True)
    T = data.shape[0] if max_frames is None else min(data.shape[0], int(max_frames))

    times, gt, motion_ids_raw = build_gt_policy_ids_for_npy(npy_path, input_pkl=input_pkl)
    times = times[:T]
    gt = gt[:T]
    motion_ids_raw = motion_ids_raw[:T]

    pred = np.full(T, -1, dtype=np.int64)
    conf = np.full(T, np.nan, dtype=np.float64)
    ent = np.full(T, np.nan, dtype=np.float64)

    num_emitted = 0
    last_out = None

    for i in range(T):
        fin = FrameInput(
            t=float(data[i, 0]),
            root_pos=data[i, 1:4].astype(np.float32),
            root_rot_wxyz=data[i, 4:8].astype(np.float32),
            dof_pos=data[i, 8:31].astype(np.float32),
        )
        out = predictor.step(fin)
        if out is None:
            continue
        last_out = out
        num_emitted += 1
        pred[i] = int(out["pred_policy_id"])
        conf[i] = float(out["confidence"])
        ent[i] = float(out["entropy"])

        if print_every > 0 and (num_emitted % print_every == 0):
            print(f"t={out['t']:.3f} pred={out['pred_policy_id']} conf={out['confidence']:.3f} ent={out['entropy']:.3f}")

    # For frames before warmup, pred=-1. For accuracy, we evaluate only emitted frames.
    valid = (pred >= 0)
    if not valid.any():
        return {
            "file": npy_path.name,
            "T": int(T),
            "emitted": int(num_emitted),
            "acc": float("nan"),
            "mean_conf": float("nan"),
            "mean_ent": float("nan"),
            "boundary": {},
        }

    acc = float((pred[valid] == gt[valid]).mean())
    mean_conf = float(np.nanmean(conf[valid]))
    mean_ent = float(np.nanmean(ent[valid]))

    bd = compute_boundary_breakdown(gt[valid], pred[valid], motion_ids_raw[valid], radius=int(boundary_radius))

    return {
        "file": npy_path.name,
        "T": int(T),
        "emitted": int(num_emitted),
        "acc": acc,
        "mean_conf": mean_conf,
        "mean_ent": mean_ent,
        "boundary": bd,
        "times": times,
        "gt": gt,
        "pred": pred,
        "confidence": conf,
        "entropy": ent,
    }


def save_csv_per_frame(out_csv: Path, times: np.ndarray, gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, ent: np.ndarray):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["timestep", "gt_policy_id", "pred_policy_id", "confidence", "entropy"])
        for t, g, p, c, e in zip(times, gt, pred, conf, ent):
            w.writerow([f"{float(t):.6f}", int(g), int(p), "" if np.isnan(c) else f"{float(c):.6f}", "" if np.isnan(e) else f"{float(e):.6f}"])


def save_summary_csv(out_csv: Path, rows: List[Dict[str, object]], boundary_radius: int):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["filename", "T", "emitted", "acc", "mean_conf", "mean_ent", "switch_idx", f"near_acc(+/-{boundary_radius})", "out_acc"])
        for r in rows:
            bd = r.get("boundary", {}) or {}
            w.writerow([
                r["file"], r["T"], r["emitted"],
                "" if np.isnan(r["acc"]) else f"{r['acc']:.6f}",
                "" if np.isnan(r["mean_conf"]) else f"{r['mean_conf']:.6f}",
                "" if np.isnan(r["mean_ent"]) else f"{r['mean_ent']:.6f}",
                "" if not bd else int(bd["switch_idx"]),
                "" if not bd else f"{bd['near_acc']:.6f}",
                "" if not bd else f"{bd['out_acc']:.6f}",
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--input_pkl", type=str, default=None)

    # Single-file mode
    parser.add_argument("--demo_npy", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)

    # Directory mode
    parser.add_argument("--npy_data_dir", type=str, default=None, help="Evaluate all .npy in a directory (streaming replay).")
    parser.add_argument("--output_dir", type=str, default=None, help="If set, save per-frame CSV and summary CSV.")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ema_alpha", type=float, default=0.2)
    parser.add_argument("--boundary_radius", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=0, help="Print every N emitted predictions (0=off).")
    args = parser.parse_args()

    predictor = RealTimePolicyIDPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_pkl=args.input_pkl,
        device=args.device,
        ema_alpha=args.ema_alpha,
    )

    out_dir = Path(args.output_dir) if args.output_dir else None
    rows = []

    def _print_one(res: Dict[str, object]):
        bd = res.get("boundary", {}) or {}
        if bd:
            print(f"[RT-Eval] {res['file']}  acc={res['acc']:.3%}  emitted={res['emitted']}/{res['T']}  "
                  f"conf={res['mean_conf']:.3f}  ent={res['mean_ent']:.3f}  "
                  f"switch={int(bd['switch_idx'])} near={bd['near_acc']:.3%} out={bd['out_acc']:.3%}")
        else:
            print(f"[RT-Eval] {res['file']}  acc={res['acc']:.3%}  emitted={res['emitted']}/{res['T']}  "
                  f"conf={res['mean_conf']:.3f}  ent={res['mean_ent']:.3f}")

    if args.demo_npy:
        npy_path = Path(args.demo_npy)
        res = eval_one_npy_streaming(
            predictor=predictor,
            npy_path=npy_path,
            input_pkl=args.input_pkl,
            boundary_radius=args.boundary_radius,
            max_frames=args.max_frames,
            print_every=args.print_every,
        )
        _print_one(res)

        if out_dir:
            per_frame_csv = out_dir / "csv_data" / f"{npy_path.stem}_rt_per_frame.csv"
            save_csv_per_frame(per_frame_csv, res["times"], res["gt"], res["pred"], res["confidence"], res["entropy"])
            summary_csv = out_dir / "csv_data" / "rt_policy_id_summary.csv"
            save_summary_csv(summary_csv, [res], args.boundary_radius)
            print(f"[Saved] {per_frame_csv}")
            print(f"[Saved] {summary_csv}")
        return

    if args.npy_data_dir:
        npy_dir = Path(args.npy_data_dir)
        files = sorted(npy_dir.glob("*.npy"))
        if not files:
            print(f"No .npy in {npy_dir}")
            return

        for f in files:
            res = eval_one_npy_streaming(
                predictor=predictor,
                npy_path=f,
                input_pkl=args.input_pkl,
                boundary_radius=args.boundary_radius,
                max_frames=args.max_frames,
                print_every=0,
            )
            _print_one(res)
            rows.append(res)

            if out_dir:
                per_frame_csv = out_dir / "csv_data" / f"{f.stem}_rt_per_frame.csv"
                save_csv_per_frame(per_frame_csv, res["times"], res["gt"], res["pred"], res["confidence"], res["entropy"])

        if rows:
            accs = np.array([r["acc"] for r in rows if not np.isnan(r["acc"])], dtype=np.float64)
            if accs.size > 0:
                print("=" * 80)
                print(f"[RT-Eval SUMMARY] files={len(rows)}  mean={accs.mean():.3%}  std={accs.std():.3%}  min={accs.min():.3%}  max={accs.max():.3%}")
                print("=" * 80)

            if out_dir:
                summary_csv = out_dir / "csv_data" / "rt_policy_id_summary.csv"
                save_summary_csv(summary_csv, rows, args.boundary_radius)
                print(f"[Saved] {summary_csv}")
        return

    print("Provide either --demo_npy (single) or --npy_data_dir (directory).")


if __name__ == "__main__":
    main()

"""
Example (single file):
python scripts/eval_policy_id_prediction_vqvae_rt.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --demo_npy /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_training/saved_desired_states_108_8_fg.npy \
  --output_dir ./evaluation_policy_id_sequence_rt \
  --boundary_radius 32

Example (directory):
python scripts/eval_policy_id_prediction_vqvae_rt.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --npy_data_dir /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/sequence_data/comparison_inertialization_training \
  --output_dir ./evaluation_policy_id_sequence_rt \
  --boundary_radius 32
"""
