#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent dir so imports work similarly to your training script
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter

# Chunk-wise classifier (uploaded implementation)
from motion_vqvae.models.policy_classifier import PolicyIDClassifier


class MotionAccuracyEvaluator:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        pkl_file_path: str,
        max_motions_for_stats: int = 500,
        eval_stride: Optional[int] = None,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.pkl_file_path = pkl_file_path
        self.max_motions_for_stats = max_motions_for_stats
        self.eval_stride = eval_stride

        # Load config
        config_loader = ConfigLoader(config_path)
        cfg = config_loader.to_dict()

        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Create agent
        self.agent = MVQVAEAgent(config=cfg, device=self.device)

        # Load data first
        self.agent.setup_from_file(pkl_file_path, motion_ids=None)

        # Build mean/std from dataset
        self._setup_stats()

        # Cache policy ids per motion if adapter provides
        self._cached_policy_ids_per_motion = getattr(self.agent.motion_adapter, "all_policy_ids_per_motion", None)

        # Load checkpoint raw
        self._ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load main model weights via agent.load (your original behavior)
        self.agent.load(checkpoint_path)

        # Cache the loaded data to avoid reloading
        self._cached_motion_data = self.agent.motion_adapter.mocap_data
        self._cached_end_indices = self.agent.motion_adapter.end_indices

        # Override eval_stride into agent.config (for overlap reconstruction)
        if self.eval_stride is None:
            self.eval_stride = max(1, int(self.agent.config.get("window_size", 32) // 2))
        self.agent.config["eval_stride"] = int(self.eval_stride)

        print(
            f"[Eval] device={self.device}, window_size={self.agent.config.get('window_size')}, "
            f"eval_stride={self.agent.config['eval_stride']}, "
            f"down_t={self.agent.config.get('down_t')}, stride_t={self.agent.config.get('stride_t')}"
        )

        # Try to load policy classifier robustly (NEW)
        self._try_load_policy_classifier_from_ckpt()

    def _setup_stats(self):
        print("[Stats] Computing statistics from loaded motion data...")
        motions = self.agent.motion_adapter.mocap_data
        end_indices = self.agent.motion_adapter.end_indices

        end_indices_np = (
            end_indices.detach().cpu().numpy().astype(np.int64).tolist()
            if isinstance(end_indices, torch.Tensor)
            else end_indices.tolist()
        )
        use_m = min(self.max_motions_for_stats, len(end_indices_np))
        subset = motions[: end_indices_np[use_m - 1] + 1]
        subset_t = subset.to(self.device) if isinstance(subset, torch.Tensor) else torch.tensor(subset, device=self.device)

        self.agent.mean = subset_t.mean(dim=0)
        self.agent.std = subset_t.std(dim=0).clamp_min(1e-8)
        print(f"[Stats] computed from first {use_m} motions, frames={subset_t.shape[0]}, feat_dim={subset_t.shape[-1]}")

    # --------------------------
    # NEW: robust policy ckpt loading
    # --------------------------
    def _extract_policy_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Supports both:
          - custom checkpoint with 'policy_classifier' key
          - Lightning checkpoint with 'state_dict' containing 'policy_classifier.*'
        """
        ckpt = self._ckpt

        # Case 1: your custom keys
        if isinstance(ckpt, dict) and "policy_classifier" in ckpt:
            sd = ckpt["policy_classifier"]
            if isinstance(sd, dict) and len(sd) > 0:
                return sd

        # Case 2: Lightning: ckpt['state_dict'] has prefixed keys
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            full_sd = ckpt["state_dict"]
            # find any keys starting with 'policy_classifier.'
            pol_keys = [k for k in full_sd.keys() if k.startswith("policy_classifier.")]
            if len(pol_keys) == 0:
                return None
            # strip prefix
            stripped = {k[len("policy_classifier.") :]: full_sd[k] for k in pol_keys}
            return stripped

        return None

    def _infer_num_policies_from_state_dict(self, pol_sd: Dict[str, torch.Tensor]) -> Optional[int]:
        """
        Heuristic: find the LAST linear layer weight in classifier.*.weight
        and use its out_features as num_policies.
        """
        # collect 2D weights under classifier
        candidates = []
        for k, v in pol_sd.items():
            if k.endswith(".weight") and v.ndim == 2 and (k.startswith("classifier.") or ".classifier." in k):
                candidates.append((k, v))

        if not candidates:
            # fallback: any 2D weight could be the head
            candidates = [(k, v) for k, v in pol_sd.items() if k.endswith(".weight") and v.ndim == 2]

        if not candidates:
            return None

        # choose the one that *looks like* output head: usually out_features is small-ish vs hidden_dim,
        # but in your case could be 2. We'll pick the smallest out_features among large-in_features weights.
        # If multiple, pick the one with smallest out_features.
        candidates_sorted = sorted(candidates, key=lambda kv: kv[1].shape[0])
        num_policies = int(candidates_sorted[0][1].shape[0])
        return num_policies

    def _filter_state_dict_by_shape(self, model: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Keep only keys that exist in model.state_dict() AND shapes match.
        This avoids hard crashes on size mismatch, at the cost of partial loading.
        """
        model_sd = model.state_dict()
        filtered = {}
        mismatched = []
        missing = []

        for k, v in sd.items():
            if k not in model_sd:
                missing.append(k)
                continue
            if tuple(model_sd[k].shape) != tuple(v.shape):
                mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            filtered[k] = v

        if mismatched:
            print("[Eval][Policy] shape-mismatched keys (show up to 10):")
            for k, s1, s2 in mismatched[:10]:
                print(f"  - {k}: ckpt={s1} vs model={s2}")
        if missing:
            print(f"[Eval][Policy] keys not in current model (count={len(missing)})")

        return filtered

    def _try_load_policy_classifier_from_ckpt(self):
        pol_sd = self._extract_policy_state_dict()
        if pol_sd is None:
            print("[Eval][Policy] No policy classifier weights found in checkpoint. policy prediction will be skipped.")
            self.agent.policy_classifier = None
            return

        # infer num_policies from ckpt weights (robust)
        inferred_num_policies = self._infer_num_policies_from_state_dict(pol_sd)
        if inferred_num_policies is None:
            print("[Eval][Policy] Could not infer num_policies from checkpoint. policy prediction will be skipped.")
            self.agent.policy_classifier = None
            return

        # Build chunk-wise classifier as a robust fallback
        cfg = self.agent.config
        arch = cfg.get("policy_classifier_architecture", "cnn1d")
        hidden = int(cfg.get("policy_classifier_hidden_dim", 256))
        layers = int(cfg.get("policy_classifier_layers", 2))
        dropout = float(cfg.get("policy_classifier_dropout", 0.1))
        num_heads = int(cfg.get("policy_classifier_num_heads", 8))
        kernel = int(cfg.get("policy_classifier_kernel_size", 3))

        # IMPORTANT:
        # Even if your training was "sequence-wise", this fallback at least enables evaluation/debugging.
        model = PolicyIDClassifier(
            num_codebooks=int(cfg["nb_code"]),
            num_policies=int(inferred_num_policies),
            code_dim=int(cfg["code_dim"]),
            hidden_dim=hidden,
            num_layers=layers,
            dropout=dropout,
            architecture=str(arch),
            num_heads=num_heads,
            kernel_size=kernel,
        ).to(self.device)

        # Try strict load first
        try:
            model.load_state_dict(pol_sd, strict=True)
            model.eval()
            self.agent.policy_classifier = model
            print(f"[Eval][Policy] Loaded policy classifier STRICT. num_policies={inferred_num_policies}, arch={arch}")
            return
        except Exception as e:
            print(f"[Eval][Policy] Strict load failed: {e}")

        # Fallback: filter by shape and load partial
        filtered = self._filter_state_dict_by_shape(model, pol_sd)
        if len(filtered) == 0:
            print("[Eval][Policy] No compatible keys after filtering. policy prediction will be skipped.")
            self.agent.policy_classifier = None
            return

        try:
            model.load_state_dict(filtered, strict=False)
            model.eval()
            self.agent.policy_classifier = model
            print(
                f"[Eval][Policy] Loaded policy classifier PARTIAL (filtered). "
                f"num_policies={inferred_num_policies}, arch={arch}, loaded_keys={len(filtered)}/{len(pol_sd)}"
            )
        except Exception as e:
            print(f"[Eval][Policy] Partial load still failed: {e}")
            self.agent.policy_classifier = None

    # --------------------------
    # original logic
    # --------------------------
    def _extract_motion_segment(self, motion_id: int) -> Tuple[torch.Tensor, Optional[np.ndarray], bool]:
        end_indices_np = (
            self._cached_end_indices.detach().cpu().numpy().astype(np.int64).tolist()
            if isinstance(self._cached_end_indices, torch.Tensor)
            else self._cached_end_indices.tolist()
        )

        if motion_id >= len(end_indices_np):
            raise IndexError(f"Motion ID {motion_id} is out of bounds (max: {len(end_indices_np) - 1})")

        start_idx = 0 if motion_id == 0 else end_indices_np[motion_id - 1] + 1
        end_idx = end_indices_np[motion_id] + 1
        gt_features = self._cached_motion_data[start_idx:end_idx]
        T = gt_features.shape[0]

        motion_policy_ids = None
        motion_has_policy_ids = False

        if motion_id < 1000 and self._cached_policy_ids_per_motion is not None:
            if motion_id < len(self._cached_policy_ids_per_motion):
                policy_id_data = self._cached_policy_ids_per_motion[motion_id]
                if policy_id_data is not None:
                    motion_policy_ids = np.asarray(policy_id_data)
                    motion_has_policy_ids = True
                    if len(motion_policy_ids) != T:
                        if len(motion_policy_ids) > T:
                            motion_policy_ids = motion_policy_ids[:T]
                        else:
                            pad_value = motion_policy_ids[-1] if len(motion_policy_ids) > 0 else -1
                            motion_policy_ids = np.pad(
                                motion_policy_ids, (0, T - len(motion_policy_ids)), constant_values=pad_value
                            )

        return gt_features.to(self.device), motion_policy_ids, motion_has_policy_ids

    def _make_overlapped_windows(self, seq: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        W = int(self.agent.config.get("window_size", 32))
        S = int(self.agent.config.get("eval_stride", max(1, W // 2)))

        seq_norm = (seq.to(self.device) - self.agent.mean.to(self.device)) / self.agent.std.to(self.device)
        T, num_windows = seq_norm.shape[0], (seq_norm.shape[0] + S - 1) // S
        windows = torch.zeros((num_windows, W, seq_norm.shape[1]), device=seq_norm.device, dtype=seq_norm.dtype)
        indices = []

        for w_idx in range(num_windows):
            start, end = w_idx * S, min(w_idx * S + W, T)
            orig_len = end - start
            windows[w_idx, :orig_len] = seq_norm[start:end]
            if orig_len < W:
                continue
                # windows[w_idx, orig_len:] = seq_norm[end - 1 : end].expand(W - orig_len, -1)
            indices.append((start, orig_len))

        return windows, indices

    def _stitch_overlapped(self, recon_windows_denorm: torch.Tensor, indices: List[Tuple[int, int]], T: int) -> torch.Tensor:
        C = recon_windows_denorm.shape[-1]
        recon = torch.zeros((T, C), device=recon_windows_denorm.device, dtype=recon_windows_denorm.dtype)
        weight = torch.zeros((T,), device=recon_windows_denorm.device, dtype=recon_windows_denorm.dtype)

        for w_idx, (start, orig_len) in enumerate(indices):
            end = min(T, start + orig_len)
            seg_len = end - start
            recon[start:end] += recon_windows_denorm[w_idx, :seg_len]
            weight[start:end] += 1.0

        recon = recon / weight.unsqueeze(-1).clamp_min(1.0)
        return recon

    def _calculate_reconstruction_metrics(self, gt: torch.Tensor, rec: torch.Tensor) -> Tuple[Dict[str, float], np.ndarray]:
        T = min(gt.shape[0], rec.shape[0])
        gt = gt[:T]
        rec = rec[:T]

        rd_s, rd_e = MotionDataAdapter.ROOT_DELTAS_START, MotionDataAdapter.ROOT_DELTAS_END
        dp_s, dp_e = MotionDataAdapter.DOF_POSITIONS_START, MotionDataAdapter.DOF_POSITIONS_END
        dv_s, dv_e = MotionDataAdapter.DOF_VELOCITIES_START, MotionDataAdapter.DOF_VELOCITIES_END

        root_diff = gt[:, rd_s:rd_e] - rec[:, rd_s:rd_e]
        root_abs_err = torch.abs(root_diff)
        root_rmse_vec = torch.sqrt(torch.mean(root_diff ** 2, dim=0))
        root_mae_vec = torch.mean(root_abs_err, dim=0)

        root_deltas_rmse = float(torch.sqrt(torch.mean(root_diff ** 2)))
        dof_pos_rmse = float(torch.sqrt(torch.mean((gt[:, dp_s:dp_e] - rec[:, dp_s:dp_e]) ** 2)))
        dof_vel_rmse = float(torch.sqrt(torch.mean((gt[:, dv_s:dv_e] - rec[:, dv_s:dv_e]) ** 2)))
        overall_rmse = float(torch.sqrt(torch.mean((gt - rec) ** 2)))

        root_abs_err_np = root_abs_err.detach().cpu().numpy()
        root_rmse_vec_np = root_rmse_vec.detach().cpu().numpy()
        root_mae_vec_np = root_mae_vec.detach().cpu().numpy()

        metrics = {
            "root_dx_rmse": float(root_rmse_vec_np[0]),
            "root_dy_rmse": float(root_rmse_vec_np[1]),
            "root_dz_rmse": float(root_rmse_vec_np[2]),
            "root_dyaw_rmse": float(root_rmse_vec_np[3]),
            "root_rmse_vec": root_rmse_vec_np.astype(np.float32).tolist(),
            "root_mae_vec": root_mae_vec_np.astype(np.float32).tolist(),
            "root_deltas_rmse": root_deltas_rmse,
            "dof_pos_rmse": dof_pos_rmse,
            "dof_vel_rmse": dof_vel_rmse,
            "overall_rmse": overall_rmse,
        }
        return metrics, root_abs_err_np

    def _evaluate_dynamic_prediction_from_windows(self, codebook_windows: torch.Tensor) -> Dict[str, float]:
        if self.agent.latent_predictor is None:
            return {}

        pred_len = int(self.agent.config.get("latent_predictor_pred_len", 1))
        if codebook_windows.shape[1] <= pred_len:
            return {"codebook_acc": 0.0, "codebook_top3_acc": 0.0, "avg_error": 0.0}

        with torch.no_grad():
            inp = codebook_windows[:, :-pred_len]
            tgt = codebook_windows[:, -pred_len:]
            pred_logits = self.agent.latent_predictor(inp)
            pred_indices = torch.argmax(pred_logits, dim=-1)

            all_preds = pred_indices.flatten().cpu()
            all_targets = tgt.flatten().cpu()
            all_logits = pred_logits.reshape(-1, pred_logits.shape[-1]).cpu()

        acc = (all_preds == all_targets).float().mean().item()
        top3 = torch.topk(all_logits, k=3, dim=-1).indices
        top3_acc = (top3 == all_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
        avg_err = torch.abs(all_preds.float() - all_targets.float()).mean().item()

        return {"codebook_acc": float(acc), "codebook_top3_acc": float(top3_acc), "avg_error": float(avg_err)}

    def _plot_reconstruction(
        self,
        gt_features: torch.Tensor,
        rec_features: torch.Tensor,
        motion_id: int,
        output_dir: Path,
        recon_metrics: Dict[str, float],
    ):
        """Plot reconstruction comparison: ground truth vs reconstructed features."""
        gt_np = gt_features.detach().cpu().numpy() if isinstance(gt_features, torch.Tensor) else gt_features
        rec_np = rec_features.detach().cpu().numpy() if isinstance(rec_features, torch.Tensor) else rec_features
        
        num_frames = min(gt_np.shape[0], rec_np.shape[0])
        fps = 30.0  # Assuming 30 FPS
        time_axis = np.arange(num_frames) / fps
        
        rd_s, rd_e = MotionDataAdapter.ROOT_DELTAS_START, MotionDataAdapter.ROOT_DELTAS_END
        dp_s, dp_e = MotionDataAdapter.DOF_POSITIONS_START, MotionDataAdapter.DOF_POSITIONS_END
        dv_s, dv_e = MotionDataAdapter.DOF_VELOCITIES_START, MotionDataAdapter.DOF_VELOCITIES_END
        
        # Plot 1: Root Deltas (dx, dy, dz, dyaw)
        fig1, axes1 = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig1.suptitle(f"Root Deltas Reconstruction - Motion {motion_id}", fontsize=16, fontweight='bold')
        
        labels = ["dx", "dy", "dz", "dyaw"]
        for i in range(4):
            axes1[i].plot(time_axis, gt_np[:num_frames, rd_s + i], label=f"GT {labels[i]}", linewidth=1.5, alpha=0.8)
            axes1[i].plot(time_axis, rec_np[:num_frames, rd_s + i], "--", label=f"REC {labels[i]}", linewidth=1.5, alpha=0.8)
            axes1[i].set_title(f"Root Delta: {labels[i]} (RMSE: {recon_metrics.get(f'root_{labels[i]}_rmse', 0.0):.4f})")
            axes1[i].set_ylabel("value")
            axes1[i].legend(loc='upper right')
            axes1[i].grid(True, linestyle="--", alpha=0.3)
        
        axes1[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path1 = output_dir / f"motion_{motion_id}_reconstruction_root_deltas.png"
        plt.savefig(save_path1, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f" Saved root deltas reconstruction plot to: {save_path1}")
        
        # Plot 2: DOF Positions and Velocities (first 5 DOFs)
        fig2, axes2 = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig2.suptitle(f"DOF Reconstruction - Motion {motion_id}", fontsize=16, fontweight='bold')
        
        num_dofs_to_plot = min(5, dp_e - dp_s)
        
        # DOF Positions
        for i in range(num_dofs_to_plot):
            axes2[0].plot(time_axis, gt_np[:num_frames, dp_s + i], label=f"GT q[{i}]", linewidth=1.5, alpha=0.7)
            axes2[0].plot(time_axis, rec_np[:num_frames, dp_s + i], "--", label=f"REC q[{i}]", linewidth=1.5, alpha=0.7)
        axes2[0].set_title(f"DOF Positions (First {num_dofs_to_plot}, RMSE: {recon_metrics.get('dof_pos_rmse', 0.0):.4f})")
        axes2[0].set_ylabel("rad")
        axes2[0].legend(ncol=2, loc='upper right', fontsize=9)
        axes2[0].grid(True, linestyle="--", alpha=0.3)
        
        # DOF Velocities
        for i in range(num_dofs_to_plot):
            axes2[1].plot(time_axis, gt_np[:num_frames, dv_s + i], label=f"GT dq[{i}]", linewidth=1.5, alpha=0.7)
            axes2[1].plot(time_axis, rec_np[:num_frames, dv_s + i], "--", label=f"REC dq[{i}]", linewidth=1.5, alpha=0.7)
        axes2[1].set_title(f"DOF Velocities (First {num_dofs_to_plot}, RMSE: {recon_metrics.get('dof_vel_rmse', 0.0):.4f})")
        axes2[1].set_ylabel("rad/s")
        axes2[1].set_xlabel("Time (s)")
        axes2[1].legend(ncol=2, loc='upper right', fontsize=9)
        axes2[1].grid(True, linestyle="--", alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path2 = output_dir / f"motion_{motion_id}_reconstruction_dof_overlays.png"
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f" Saved DOF reconstruction plot to: {save_path2}")
        
        return save_path1, save_path2

    def _plot_policy_id_tracking(
        self,
        gt_policy_ids: np.ndarray,
        pred_policy_ids: np.ndarray,
        motion_id: int,
        output_dir: Path,
        accuracy: float,
    ):
        """Plot policy ID tracking over time (ground truth vs predicted)."""
        T = len(gt_policy_ids)
        frames = np.arange(T)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Get unique policy IDs for coloring
        all_pids = np.unique(np.concatenate([gt_policy_ids, pred_policy_ids]))
        num_policies = len(all_pids)
        colors = plt.cm.tab10(np.linspace(0, 1, max(num_policies, 10)))
        pid_to_color = {pid: colors[i % len(colors)] for i, pid in enumerate(all_pids)}
        
        # Plot 1: Ground Truth Policy IDs
        ax1 = axes[0]
        for pid in all_pids:
            mask = gt_policy_ids == pid
            if mask.any():
                ax1.scatter(frames[mask], gt_policy_ids[mask], 
                           c=[pid_to_color[pid]], label=f'Policy {pid}', 
                           alpha=0.7, s=20, edgecolors='black', linewidth=0.3)
        ax1.set_ylabel('Policy ID', fontsize=12)
        ax1.set_title(f'Ground Truth Policy IDs (Motion {motion_id})', fontsize=14, fontweight='bold')
        ax1.set_ylim([min(all_pids) - 0.5, max(all_pids) + 0.5])
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', ncol=min(num_policies, 5), fontsize=9)
        ax1.set_yticks(sorted(all_pids))
        
        # Plot 2: Predicted Policy IDs with error highlighting
        ax2 = axes[1]
        correct_mask = gt_policy_ids == pred_policy_ids
        incorrect_mask = ~correct_mask
        
        # Plot correct predictions
        if correct_mask.any():
            ax2.scatter(frames[correct_mask], pred_policy_ids[correct_mask],
                       c='green', alpha=0.6, s=20, label='Correct', 
                       edgecolors='darkgreen', linewidth=0.3, marker='o')
        
        # Plot incorrect predictions
        if incorrect_mask.any():
            ax2.scatter(frames[incorrect_mask], pred_policy_ids[incorrect_mask],
                       c='darkred', alpha=0.8, s=30, label='Incorrect',
                       marker='x', linewidth=1.5)
            # Also show what the ground truth was at those points
            for frame_idx in frames[incorrect_mask]:
                gt_val = gt_policy_ids[frame_idx]
                pred_val = pred_policy_ids[frame_idx]
                # Draw a line from prediction to ground truth
                ax2.plot([frame_idx, frame_idx], [pred_val, gt_val], 
                        'r--', alpha=0.4, linewidth=1)
        
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Policy ID', fontsize=12)
        ax2.set_title(f'Predicted Policy IDs (Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')
        ax2.set_ylim([min(all_pids) - 0.5, max(all_pids) + 0.5])
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_yticks(sorted(all_pids))
        
        # Add statistics text box
        stats_text = f'Total Frames: {T}\n'
        stats_text += f'Accuracy: {accuracy:.2%}\n'
        stats_text += f'Correct: {correct_mask.sum()}\n'
        stats_text += f'Incorrect: {incorrect_mask.sum()}'
        
        # Add text box to the plot
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        plt.tight_layout()
        
        # Save the plot
        save_path = output_dir / f"motion_{motion_id}_policy_id_tracking.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path

    def _evaluate_policy_prediction_windowed(
        self,
        codebook_windows: torch.Tensor,
        window_indices: List[Tuple[int, int]],
        policy_ids: np.ndarray,
        T: int,
    ) -> Dict[str, object]:
        if self.agent.policy_classifier is None:
            return {}

        with torch.no_grad():
            logits = self.agent.policy_classifier(codebook_windows)
            pred_class = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)

        # direct: class == policy_id (or map if you have it)
        pred_policy_per_window = pred_class

        # overlap vote to frames
        max_pid, min_pid = int(pred_policy_per_window.max()), int(pred_policy_per_window.min())
        offset = -min_pid if min_pid < 0 else 0
        num_policies = max_pid + 1 + offset

        vote_counts = torch.zeros((T, num_policies), device=self.device, dtype=torch.long)
        for w_idx, (start, orig_len) in enumerate(window_indices):
            pid = int(pred_policy_per_window[w_idx]) + offset
            vote_counts[start : min(T, start + orig_len), pid] += 1

        pred_per_frame = (torch.argmax(vote_counts, dim=1).cpu().numpy().astype(np.int64) - offset)
        no_vote_mask = (vote_counts.sum(dim=1) == 0).cpu().numpy()
        if no_vote_mask.any():
            pred_per_frame[no_vote_mask] = int(pred_policy_per_window[-1])

        gt = policy_ids[:T].astype(np.int64)
        correct = (pred_per_frame == gt)
        acc = float(correct.mean())

        conf = {}
        for g, p in zip(gt, pred_per_frame):
            conf[(int(g), int(p))] = conf.get((int(g), int(p)), 0) + 1
        confusion = [(k[0], k[1], v) for k, v in sorted(conf.items(), key=lambda x: -x[1])]

        gt_mode = int(np.bincount(gt).argmax()) if len(gt) > 0 else -1
        pred_mode = int(np.bincount(pred_per_frame).argmax()) if len(pred_per_frame) > 0 else -1

        return {
            "policy_acc": acc,
            "gt_policy_id": gt_mode,
            "pred_policy_id": pred_mode,
            "pred_policy_ids_per_frame": pred_per_frame,
            "gt_policy_ids_per_frame": gt,
            "confusion": confusion,
        }

    def evaluate_motion(self, motion_id: int, output_dir: Optional[str] = None) -> Optional[Dict[str, object]]:
        gt_features, motion_policy_ids, _ = self._extract_motion_segment(motion_id)
        T = gt_features.shape[0]

        if motion_policy_ids is not None:
            unique_pids = np.unique(motion_policy_ids)
            print(f"[Eval] Motion {motion_id}: {len(motion_policy_ids)} frames, policy IDs: {unique_pids.tolist()}")

        self.agent.mocap_data = gt_features
        self.agent.end_indices = torch.as_tensor([T - 1], device=self.device)

        print("\n[1] Reconstruction Evaluation")
        print("-" * 80)

        windows, win_indices = self._make_overlapped_windows(gt_features)

        self.agent.model.eval()
        with torch.no_grad():
            recon_windows_norm, _, _ = self.agent.model(windows)
            mean = self.agent.mean.to(recon_windows_norm.device)
            std = self.agent.std.to(recon_windows_norm.device)
            recon_windows = recon_windows_norm * std + mean
            codebook_windows = self.agent.model.encode(windows).long()

        reconstructed = self._stitch_overlapped(recon_windows, win_indices, T)

        recon_metrics, per_frame_root_abs_err = self._calculate_reconstruction_metrics(gt_features, reconstructed)
        print(" Reconstruction Accuracy (MVQ Feature Space):")
        print(f"  - Root Deltas RMSE (dx, dy, dz, dyaw): {recon_metrics['root_rmse_vec']}")
        print(f"  - Root Deltas MAE  (dx, dy, dz, dyaw): {recon_metrics['root_mae_vec']}")
        print(f"  - DOF Position RMSE: {recon_metrics['dof_pos_rmse']:.4f}")
        print(f"  - DOF Velocity RMSE: {recon_metrics['dof_vel_rmse']:.4f}")
        print(f"  - Overall Feature RMSE: {recon_metrics['overall_rmse']:.4f}")

        dyn_metrics = {}
        if self.agent.latent_predictor is not None:
            print("\n[2] Dynamic Prediction Evaluation (Next Codebook) [WINDOWED]")
            print("-" * 80)
            dyn_metrics = self._evaluate_dynamic_prediction_from_windows(codebook_windows)
            print(f"  - Codebook Prediction Accuracy: {dyn_metrics['codebook_acc']:.2%}")
            print(f"  - Codebook Prediction Top-3 Accuracy: {dyn_metrics['codebook_top3_acc']:.2%}")
            print(f"  - Average Prediction Error: {dyn_metrics['avg_error']:.4f}")
        else:
            print("\n[2] Dynamic Prediction Evaluation: SKIPPED (latent predictor not available)")

        policy_metrics = {}
        if self.agent.policy_classifier is not None and motion_policy_ids is not None:
            print("\n[3] Policy ID Prediction Evaluation [WINDOWED]")
            print("-" * 80)
            policy_metrics = self._evaluate_policy_prediction_windowed(
                codebook_windows=codebook_windows,
                window_indices=win_indices,
                policy_ids=motion_policy_ids,
                T=T,
            )
            print(f"  - Policy ID Accuracy: {policy_metrics['policy_acc']:.2%}")
            print("  - Policy ID Confusion Matrix (top counts):")
            for true_pid, pred_pid, count in policy_metrics["confusion"][:20]:
                print(f"    True={true_pid}, Pred={pred_pid}: {count} frames")
        else:
            reason = "policy classifier not available" if self.agent.policy_classifier is None else f"motion {motion_id} has no policy IDs"
            print(f"\n[3] Policy ID Prediction Evaluation: SKIPPED ({reason})")

        all_metrics = {**recon_metrics, **dyn_metrics, **{k: v for k, v in policy_metrics.items() if k not in ["pred_policy_ids_per_frame", "gt_policy_ids_per_frame"]}}

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            err_path = out / f"motion_{motion_id}_errors_root_deltas.npz"
            np.savez_compressed(err_path, abs_err=per_frame_root_abs_err)
            print(f"\n Saved per-frame root delta errors to: {err_path}")
            
            # Save reconstruction plots
            self._plot_reconstruction(
                gt_features=gt_features,
                rec_features=reconstructed,
                motion_id=motion_id,
                output_dir=out,
                recon_metrics=recon_metrics,
            )
            
            # Save policy ID tracking plot if available
            if policy_metrics and "pred_policy_ids_per_frame" in policy_metrics and "gt_policy_ids_per_frame" in policy_metrics:
                tracking_path = self._plot_policy_id_tracking(
                    gt_policy_ids=policy_metrics["gt_policy_ids_per_frame"],
                    pred_policy_ids=policy_metrics["pred_policy_ids_per_frame"],
                    motion_id=motion_id,
                    output_dir=out,
                    accuracy=policy_metrics["policy_acc"],
                )
                print(f" Saved policy ID tracking plot to: {tracking_path}")
            
            # Save metrics to JSON file
            metrics_path = out / f"motion_{motion_id}_metrics.json"
            # Convert numpy arrays and other non-serializable types to Python types
            metrics_serializable = {}
            for k, v in all_metrics.items():
                if isinstance(v, (np.ndarray, np.generic)):
                    metrics_serializable[k] = v.tolist() if hasattr(v, 'tolist') else float(v)
                elif isinstance(v, (np.integer, np.floating)):
                    metrics_serializable[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (list, tuple)):
                    metrics_serializable[k] = [float(x) if isinstance(x, (np.generic, np.ndarray)) else x for x in v]
                else:
                    metrics_serializable[k] = v
            with open(metrics_path, 'w') as f:
                json.dump(metrics_serializable, f, indent=2)
            print(f" Saved metrics to: {metrics_path}")

        return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQVAE reconstruction + latent prediction + policy ID prediction (robust policy ckpt loading)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to agent config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to input PKL motion data file")
    parser.add_argument("--motion_id", type=int, required=True, help="Motion ID (index) to evaluate")
    parser.add_argument("--max_motions_for_stats", type=int, default=1000, help="Max motions to use for normalization stats")
    parser.add_argument("--output_dir", type=str, default="./evaluation_plots", help="Directory to save plots/errors")
    parser.add_argument("--eval_stride", type=int, default=None, help="Stride (frames) for overlapped windows (default: window_size//2)")
    args = parser.parse_args()

    evaluator = MotionAccuracyEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        pkl_file_path=args.input_pkl,
        max_motions_for_stats=args.max_motions_for_stats,
        eval_stride=args.eval_stride,
    )
    metrics = evaluator.evaluate_motion(args.motion_id, args.output_dir)

    if metrics:
        print("\nEvaluation complete.")
    else:
        print("\nEvaluation failed.")


if __name__ == "__main__":
    main()


'''

# Evaluate reconstruction, dynamic prediction, and policy ID prediction

python scripts/eval_vqvae_rec_pred_dyn.py \
  --config configs/agent_codebook_switching_base.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id_base/best_model.ckpt \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --motion_id 0 \
  --output_dir ./evaluation_plots_rec_pred_dyn_base




python scripts/eval_vqvae_rec_pred_dyn.py \
  --config configs/agent_codebook_switching.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id/best_model.ckpt \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --motion_id 211 \
  --output_dir ./evaluation_plots_rec_pred_dyn




python scripts/eval_vqvae_rec_pred_dyn.py \
  --config configs/agent_codebook_switching_seq.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id_seq/best_model.ckpt \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --motion_id 8 \
  --output_dir ./evaluation_plots_rec_pred_dyn_sequence_wise



python scripts/eval_vqvae_rec_pred_dyn.py \
  --config configs/agent_codebook_switching_cls.yaml \
  --checkpoint /home/baekdh/dh_workspace/vqvae_motion_g1/checkpoints/run_0_1000_switching_policy_id_seq_cls/best_model.ckpt \
  --input_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --motion_id 8 \
  --output_dir ./evaluation_plots_rec_pred_dyn_cls

'''