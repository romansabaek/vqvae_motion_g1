#!/usr/bin/env python3
"""
Simplified VQVAE reconstruction accuracy evaluator with plotting.

This script computes reconstruction accuracy for a single motion and
generates plots comparing the ground-truth and reconstructed feature
trajectories over time.

Added:
- Per-component metrics for root deltas (dx, dy, dz, dyaw)
- Optional saving of per-frame 4xT absolute error trajectories
- Device consistency to avoid CPU/CUDA mismatches
"""

import numpy as np
import torch
import joblib
import argparse
from pathlib import Path
import sys
from typing import Dict
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Add the parent directory to sys.path to allow imports from motion_vqvae
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from scripts.vqvae_gen_init import (
    load_config_and_agent,
    ensure_stats,
    initialize_model,
)


class MotionAccuracyEvaluator:
    """
    Evaluates motion accuracy and plots feature-level trajectories.
    """
    def __init__(self, config_path: str, checkpoint_path: str, pkl_file: str, max_motions_for_stats: int = 300, eval_stride: int = None):
        print("Initializing Motion Accuracy Evaluator...")

        self.pkl_file_path = pkl_file
        self.config, self.agent, self.motion_adapter = load_config_and_agent(config_path, checkpoint_path)

        # Propagate optional eval_stride into config/agent so agent.evaluate_policy_rec uses overlap
        if eval_stride is not None:
            self.config["eval_stride"] = int(eval_stride)
            self.agent.config["eval_stride"] = int(eval_stride)

        # Load motion keys just to know valid range
        print(f"Loading motion data from: {pkl_file}")
        all_motions = joblib.load(pkl_file)
        self.motion_keys = list(all_motions.keys())

        # agent and adapter already initialized above

        # Load a subset of motions to compute normalization statistics
        num_motions_to_load = min(max_motions_for_stats, len(self.motion_keys))
        subset_motion_ids = list(range(num_motions_to_load))
        print(f"Loading first {num_motions_to_load} motions for normalization stats...")

        mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
            self.pkl_file_path, subset_motion_ids
        )

        # Ensure tensors live on the agent device
        if isinstance(mocap_data, torch.Tensor):
            mocap_data = mocap_data.to(self.agent.device)
        if isinstance(end_indices, torch.Tensor):
            end_indices = end_indices.to(self.agent.device)

        print(f"Loaded data for stats: shape={mocap_data.shape}, frame_size={frame_size}")

        # Calculate and attach normalization stats to the agent (or keep ckpt stats)
        self.agent.frame_size = int(frame_size)
        ensure_stats(self.agent, mocap_data)

        # Initialize and load the trained model
        initialize_model(self.agent, self.config, self.agent.frame_size, checkpoint_path)
        print("✅ Evaluator initialized successfully!")

    def _initialize_model(self):
        """Kept for backward compatibility; now handled by function utilities."""
        pass

    def evaluate_motion(self, motion_id: int, output_dir: str = None) -> Dict[str, float]:
        """
        Evaluates reconstruction accuracy for a single motion ID and optionally plots/saves the results.
        Returns a dict of metrics (including per-component root delta errors).
        """
        if not (0 <= motion_id < len(self.motion_keys)):
            print(f"❌ Error: Motion ID {motion_id} is out of range (max: {len(self.motion_keys) - 1})")
            return None

        print(f"\n▶️ Evaluating Motion ID: {motion_id} (Key: {self.motion_keys[motion_id]})")
        print("=" * 60)

        # Load the ground-truth motion data for the specified ID
        gt_features, _, _ = self.motion_adapter.load_motion_data(self.pkl_file_path, [motion_id])
        if isinstance(gt_features, torch.Tensor):
            gt_features = gt_features.to(self.agent.device)

        # Temporarily set the agent's context to this single motion
        self.agent.mocap_data = gt_features
        # end_indices is either tensor or list; ensure index on device
        last_idx = gt_features.shape[0] - 1
        self.agent.end_indices = torch.as_tensor([last_idx], device=self.agent.device)

        # Reconstruct the motion using the trained model
        with torch.no_grad():
            reconstructed_features, _, _ = self.agent.evaluate_policy_rec(
                torch.tensor(0, device=self.agent.device)
            )

        # Compute accuracy metrics in the feature space
        metrics, per_frame_root_abs_err = self._calculate_metrics(gt_features, reconstructed_features)

        print("📊 Accuracy Results (MVQ Feature Space):")
        print(f"  - Root Deltas RMSE (dx, dy, dz, dyaw): {metrics['root_rmse_vec']}")
        print(f"  - Root Deltas MAE  (dx, dy, dz, dyaw): {metrics['root_mae_vec']}")
        print(f"  - DOF Position RMSE: {metrics['dof_pos_rmse']:.4f}")
        print(f"  - DOF Velocity RMSE: {metrics['dof_vel_rmse']:.4f}")
        print(f"  - Overall Feature RMSE: {metrics['overall_rmse']:.4f}")

        # Plot and save artifacts if requested
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save per-frame 4xT absolute errors for root deltas
            err_path = output_path / f"motion_{motion_id}_errors_root_deltas.npz"
            # per_frame_root_abs_err shape: [T, 4] -> save as is
            np.savez_compressed(err_path, abs_err=per_frame_root_abs_err)
            print(f"🧾 Saved per-frame root delta errors to: {err_path}")

            # Make plots
            self._plot_feature_trajectories(
                gt_features.detach().cpu().numpy(),
                reconstructed_features.detach().cpu().numpy(),
                motion_id,
                output_dir,
            )

        return metrics

    def _calculate_metrics(
        self, ground_truth: torch.Tensor, reconstructed: torch.Tensor
    ) -> (Dict[str, float], np.ndarray):
        """Calculates RMSE/MAE metrics; also returns per-frame absolute error for root deltas (T,4)."""
        gt = ground_truth.detach().cpu().numpy()
        rec = reconstructed.detach().cpu().numpy()

        # Align lengths
        T = min(gt.shape[0], rec.shape[0])
        gt, rec = gt[:T], rec[:T]

        # Index ranges from adapter (assumed constants)
        rd_s, rd_e = MotionDataAdapter.ROOT_DELTAS_START, MotionDataAdapter.ROOT_DELTAS_END  # 4 dims
        dp_s, dp_e = MotionDataAdapter.DOF_POSITIONS_START, MotionDataAdapter.DOF_POSITIONS_END
        dv_s, dv_e = MotionDataAdapter.DOF_VELOCITIES_START, MotionDataAdapter.DOF_VELOCITIES_END

        # Root deltas per-dim absolute error over time (T,4)
        root_abs_err = np.abs(gt[:, rd_s:rd_e] - rec[:, rd_s:rd_e])  # [T,4]

        # Per-dim RMSE & MAE (4,)
        root_rmse_vec = np.sqrt(np.mean((gt[:, rd_s:rd_e] - rec[:, rd_s:rd_e]) ** 2, axis=0))
        root_mae_vec = np.mean(root_abs_err, axis=0)

        # Aggregate RMSEs
        root_deltas_rmse = float(np.sqrt(np.mean((gt[:, rd_s:rd_e] - rec[:, rd_s:rd_e]) ** 2)))
        dof_pos_rmse = float(np.sqrt(np.mean((gt[:, dp_s:dp_e] - rec[:, dp_s:dp_e]) ** 2)))
        dof_vel_rmse = float(np.sqrt(np.mean((gt[:, dv_s:dv_e] - rec[:, dv_s:dv_e]) ** 2)))
        overall_rmse = float(np.sqrt(np.mean((gt - rec) ** 2)))

        metrics = {
            # Per-component
            "root_dx_rmse": float(root_rmse_vec[0]),
            "root_dy_rmse": float(root_rmse_vec[1]),
            "root_dz_rmse": float(root_rmse_vec[2]),
            "root_dyaw_rmse": float(root_rmse_vec[3]),
            "root_rmse_vec": root_rmse_vec.astype(np.float32).tolist(),
            "root_mae_vec": root_mae_vec.astype(np.float32).tolist(),

            # Aggregates
            "root_deltas_rmse": root_deltas_rmse,
            "dof_pos_rmse": dof_pos_rmse,
            "dof_vel_rmse": dof_vel_rmse,
            "overall_rmse": overall_rmse,
        }
        return metrics, root_abs_err  # (dict, [T,4])

    def _plot_feature_trajectories(
        self, gt_features: np.ndarray, rec_features: np.ndarray, motion_id: int, output_dir: str
    ):
        """Generates and saves plots comparing feature trajectories."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_frames = min(gt_features.shape[0], rec_features.shape[0])
        time_axis = np.arange(num_frames) / 30.0  # Assuming 30 FPS

        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        fig.suptitle(f"Feature Trajectory Comparison for Motion ID: {motion_id}", fontsize=16)

        # 1. Root deltas (dx, dy, dz, dyaw) — plotted separately for clarity
        rd_s, rd_e = MotionDataAdapter.ROOT_DELTAS_START, MotionDataAdapter.ROOT_DELTAS_END
        labels = ["dx", "dy", "dz", "dyaw"]
        for i in range(4):
            axes[i].plot(time_axis, gt_features[:num_frames, rd_s + i], label=f"GT {labels[i]}")
            axes[i].plot(time_axis, rec_features[:num_frames, rd_s + i], "--", label=f"REC {labels[i]}")
            axes[i].set_title(f"Root Delta: {labels[i]}")
            axes[i].set_ylabel("value")
            axes[i].legend()
            axes[i].grid(True, linestyle="--", alpha=0.6)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = output_path / f"motion_{motion_id}_root_deltas.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"📈 Root delta plots saved to: {save_path}")

        # Optional: small overlay plot of first 5 DOF pos/vel (kept concise)
        fig2, axes2 = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        dp_s = MotionDataAdapter.DOF_POSITIONS_START
        dv_s = MotionDataAdapter.DOF_VELOCITIES_START
        num_dofs_to_plot = min(5, rec_features.shape[1] - dp_s)

        for i in range(num_dofs_to_plot):
            axes2[0].plot(time_axis, gt_features[:num_frames, dp_s + i], label=f"GT q[{i}]")
            axes2[0].plot(time_axis, rec_features[:num_frames, dp_s + i], "--", label=f"REC q[{i}]")
        axes2[0].set_title(f"DOF Positions (First {num_dofs_to_plot})")
        axes2[0].set_ylabel("rad")
        axes2[0].legend(ncol=2)
        axes2[0].grid(True, linestyle="--", alpha=0.6)

        for i in range(num_dofs_to_plot):
            axes2[1].plot(time_axis, gt_features[:num_frames, dv_s + i], label=f"GT dq[{i}]")
            axes2[1].plot(time_axis, rec_features[:num_frames, dv_s + i], "--", label=f"REC dq[{i}]")
        axes2[1].set_title(f"DOF Velocities (First {num_dofs_to_plot})")
        axes2[1].set_ylabel("rad/s")
        axes2[1].legend(ncol=2)
        axes2[1].grid(True, linestyle="--", alpha=0.6)

        plt.xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path2 = output_path / f"motion_{motion_id}_dof_overlays.png"
        plt.savefig(save_path2, dpi=150)
        plt.close(fig2)
        print(f"📈 DOF plots saved to: {save_path2}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VQVAE reconstruction accuracy and plot feature trajectories."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to agent config file (e.g., configs/agent.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to input PKL motion data file")
    parser.add_argument("--motion_id", type=int, required=True, help="The ID (index) of the motion to evaluate")
    parser.add_argument("--max_motions_for_stats", type=int, default=300, help="Max motions to use for normalization stats")
    parser.add_argument("--output_dir", type=str, default="./evaluation_plots", help="Directory to save plots/errors")
    parser.add_argument("--eval_stride", type=int, default=None, help="Stride for overlapped reconstruction (default: window_size//2)")

    args = parser.parse_args()

    evaluator = MotionAccuracyEvaluator(
        args.config, args.checkpoint, args.input_pkl, args.max_motions_for_stats, args.eval_stride
    )
    metrics = evaluator.evaluate_motion(args.motion_id, args.output_dir)

    if metrics:
        print("\n✅ Evaluation complete.")
    else:
        print("\n❌ Evaluation failed.")


if __name__ == "__main__":
    main()


'''

python scripts/eval_vqvae.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_id 0 \
  --output_dir ./evaluation_plots

python scripts/eval_vqvae.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300_v2/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_id 0 \
  --output_dir ./evaluation_plots_v2

'''