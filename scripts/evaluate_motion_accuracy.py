#!/usr/bin/env python3
"""
Comprehensive motion accuracy evaluation script.
Compares AMASS ground truth vs VQVAE-generated motions at feature level.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Add motion_vqvae to path
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.agent import MVQVAEAgent
from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.data.motion_data_adapter import MotionDataAdapter

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MotionAccuracyEvaluator:
    """Evaluate motion accuracy between AMASS ground truth and VQVAE-generated motions."""
    
    def __init__(self, config_path: str, checkpoint_path: str, input_pkl_file: str, max_motions: int = 300):
        """Initialize evaluator with config, checkpoint, and motion data."""
        print("🔧 Initializing Motion Accuracy Evaluator...")
        
        # Load config
        config_loader = ConfigLoader()
        self.config = config_loader.load_config(config_path)
        
        # Load original AMASS data (limit to first 300 motions)
        print(f"📂 Loading AMASS data from: {input_pkl_file}")
        self.original_motions = joblib.load(input_pkl_file)
        self.original_keys = list(self.original_motions.keys())
        
        # Limit motions for evaluation
        actual_max_motions = min(max_motions, len(self.original_keys))
        self.original_keys = self.original_keys[:actual_max_motions]
        print(f"📊 Loaded {len(self.original_keys)} original motions (limited to first {max_motions})")
        
        # Initialize agent and motion adapter
        self.agent = MVQVAEAgent(config=self.config)
        self.checkpoint_path = checkpoint_path
        self.motion_adapter = MotionDataAdapter(self.config)
        
        # Load motion data for normalization (use a small subset for efficiency)
        print(f"🔄 Loading motion data for normalization...")
        # Use first 50 motions for normalization statistics (much faster)
        subset_size = min(300, len(self.original_keys))
        subset_motion_ids = list(range(subset_size))
        print(f"📊 Using first {subset_size} motions for normalization statistics")
        
        print(f"📊 Loading {subset_size} motions for normalization statistics...")
        self.mocap_data, self.end_indices, self.frame_size = self.motion_adapter.load_motion_data(
            input_pkl_file, subset_motion_ids
        )
        
        print(f"📊 Loaded motion data: {self.mocap_data.shape}, frame_size: {self.frame_size}")
        
        # Setup agent with motion data and normalization statistics
        self.agent.mocap_data = self.mocap_data
        self.agent.end_indices = self.end_indices
        self.agent.frame_size = self.frame_size
        
        # Calculate normalization statistics from subset
        print("📊 Calculating normalization statistics...")
        mean = self.mocap_data.mean(dim=0)
        std = self.mocap_data.std(dim=0)
        std[std == 0] = 1.0
        
        self.agent.mean = mean
        self.agent.std = std
        
        # Store normalization stats and motion file path for on-demand loading
        self.normalization_mean = mean
        self.normalization_std = std
        self.motion_file_path = input_pkl_file
        
        # Initialize model
        self._initialize_model()
        
        print("✅ Motion Accuracy Evaluator initialized successfully!")
    
    def _initialize_model(self):
        """Initialize and load the trained VQVAE model."""
        from motion_vqvae.models.models import MotionVQVAE
        
        # Ensure frame_size is in config
        self.agent.config['frame_size'] = self.frame_size
        
        # Initialize VQVAE model
        self.agent.model = MotionVQVAE(
            self.agent,
            self.config['nb_code'],
            self.config['code_dim'],
            self.config['output_emb_width'],
            self.config['down_t'],
            self.config['stride_t'],
            self.config['width'],
            self.config['depth'],
            self.config['dilation_growth_rate'],
            self.config['vq_act'],
            self.config['vq_norm']
        ).to(self.agent.device)
        
        # Load trained model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.agent.device)
        self.agent.model.load_state_dict(checkpoint['model'])
        self.agent.model.eval()
        
        print(f"🤖 Loaded trained model from: {self.checkpoint_path}")
    
    def evaluate_single_motion(self, motion_id: int, save_plots: bool = True, output_dir: str = "./outputs/evaluation"):
        """Evaluate accuracy for a single motion and generate detailed plots."""
        if motion_id >= len(self.original_keys):
            print(f"❌ Motion ID {motion_id} out of range (max: {len(self.original_keys)-1})")
            return None
        
        print(f"\n🎯 Evaluating Motion ID: {motion_id}")
        print("=" * 50)
        
        # Get original motion
        original_key = self.original_keys[motion_id]
        original_motion = self.original_motions[original_key]
        
        print(f"📝 Motion Key: {original_key}")
        print(f"⏱️  Duration: {original_motion['dof'].shape[0] / 30.0:.2f}s ({original_motion['dof'].shape[0]} frames)")
        
        # Generate VQVAE motion
        # Load this specific motion for evaluation
        print(f"  🔄 Loading motion {motion_id} for evaluation...")
        motion_data, _, _ = self.motion_adapter.load_motion_data(
            self.motion_file_path, [motion_id]
        )
        # Update agent data temporarily
        self.agent.mocap_data = motion_data
        self.agent.end_indices = [motion_data.shape[0] - 1]
        
        with torch.no_grad():
            reconstructed_motion, original_seq, codebook_sequence = self.agent.evaluate_policy_rec(torch.tensor(0))
        
        # Convert to AMASS format for comparison
        vqvae_motion = self._convert_vqvae_to_amass_format(reconstructed_motion, original_motion)
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(original_motion, vqvae_motion)
        
        # Display results
        print(f"\n📊 Accuracy Results for Motion {motion_id}:")
        print(f"  Root Position RMSE: {accuracy_metrics['root_pos_rmse']:.4f} m")
        print(f"  Root Rotation RMSE: {accuracy_metrics['root_rot_rmse']:.4f} rad")
        print(f"  DOF Position RMSE: {accuracy_metrics['dof_pos_rmse']:.4f} rad")
        print(f"  DOF Velocity RMSE: {accuracy_metrics['dof_vel_rmse']:.4f} rad/s")
        print(f"  Local Position RMSE: {accuracy_metrics['local_pos_rmse']:.4f} m")
        print(f"  Local Velocity RMSE: {accuracy_metrics['local_vel_rmse']:.4f} m/s")
        
        # Generate plots
        if save_plots:
            print("📊 Generating detailed plots...")
            self._plot_single_motion_analysis(original_motion, vqvae_motion, motion_id, accuracy_metrics, output_dir)
        
        return accuracy_metrics
    
    def evaluate_all_motions(self, motion_ids: List[int] = None, output_dir: str = "./outputs/evaluation"):
        """Evaluate accuracy across all specified motions and generate summary plots."""
        if motion_ids is None:
            motion_ids = list(range(len(self.original_keys)))
        
        print(f"\n📊 Evaluating {len(motion_ids)} motions...")
        print("=" * 50)
        
        all_metrics = []
        
        # Filter valid motion IDs
        valid_motion_ids = [mid for mid in motion_ids if mid < len(self.original_keys)]
        
        # Create progress bar
        pbar = tqdm(valid_motion_ids, desc="Processing motions", unit="motion")
        
        for motion_id in pbar:
            pbar.set_description(f"Processing motion ID {motion_id}")
            
            try:
                # Get original motion
                original_key = self.original_keys[motion_id]
                original_motion = self.original_motions[original_key]
                
                # Generate VQVAE motion
                with torch.no_grad():
                    reconstructed_motion, original_seq, codebook_sequence = self.agent.evaluate_policy_rec(torch.tensor(motion_id))
                
                # Convert to AMASS format
                vqvae_motion = self._convert_vqvae_to_amass_format(reconstructed_motion, original_motion)
                
                # Calculate accuracy metrics
                metrics = self._calculate_accuracy_metrics(original_motion, vqvae_motion)
                metrics['motion_id'] = motion_id
                metrics['motion_key'] = original_key
                metrics['duration'] = original_motion['dof'].shape[0] / 30.0
                
                all_metrics.append(metrics)
                
                # Update progress bar with current metrics
                current_rmse = metrics['root_pos_rmse']
                pbar.set_postfix({
                    'Root RMSE': f'{current_rmse:.4f}',
                    'Completed': f'{len(all_metrics)}/{len(valid_motion_ids)}'
                })
                
            except Exception as e:
                tqdm.write(f"❌ Error processing motion {motion_id}: {e}")
                continue
        
        pbar.close()
        
        # Create summary DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "motion_accuracy_results.csv"
        df.to_csv(results_file, index=False)
        print(f"📄 Results saved to: {results_file}")
        
        # Generate summary plots
        print("📊 Generating summary plots...")
        self._plot_summary_analysis(df, output_path)
        
        # Print summary statistics
        self._print_summary_statistics(df)
        
        return df
    
    def _convert_vqvae_to_amass_format(self, reconstructed_motion: torch.Tensor, reference_motion: Dict):
        """Convert VQVAE output to AMASS format for comparison."""
        mvq_np = reconstructed_motion.cpu().numpy()
        seq_len = mvq_np.shape[0]
        fps = reference_motion.get('fps', 30)
        
        # Extract MVQ components
        root_deltas = mvq_np[:, 0:4]
        dof_positions = mvq_np[:, 4:27]
        dof_velocities = mvq_np[:, 27:50]
        
        # Convert root deltas to global positions
        root_trans_offset = np.zeros((seq_len, 3))
        root_trans_offset[0] = reference_motion['root_trans_offset'][0]
        
        root_rot = np.zeros((seq_len, 4))
        root_rot[0] = reference_motion['root_rot'][0]
        
        for i in range(1, seq_len):
            local_delta = root_deltas[i, :3]
            yaw_delta = root_deltas[i, 3]
            
            prev_yaw = np.arctan2(root_rot[i-1, 3], root_rot[i-1, 0]) * 2
            
            # Rotate local delta to global frame
            cos_yaw = np.cos(prev_yaw)
            sin_yaw = np.sin(prev_yaw)
            global_delta = np.array([
                cos_yaw * local_delta[0] - sin_yaw * local_delta[1],
                sin_yaw * local_delta[0] + cos_yaw * local_delta[1],
                local_delta[2]
            ])
            
            root_trans_offset[i] = root_trans_offset[i-1] + global_delta
            
            # Update rotation
            new_yaw = prev_yaw + yaw_delta
            root_rot[i] = np.array([np.cos(new_yaw/2), 0, 0, np.sin(new_yaw/2)])
        
        # Calculate local frame features for comparison
        local_pos, local_vel = self._calculate_local_features(root_trans_offset, root_rot, fps)
        
        return {
            'root_trans_offset': root_trans_offset,
            'root_rot': root_rot,
            'dof': dof_positions,
            'dof_vel': dof_velocities,
            'local_pos': local_pos,
            'local_vel': local_vel,
            'fps': fps
        }
    
    def _calculate_local_features(self, root_pos: np.ndarray, root_rot: np.ndarray, fps: float):
        """Calculate local frame position and velocity."""
        # Position in local frame (relative to initial position)
        local_pos = root_pos - root_pos[0]
        
        # Velocity in local frame
        local_vel = np.zeros_like(root_pos)
        local_vel[1:] = fps * (root_pos[1:] - root_pos[:-1])
        
        return local_pos, local_vel
    
    def _calculate_accuracy_metrics(self, original: Dict, vqvae: Dict) -> Dict:
        """Calculate comprehensive accuracy metrics including per-axis and per-joint analysis."""
        metrics = {}
        
        # Root position accuracy (global)
        root_pos_diff = original['root_trans_offset'] - vqvae['root_trans_offset']
        metrics['root_pos_rmse'] = np.sqrt(np.mean(root_pos_diff**2))
        metrics['root_pos_mae'] = np.mean(np.abs(root_pos_diff))
        
        # Root position per-axis accuracy (global)
        for i, axis in enumerate(['x', 'y', 'z']):
            metrics[f'root_pos_{axis}_rmse'] = np.sqrt(np.mean(root_pos_diff[:, i]**2))
            metrics[f'root_pos_{axis}_mae'] = np.mean(np.abs(root_pos_diff[:, i]))
        
        # Root rotation accuracy (quaternion distance)
        root_rot_diff = self._quaternion_distance(original['root_rot'], vqvae['root_rot'])
        metrics['root_rot_rmse'] = np.sqrt(np.mean(root_rot_diff**2))
        metrics['root_rot_mae'] = np.mean(np.abs(root_rot_diff))
        
        # Root rotation per-axis accuracy (extract roll, pitch, yaw)
        roll_diff, pitch_diff, yaw_diff = self._quaternion_to_rpy_diff(original['root_rot'], vqvae['root_rot'])
        metrics['root_rot_roll_rmse'] = np.sqrt(np.mean(roll_diff**2))
        metrics['root_rot_pitch_rmse'] = np.sqrt(np.mean(pitch_diff**2))
        metrics['root_rot_yaw_rmse'] = np.sqrt(np.mean(yaw_diff**2))
        metrics['root_rot_roll_mae'] = np.mean(np.abs(roll_diff))
        metrics['root_rot_pitch_mae'] = np.mean(np.abs(pitch_diff))
        metrics['root_rot_yaw_mae'] = np.mean(np.abs(yaw_diff))
        
        # DOF position accuracy (overall)
        dof_pos_diff = original['dof'] - vqvae['dof']
        metrics['dof_pos_rmse'] = np.sqrt(np.mean(dof_pos_diff**2))
        metrics['dof_pos_mae'] = np.mean(np.abs(dof_pos_diff))
        
        # DOF position per-joint accuracy
        num_dof = dof_pos_diff.shape[1]
        for i in range(num_dof):
            metrics[f'dof_{i}_rmse'] = np.sqrt(np.mean(dof_pos_diff[:, i]**2))
            metrics[f'dof_{i}_mae'] = np.mean(np.abs(dof_pos_diff[:, i]))
        
        # DOF velocity accuracy (if available)
        if 'dof_vel' in vqvae:
            dof_vel_diff = original.get('dof_vel', np.zeros_like(original['dof'])) - vqvae['dof_vel']
            metrics['dof_vel_rmse'] = np.sqrt(np.mean(dof_vel_diff**2))
            metrics['dof_vel_mae'] = np.mean(np.abs(dof_vel_diff))
            
            # DOF velocity per-joint accuracy
            for i in range(num_dof):
                metrics[f'dof_vel_{i}_rmse'] = np.sqrt(np.mean(dof_vel_diff[:, i]**2))
                metrics[f'dof_vel_{i}_mae'] = np.mean(np.abs(dof_vel_diff[:, i]))
        else:
            metrics['dof_vel_rmse'] = 0.0
            metrics['dof_vel_mae'] = 0.0
            for i in range(num_dof):
                metrics[f'dof_vel_{i}_rmse'] = 0.0
                metrics[f'dof_vel_{i}_mae'] = 0.0
        
        # Local frame accuracy
        local_pos_diff = original.get('local_pos', original['root_trans_offset'] - original['root_trans_offset'][0]) - vqvae['local_pos']
        metrics['local_pos_rmse'] = np.sqrt(np.mean(local_pos_diff**2))
        metrics['local_pos_mae'] = np.mean(np.abs(local_pos_diff))
        
        # Local position per-axis accuracy
        for i, axis in enumerate(['x', 'y', 'z']):
            metrics[f'local_pos_{axis}_rmse'] = np.sqrt(np.mean(local_pos_diff[:, i]**2))
            metrics[f'local_pos_{axis}_mae'] = np.mean(np.abs(local_pos_diff[:, i]))
        
        local_vel_diff = original.get('local_vel', np.zeros_like(original['root_trans_offset'])) - vqvae['local_vel']
        metrics['local_vel_rmse'] = np.sqrt(np.mean(local_vel_diff**2))
        metrics['local_vel_mae'] = np.mean(np.abs(local_vel_diff))
        
        # Local velocity per-axis accuracy
        for i, axis in enumerate(['x', 'y', 'z']):
            metrics[f'local_vel_{axis}_rmse'] = np.sqrt(np.mean(local_vel_diff[:, i]**2))
            metrics[f'local_vel_{axis}_mae'] = np.mean(np.abs(local_vel_diff[:, i]))
        
        return metrics
    
    def _quaternion_to_rpy_diff(self, q1: np.ndarray, q2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert quaternions to roll-pitch-yaw and calculate differences."""
        def quat_to_rpy(q):
            # Convert quaternion to roll-pitch-yaw (ZYX convention)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (w * y - z * x)
            pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw
        
        r1, p1, y1 = quat_to_rpy(q1)
        r2, p2, y2 = quat_to_rpy(q2)
        
        # Calculate differences (handle angle wrapping)
        roll_diff = np.arctan2(np.sin(r1 - r2), np.cos(r1 - r2))
        pitch_diff = np.arctan2(np.sin(p1 - p2), np.cos(p1 - p2))
        yaw_diff = np.arctan2(np.sin(y1 - y2), np.cos(y1 - y2))
        
        return roll_diff, pitch_diff, yaw_diff
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Calculate distance between quaternions."""
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
        
        # Calculate dot product
        dot = np.sum(q1 * q2, axis=1)
        
        # Clamp to avoid numerical issues
        dot = np.clip(dot, -1.0, 1.0)
        
        # Calculate angular distance
        return 2 * np.arccos(np.abs(dot))
    
    def _plot_single_motion_analysis(self, original: Dict, vqvae: Dict, motion_id: int, metrics: Dict, output_dir: str):
        """Generate detailed plots for a single motion."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        time = np.arange(original['dof'].shape[0]) / 30.0
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Motion {motion_id} - AMASS vs VQVAE Comparison', fontsize=16)
        
        # 1. Root Position
        axes[0, 0].plot(time, original['root_trans_offset'][:, 0], 'b-', label='AMASS X', linewidth=2)
        axes[0, 0].plot(time, original['root_trans_offset'][:, 1], 'g-', label='AMASS Y', linewidth=2)
        axes[0, 0].plot(time, original['root_trans_offset'][:, 2], 'r-', label='AMASS Z', linewidth=2)
        axes[0, 0].plot(time, vqvae['root_trans_offset'][:, 0], 'b--', label='VQVAE X', alpha=0.7)
        axes[0, 0].plot(time, vqvae['root_trans_offset'][:, 1], 'g--', label='VQVAE Y', alpha=0.7)
        axes[0, 0].plot(time, vqvae['root_trans_offset'][:, 2], 'r--', label='VQVAE Z', alpha=0.7)
        axes[0, 0].set_title(f'Root Position (RMSE: {metrics["root_pos_rmse"]:.4f})')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Root Rotation
        rot_dist = self._quaternion_distance(original['root_rot'], vqvae['root_rot'])
        axes[0, 1].plot(time, rot_dist, 'purple', linewidth=2)
        axes[0, 1].set_title(f'Root Rotation Distance (RMSE: {metrics["root_rot_rmse"]:.4f})')
        axes[0, 1].set_ylabel('Angular Distance (rad)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. DOF Positions (first 5 DOF)
        for i in range(min(5, original['dof'].shape[1])):
            axes[1, 0].plot(time, original['dof'][:, i], 'b-', alpha=0.7, label=f'AMASS DOF{i}' if i == 0 else "")
            axes[1, 0].plot(time, vqvae['dof'][:, i], 'r--', alpha=0.7, label=f'VQVAE DOF{i}' if i == 0 else "")
        axes[1, 0].set_title(f'DOF Positions - First 5 DOF (RMSE: {metrics["dof_pos_rmse"]:.4f})')
        axes[1, 0].set_ylabel('Angle (rad)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. DOF Velocities (if available)
        if 'dof_vel' in vqvae:
            for i in range(min(5, vqvae['dof_vel'].shape[1])):
                axes[1, 1].plot(time, vqvae['dof_vel'][:, i], 'r--', alpha=0.7, label=f'VQVAE DOF{i}' if i == 0 else "")
            axes[1, 1].set_title(f'DOF Velocities - First 5 DOF (RMSE: {metrics["dof_vel_rmse"]:.4f})')
            axes[1, 1].set_ylabel('Velocity (rad/s)')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'DOF Velocities\nNot Available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('DOF Velocities')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Local Position
        axes[2, 0].plot(time, original.get('local_pos', original['root_trans_offset'] - original['root_trans_offset'][0])[:, 0], 'b-', label='AMASS Local X', linewidth=2)
        axes[2, 0].plot(time, original.get('local_pos', original['root_trans_offset'] - original['root_trans_offset'][0])[:, 1], 'g-', label='AMASS Local Y', linewidth=2)
        axes[2, 0].plot(time, original.get('local_pos', original['root_trans_offset'] - original['root_trans_offset'][0])[:, 2], 'r-', label='AMASS Local Z', linewidth=2)
        axes[2, 0].plot(time, vqvae['local_pos'][:, 0], 'b--', label='VQVAE Local X', alpha=0.7)
        axes[2, 0].plot(time, vqvae['local_pos'][:, 1], 'g--', label='VQVAE Local Y', alpha=0.7)
        axes[2, 0].plot(time, vqvae['local_pos'][:, 2], 'r--', label='VQVAE Local Z', alpha=0.7)
        axes[2, 0].set_title(f'Local Position (RMSE: {metrics["local_pos_rmse"]:.4f})')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Local Position (m)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Local Velocity
        axes[2, 1].plot(time, original.get('local_vel', np.zeros_like(original['root_trans_offset']))[:, 0], 'b-', label='AMASS Local VX', linewidth=2)
        axes[2, 1].plot(time, original.get('local_vel', np.zeros_like(original['root_trans_offset']))[:, 1], 'g-', label='AMASS Local VY', linewidth=2)
        axes[2, 1].plot(time, original.get('local_vel', np.zeros_like(original['root_trans_offset']))[:, 2], 'r-', label='AMASS Local VZ', linewidth=2)
        axes[2, 1].plot(time, vqvae['local_vel'][:, 0], 'b--', label='VQVAE Local VX', alpha=0.7)
        axes[2, 1].plot(time, vqvae['local_vel'][:, 1], 'g--', label='VQVAE Local VY', alpha=0.7)
        axes[2, 1].plot(time, vqvae['local_vel'][:, 2], 'r--', label='VQVAE Local VZ', alpha=0.7)
        axes[2, 1].set_title(f'Local Velocity (RMSE: {metrics["local_vel_rmse"]:.4f})')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Local Velocity (m/s)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_path / f"motion_{motion_id:03d}_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Single motion plot saved to: {plot_file}")
    
    def _plot_summary_analysis(self, df: pd.DataFrame, output_path: Path):
        """Generate summary plots across all motions with detailed error bars."""
        
        print("  📈 Creating feature-wise RMSE bar chart with error bars...")
        
        # 1. Feature-wise RMSE bar chart with error bars
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Motion Accuracy Summary - Feature-wise RMSE with Error Bars', fontsize=16)
        
        # 1. Root Position per-axis (X, Y, Z)
        root_pos_axes = ['root_pos_x_rmse', 'root_pos_y_rmse', 'root_pos_z_rmse']
        root_pos_means = [df[metric].mean() for metric in root_pos_axes]
        root_pos_stds = [df[metric].std() for metric in root_pos_axes]
        root_pos_labels = ['X', 'Y', 'Z']
        
        bars = axes[0, 0].bar(root_pos_labels, root_pos_means, yerr=root_pos_stds, 
                             capsize=5, color=['red', 'green', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Root Position RMSE (Global Frame)')
        axes[0, 0].set_ylabel('RMSE (m)')
        for i, (mean, std) in enumerate(zip(root_pos_means, root_pos_stds)):
            axes[0, 0].text(i, mean + std + max(root_pos_means) * 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 2. Root Rotation per-axis (Roll, Pitch, Yaw)
        root_rot_axes = ['root_rot_roll_rmse', 'root_rot_pitch_rmse', 'root_rot_yaw_rmse']
        root_rot_means = [df[metric].mean() for metric in root_rot_axes]
        root_rot_stds = [df[metric].std() for metric in root_rot_axes]
        root_rot_labels = ['Roll', 'Pitch', 'Yaw']
        
        bars = axes[0, 1].bar(root_rot_labels, root_rot_means, yerr=root_rot_stds, 
                             capsize=5, color=['orange', 'purple', 'brown'], alpha=0.7)
        axes[0, 1].set_title('Root Rotation RMSE')
        axes[0, 1].set_ylabel('RMSE (rad)')
        for i, (mean, std) in enumerate(zip(root_rot_means, root_rot_stds)):
            axes[0, 1].text(i, mean + std + max(root_rot_means) * 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 3. Local Position per-axis (X, Y, Z)
        local_pos_axes = ['local_pos_x_rmse', 'local_pos_y_rmse', 'local_pos_z_rmse']
        local_pos_means = [df[metric].mean() for metric in local_pos_axes]
        local_pos_stds = [df[metric].std() for metric in local_pos_axes]
        local_pos_labels = ['X', 'Y', 'Z']
        
        bars = axes[1, 0].bar(local_pos_labels, local_pos_means, yerr=local_pos_stds, 
                             capsize=5, color=['darkred', 'darkgreen', 'darkblue'], alpha=0.7)
        axes[1, 0].set_title('Local Position RMSE (Robot Frame)')
        axes[1, 0].set_ylabel('RMSE (m)')
        for i, (mean, std) in enumerate(zip(local_pos_means, local_pos_stds)):
            axes[1, 0].text(i, mean + std + max(local_pos_means) * 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 4. Joint tracking accuracy (first 10 DOF)
        dof_metrics = [f'dof_{i}_rmse' for i in range(min(10, 23))]  # First 10 DOF
        dof_means = [df[metric].mean() for metric in dof_metrics]
        dof_stds = [df[metric].std() for metric in dof_metrics]
        dof_labels = [f'DOF {i}' for i in range(min(10, 23))]
        
        bars = axes[1, 1].bar(dof_labels, dof_means, yerr=dof_stds, 
                             capsize=3, color=plt.cm.tab10(np.linspace(0, 1, len(dof_labels))), alpha=0.7)
        axes[1, 1].set_title('Joint Tracking RMSE (First 10 DOF)')
        axes[1, 1].set_ylabel('RMSE (rad)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, (mean, std) in enumerate(zip(dof_means, dof_stds)):
            axes[1, 1].text(i, mean + std + max(dof_means) * 0.01, f'{mean:.3f}', 
                           ha='center', va='bottom', fontsize=7, rotation=90)
        
        # 5. Local Velocity per-axis (X, Y, Z)
        local_vel_axes = ['local_vel_x_rmse', 'local_vel_y_rmse', 'local_vel_z_rmse']
        local_vel_means = [df[metric].mean() for metric in local_vel_axes]
        local_vel_stds = [df[metric].std() for metric in local_vel_axes]
        local_vel_labels = ['X', 'Y', 'Z']
        
        bars = axes[2, 0].bar(local_vel_labels, local_vel_means, yerr=local_vel_stds, 
                             capsize=5, color=['coral', 'lightgreen', 'lightblue'], alpha=0.7)
        axes[2, 0].set_title('Local Velocity RMSE (Robot Frame)')
        axes[2, 0].set_ylabel('RMSE (m/s)')
        for i, (mean, std) in enumerate(zip(local_vel_means, local_vel_stds)):
            axes[2, 0].text(i, mean + std + max(local_vel_means) * 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=8)
        
        # 6. Overall summary with error bars
        all_metrics = ['root_pos_rmse', 'root_rot_rmse', 'dof_pos_rmse', 'local_pos_rmse']
        all_means = [df[metric].mean() for metric in all_metrics]
        all_stds = [df[metric].std() for metric in all_metrics]
        all_labels = ['Root Pos', 'Root Rot', 'DOF Pos', 'Local Pos']
        
        bars = axes[2, 1].bar(all_labels, all_means, yerr=all_stds, 
                             capsize=5, color=plt.cm.Set3(np.linspace(0, 1, len(all_labels))), alpha=0.7)
        axes[2, 1].set_title('Overall Feature RMSE Summary')
        axes[2, 1].set_ylabel('RMSE')
        axes[2, 1].tick_params(axis='x', rotation=45)
        for i, (mean, std) in enumerate(zip(all_means, all_stds)):
            axes[2, 1].text(i, mean + std + max(all_means) * 0.01, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_file = output_path / "motion_accuracy_summary.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  📊 Creating distribution plots...")
        
        # 2. Distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Motion Accuracy Distribution Analysis', fontsize=16)
        
        metrics_to_plot = ['root_pos_rmse', 'root_rot_rmse', 'dof_pos_rmse', 'dof_vel_rmse', 'local_pos_rmse', 'local_vel_rmse']
        metric_labels = ['Root Position', 'Root Rotation', 'DOF Position', 'DOF Velocity', 'Local Position', 'Local Velocity']
        
        # Progress bar for plotting
        plot_pbar = tqdm(zip(metrics_to_plot, metric_labels), total=len(metrics_to_plot), desc="Creating plots", unit="plot")
        
        for i, (metric, label) in enumerate(plot_pbar):
            plot_pbar.set_description(f"Plotting {label}")
            row, col = i // 3, i % 3
            
            axes[row, col].hist(df[metric], bins=20, alpha=0.7, color=plt.cm.Set3(i))
            axes[row, col].axvline(df[metric].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[metric].mean():.4f}')
            axes[row, col].axvline(df[metric].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df[metric].median():.4f}')
            axes[row, col].set_title(f'{label} RMSE Distribution')
            axes[row, col].set_xlabel('RMSE')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save distribution plot
        dist_file = output_path / "motion_accuracy_distribution.png"
        plt.savefig(dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_pbar.close()
        
        print(f"📊 Summary plots saved to: {summary_file} and {dist_file}")
    
    def _print_summary_statistics(self, df: pd.DataFrame):
        """Print comprehensive summary statistics."""
        print(f"\n📈 MOTION ACCURACY SUMMARY STATISTICS")
        print("=" * 80)
        
        # Overall metrics
        metrics = ['root_pos_rmse', 'root_rot_rmse', 'dof_pos_rmse', 'dof_vel_rmse', 'local_pos_rmse', 'local_vel_rmse']
        metric_names = ['Root Position', 'Root Rotation', 'DOF Position', 'DOF Velocity', 'Local Position', 'Local Velocity']
        
        print("🎯 OVERALL FEATURE ACCURACY:")
        for metric, name in zip(metrics, metric_names):
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()
            
            print(f"{name:15} RMSE: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
        
        print(f"\n🎯 ROOT POSITION PER-AXIS (Global Frame):")
        for axis in ['x', 'y', 'z']:
            metric = f'root_pos_{axis}_rmse'
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"  {axis.upper()}-axis:     {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n🎯 ROOT ROTATION PER-AXIS:")
        for axis in ['roll', 'pitch', 'yaw']:
            metric = f'root_rot_{axis}_rmse'
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"  {axis.capitalize()}:      {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n🎯 LOCAL POSITION PER-AXIS (Robot Frame):")
        for axis in ['x', 'y', 'z']:
            metric = f'local_pos_{axis}_rmse'
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"  {axis.upper()}-axis:     {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n🎯 JOINT TRACKING ACCURACY (First 5 DOF):")
        for i in range(min(5, 23)):
            metric = f'dof_{i}_rmse'
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            print(f"  DOF {i:2d}:         {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total motions evaluated: {len(df)}")
        print(f"  Average motion duration: {df['duration'].mean():.2f}s")
        print(f"  Duration range: {df['duration'].min():.2f}s - {df['duration'].max():.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Evaluate motion accuracy between AMASS and VQVAE')
    parser.add_argument('--config', type=str, default='configs/agent.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt', help='Path to model checkpoint')
    parser.add_argument('--input_pkl', type=str, required=True, help='Path to input PKL motion data file')
    parser.add_argument('--motion_id', type=int, default=None, help='Single motion ID to analyze in detail')
    parser.add_argument('--motion_ids', type=str, default=None, help='Comma-separated motion IDs for batch evaluation (e.g., "0-10")')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation', help='Output directory for results')
    parser.add_argument('--eval_all', action='store_true', help='Evaluate all motions (warning: slow)')
    parser.add_argument('--max_motions', type=int, default=300, help='Maximum number of motions to load (default: 300)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MotionAccuracyEvaluator(args.config, args.checkpoint, args.input_pkl, args.max_motions)
    
    # Parse motion IDs
    if args.motion_id is not None:
        # Single motion detailed analysis
        evaluator.evaluate_single_motion(args.motion_id, save_plots=True, output_dir=args.output_dir)
        
    elif args.motion_ids is not None:
        # Batch evaluation
        motion_ids = []
        for x in args.motion_ids.split(','):
            x = x.strip()
            if '-' in x:
                start, end = map(int, x.split('-'))
                motion_ids.extend(range(start, end + 1))
            else:
                motion_ids.append(int(x))
        
        evaluator.evaluate_all_motions(motion_ids, args.output_dir)
        
    elif args.eval_all:
        # Evaluate all motions (limited to first 300)
        all_motion_ids = list(range(len(evaluator.original_keys)))
        print(f"📊 Evaluating all {len(all_motion_ids)} motions (limited to first 300)")
        evaluator.evaluate_all_motions(all_motion_ids, args.output_dir)
        
    else:
        print("❌ Please specify --motion_id, --motion_ids, or --eval_all")
        return
    
    print(f"\n🎉 Motion accuracy evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

'''
# Evaluate single motion in detail
python scripts/evaluate_motion_accuracy.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_id 0 \
    --output_dir ./outputs/evaluation


# Evaluate all motions (limited to first 300)
python scripts/evaluate_motion_accuracy.py \
    --config configs/agent.yaml \
    --checkpoint outputs/run_0_300_g1/best_model.ckpt \
    --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --eval_all \
    --max_motions 300 \
    --output_dir ./outputs/evaluation


'''
