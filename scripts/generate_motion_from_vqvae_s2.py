#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Generate PKL files in AMASS-like format (XYZW quaternion) from a trained Motion-VQVAE.
Global/world export only: integrates local root deltas to world (AMASS-like globals).
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import joblib
import numpy as np
import torch
import sys
import pandas as pd

# Add motion_vqvae to path (repo layout assumption)
sys.path.append(str(Path(__file__).parent.parent))

from scripts.vqvae_gen_init import (
    load_config_and_agent,
    load_original_pkl,
    infer_frame_size,
    initialize_model,
    ensure_stats,
    parse_motion_ids,
    quat_mul_xyzw,
    quat_rotate_xyzw_numpy,
    convert_to_amass_global_from_local,
    generate_motion_pkl_files,
)


class AMASSFormatGenerator:
    """Generate PKL files in AMASS-like global format from a trained VQVAE model."""

    def __init__(self, config_path: str, checkpoint_path: str, input_pkl_file: str, eval_stride: int = None):
        self.input_pkl_file = input_pkl_file

        # Shared init
        self.config, self.agent, self.motion_adapter = load_config_and_agent(config_path, checkpoint_path)

        # Allow overriding evaluation stride for overlapped reconstruction
        if eval_stride is not None:
            self.config["eval_stride"] = int(eval_stride)
            self.agent.config["eval_stride"] = int(eval_stride)

        # Load original AMASS data
        print(f"Loading original AMASS data from: {input_pkl_file}")
        self.original_motions, self.original_keys = load_original_pkl(input_pkl_file)
        print(f"Loaded {len(self.original_keys)} original motions")

        # Load multiple motions for proper normalization statistics (same as eval_vqvae.py)
        max_motions_for_stats = min(300, len(self.original_keys))
        subset_motion_ids = list(range(max_motions_for_stats))
        print(f"Loading first {max_motions_for_stats} motions for normalization stats...")
        
        mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
            input_pkl_file, subset_motion_ids
        )
        
        print(f"Loaded data for stats: shape={mocap_data.shape}, frame_size={frame_size}")
        print(f"Using device: {self.agent.device}")

        # Calculate normalization stats and initialize model (same as eval_vqvae.py)
        self.agent.frame_size = int(frame_size)
        self.frame_size = int(frame_size)  # Store for later use
        ensure_stats(self.agent, mocap_data)
        initialize_model(self.agent, self.config, self.frame_size, checkpoint_path)

    def generate_amass_format_pkl(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Generate one PKL per motion id (global/world export)."""
        return generate_motion_pkl_files(
            motion_ids=motion_ids,
            original_keys=self.original_keys,
            motion_generator_func=self._generate_single_motion,
            output_dir=output_dir or "./outputs/vqvae_motions",
            filename_prefix="vqvae_motion_",
            generation_type="AMASS Format"
        )

    def extract_codebook_sequences(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Extract codebook sequences and full motion data for given motion IDs and save to CSV files.
        
        The CSV files will contain data at 30Hz (original motion frequency):
        - Codebook indices mapped to each motion frame (30Hz)
        - Reconstructed motion features for each frame (30Hz)
        - Original motion features for each frame (30Hz)
        - Reconstruction errors for each frame (30Hz)
        - Codebook change indicators (when codebook index changes at 30Hz)
        - Metadata (motion_id, original_key, etc.)
        
        Note: VQVAE processes motion in windows, but this maps codebook indices
        to each individual frame at the original 30Hz frequency for detailed analysis.
        """
        print(f"\n=== Extracting Codebook Sequences ===")
        out_dir = Path(output_dir or "./outputs/codebook_sequences")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated_files: List[str] = []

        for motion_id in motion_ids:
            if motion_id < 0 or motion_id >= len(self.original_keys):
                print(f"Warning: Motion ID {motion_id} out of range. Skipping.")
                continue

            print(f"Extracting codebook sequence for motion ID {motion_id}...")
            codebook_data = self._extract_single_codebook_sequence(motion_id)

            if codebook_data is None:
                print(f"  ❌ Failed to extract codebook sequence for motion {motion_id}")
                continue

            # Save to CSV with all motion data at each step
            original_key = self.original_keys[motion_id]
            out_path = out_dir / f"codebook_sequence_{motion_id:03d}.csv"
            
            # Get motion data
            reconstructed_motion = codebook_data['reconstructed_motion']
            original_motion = codebook_data['original_motion']
            codebook_sequence = codebook_data['codebook_sequence']
            
            # The codebook sequence represents windows, not individual frames
            # We need to map codebook indices to each original motion frame (30Hz)
            motion_length = min(reconstructed_motion.shape[0], original_motion.shape[0])
            codebook_length = len(codebook_sequence)
            
            print(f"    Motion data shape: {reconstructed_motion.shape}, Original shape: {original_motion.shape}")
            print(f"    Codebook sequence length: {codebook_length} (windows)")
            print(f"    Original motion length: {motion_length} (frames at 30Hz)")
            
            # Map codebook indices to each motion frame
            # Each codebook index represents a window, so we need to repeat it for all frames in that window
            window_size = self.config.get('window_size', 64)  # Default window size
            codebook_per_frame = np.zeros(motion_length, dtype=np.int32)
            
            # Calculate how many frames each codebook index should cover
            frames_per_codebook = motion_length // codebook_length
            remainder_frames = motion_length % codebook_length
            
            frame_idx = 0
            for i, codebook_idx in enumerate(codebook_sequence):
                # Distribute remainder frames across first few codebook indices
                current_frames = frames_per_codebook + (1 if i < remainder_frames else 0)
                codebook_per_frame[frame_idx:frame_idx + current_frames] = codebook_idx
                frame_idx += current_frames
            
            # Use original motion length for the sequence
            sequence_length = motion_length
            
            # Prepare data for DataFrame
            data_dict = {
                'frame_idx': range(sequence_length),
                'codebook_idx': codebook_per_frame,  # Use mapped codebook indices per frame
                'motion_id': [motion_id] * sequence_length,
                'original_key': [original_key] * sequence_length,
                'sequence_length': [sequence_length] * sequence_length
            }
            
            # Add each feature dimension as a separate column
            for feat_idx in range(reconstructed_motion.shape[1]):
                data_dict[f'reconstructed_feat_{feat_idx}'] = reconstructed_motion[:, feat_idx]
                data_dict[f'original_feat_{feat_idx}'] = original_motion[:, feat_idx]
            
            # Calculate reconstruction error for each frame
            reconstruction_errors = np.abs(reconstructed_motion - original_motion)
            for feat_idx in range(reconstruction_errors.shape[1]):
                data_dict[f'reconstruction_error_feat_{feat_idx}'] = reconstruction_errors[:, feat_idx]
            
            # Calculate total reconstruction error per frame
            data_dict['total_reconstruction_error'] = np.sum(reconstruction_errors, axis=1)
            
            # Calculate codebook change indicators (when codebook index changes)
            codebook_changes = np.diff(codebook_per_frame, prepend=codebook_per_frame[0])
            data_dict['codebook_changed'] = codebook_changes != 0
            
            df = pd.DataFrame(data_dict)
            df.to_csv(out_path, index=False)
            generated_files.append(str(out_path))
            print(f"  ✅ Saved: {original_key} -> {out_path}")
            print(f"    Motion frames (30Hz): {sequence_length}")
            print(f"    Motion features: {reconstructed_motion.shape[1]}")
            print(f"    Codebook windows: {codebook_length}")
            print(f"    Codebook changes: {np.sum(codebook_changes != 0)}")

        print(f"\nSaved {len(generated_files)} codebook sequence files to: {out_dir}")
        return generated_files, str(out_dir)

    # ---- internal helpers ----

    def _generate_single_motion(self, motion_id: int) -> Optional[Dict]:
        """Reconstruct a single motion and convert to AMASS-like global output."""
        try:
            # Load MVQ data for this motion only
            mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
                self.input_pkl_file, [motion_id]
            )
            # Move to model device
            if isinstance(mocap_data, torch.Tensor):
                mocap_data = mocap_data.to(self.agent.device, non_blocking=True)
            if isinstance(end_indices, torch.Tensor):
                end_indices = end_indices.to(self.agent.device)

            self.agent.mocap_data = mocap_data
            self.agent.end_indices = end_indices
            self.agent.frame_size = int(frame_size)


            # Reconstruct with the VQVAE
            with torch.no_grad():
                idx = torch.tensor(0, device=self.agent.device)
                reconstructed_motion, original_seq, _codebook = self.agent.evaluate_policy_rec(idx)

            # Convert to AMASS-like global representation
            mvq = reconstructed_motion.detach().cpu().numpy().astype(np.float32)
            original_amass = self.original_motions[self.original_keys[motion_id]]
            return convert_to_amass_global_from_local(
                local_features=mvq,
                original_motion=original_amass,
                motion_id=motion_id,
            )
        except Exception as e:
            print(f"Error generating motion {motion_id}: {e}")
 
            return None

    def _extract_single_codebook_sequence(self, motion_id: int) -> Optional[Dict]:
        """Extract codebook sequence and motion data for a single motion."""
        try:
            # Load MVQ data for this motion only
            mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(
                self.input_pkl_file, [motion_id]
            )
            # Move to model device
            if isinstance(mocap_data, torch.Tensor):
                mocap_data = mocap_data.to(self.agent.device, non_blocking=True)
            if isinstance(end_indices, torch.Tensor):
                end_indices = end_indices.to(self.agent.device)

            self.agent.mocap_data = mocap_data
            self.agent.end_indices = end_indices
            self.agent.frame_size = int(frame_size)

            # Extract codebook sequence with the VQVAE
            with torch.no_grad():
                idx = torch.tensor(0, device=self.agent.device)
                reconstructed_motion, original_seq, codebook_seq = self.agent.evaluate_policy_rec(idx)

            # Convert to numpy and return
            codebook_sequence = codebook_seq.detach().cpu().numpy().astype(np.int32)
            reconstructed_motion_np = reconstructed_motion.detach().cpu().numpy().astype(np.float32)
            original_seq_np = original_seq.detach().cpu().numpy().astype(np.float32)
            
            return {
                'codebook_sequence': codebook_sequence,
                'reconstructed_motion': reconstructed_motion_np,
                'original_motion': original_seq_np,
                'sequence_length': len(codebook_sequence),
                'motion_id': motion_id,
                'original_key': self.original_keys[motion_id]
            }
        except Exception as e:
            print(f"Error extracting codebook sequence for motion {motion_id}: {e}")
            return None





def main():
    parser = argparse.ArgumentParser(description="Generate AMASS-like PKLs from trained VQVAE")
    parser.add_argument("--config", type=str, default="configs/agent.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt/.pt)")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to input AMASS PKL (dict)")
    parser.add_argument("--motion_ids", type=str, default="0", help='Comma/range list, e.g. "0,2,5-12"')
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save PKLs")
    # global-only export; no --space option
    parser.add_argument("--eval_stride", type=int, default=None, help="Stride for overlapped reconstruction (default: window_size//2)")
    
    # Codebook extraction options
    parser.add_argument("--extract_codebook", action="store_true", help="Extract codebook sequences to CSV files")
    parser.add_argument("--codebook_output_dir", type=str, default=None, help="Directory to save codebook CSV files")

    args = parser.parse_args()
    motion_ids = parse_motion_ids(args.motion_ids)

    generator = AMASSFormatGenerator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_pkl_file=args.input_pkl,
        eval_stride=args.eval_stride,
    )

    if args.extract_codebook:
        # Extract codebook sequences
        codebook_files, codebook_dir = generator.extract_codebook_sequences(
            motion_ids, args.codebook_output_dir
        )
        
        print("\n=== Codebook Extraction Complete ===")
        print(f"Codebook output directory: {codebook_dir}")
        print(f"Total codebook files: {len(codebook_files)}")
        for p in codebook_files:
            print(f"  - {p}")
    else:
        # Generate AMASS format PKLs (original functionality)
        generated_files, out_dir = generator.generate_amass_format_pkl(motion_ids, args.output_dir)

        print("\n=== Generation Complete ===")
        print(f"Output directory: {out_dir}")
        print("Space: global")
        print(f"Total files: {len(generated_files)}")
        for p in generated_files:
            print(f"  - {p}")


if __name__ == "__main__":
    main()


'''
# Example usage for generating AMASS format PKLs:
python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "1" 

# Example usage for extracting codebook sequences with full motion data to CSV:
python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300_v1/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "1" \
  --extract_codebook \
  --codebook_output_dir ./outputs/codebook_sequences

# Example usage for multiple motion IDs:
python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300_v1/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0,1,2,5-10" \
  --extract_codebook
'''
