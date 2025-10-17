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





def main():
    parser = argparse.ArgumentParser(description="Generate AMASS-like PKLs from trained VQVAE")
    parser.add_argument("--config", type=str, default="configs/agent.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt/.pt)")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to input AMASS PKL (dict)")
    parser.add_argument("--motion_ids", type=str, default="0", help='Comma/range list, e.g. "0,2,5-12"')
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save PKLs")
    # global-only export; no --space option
    parser.add_argument("--eval_stride", type=int, default=None, help="Stride for overlapped reconstruction (default: window_size//2)")

    args = parser.parse_args()
    motion_ids = parse_motion_ids(args.motion_ids)

    generator = AMASSFormatGenerator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        input_pkl_file=args.input_pkl,
        eval_stride=args.eval_stride,
    )

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
checkpoints/best_model.ckpt \


python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "1" 



python scripts/generate_motion_from_vqvae_s2.py \
  --config configs/agent.yaml \
  --checkpoint outputs/run_0_300_v2/best_model.ckpt \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "1" 


'''
