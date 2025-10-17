#!/usr/bin/env python3
"""
Generate PKL files in AMASS-like format using original AMASS global data as ground truth.
Uses motion_lib_g1_amass.py to extract local features from the original global data.
This matches the training data format where global is GT and local features are computed.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import sys
import joblib

# Add motion_vqvae to path (repo layout assumption)
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.data.motion_data_adapter import MotionDataAdapter
from scripts.vqvae_gen_init import (
    parse_motion_ids,
    convert_to_amass_global_from_local,
    generate_motion_pkl_files,
)


class AMASSGTGenerator:
    """Generate PKL files using original AMASS global data as ground truth."""

    def __init__(self, input_pkl_file: str, device: str = "cuda"):
        self.input_pkl_file = input_pkl_file
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load original AMASS data
        print(f"Loading original AMASS data from: {input_pkl_file}")
        self.original_motions = joblib.load(input_pkl_file)
        self.original_keys = list(self.original_motions.keys())
        print(f"Loaded {len(self.original_keys)} original motions")
        
        # Initialize motion data adapter for local feature extraction
        config = {"device": str(self.device)}
        self.motion_adapter = MotionDataAdapter(config)

    def generate_amass_gt_pkl(self, motion_ids: List[int], output_dir: Optional[str] = None) -> Tuple[List[str], str]:
        """Generate one PKL per motion id using original AMASS global data."""
        return generate_motion_pkl_files(
            motion_ids=motion_ids,
            original_keys=self.original_keys,
            motion_generator_func=self._generate_single_motion_gt,
            output_dir=output_dir or "./outputs/amass_gt_motions",
            filename_prefix="amass_gt_motion_",
            generation_type="AMASS GT Format"
        )

    def _generate_single_motion_gt(self, motion_id: int) -> Optional[Dict]:
        """Generate motion using original AMASS global data and local features from motion adapter."""
        try:
            original_amass = self.original_motions[self.original_keys[motion_id]]
            
            # Use motion adapter to extract local features (same as training)
            # Create a temporary single-motion dict for the adapter
            temp_motion_dict = {self.original_keys[motion_id]: original_amass}
            temp_pkl_path = f"/tmp/temp_motion_{motion_id}.pkl"
            joblib.dump(temp_motion_dict, temp_pkl_path)
            
            # Load motion data using the adapter to get local features
            mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(temp_pkl_path, [0])
            
            # Extract local features from the adapter output
            # mocap_data is a tensor of shape [total_frames, frame_size] where frame_size=50
            local_features = mocap_data.detach().cpu().numpy()  # Convert to numpy
            
            # Convert local features to AMASS-like global representation
            amass_motion = convert_to_amass_global_from_local(
                local_features=local_features,
                original_motion=original_amass,
                motion_id=motion_id,
            )
            
            # Clean up temporary file
            import os
            if os.path.exists(temp_pkl_path):
                os.remove(temp_pkl_path)
            
            return amass_motion
            
        except Exception as e:
            print(f"Error generating motion {motion_id}: {e}")
            import traceback
            traceback.print_exc()
            return None



def main():
    parser = argparse.ArgumentParser(description="Generate AMASS GT PKLs from original AMASS data")
    parser.add_argument("--input_pkl", type=str, required=True, help="Path to input AMASS PKL (dict)")
    parser.add_argument("--motion_ids", type=str, default="0", help='Comma/range list, e.g. "0,2,5-12"')
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save PKLs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()
    motion_ids = parse_motion_ids(args.motion_ids)

    generator = AMASSGTGenerator(
        input_pkl_file=args.input_pkl,
        device=args.device,
    )

    generated_files, out_dir = generator.generate_amass_gt_pkl(motion_ids, args.output_dir)

    print("\n=== Generation Complete ===")
    print(f"Output directory: {out_dir}")
    print(f"Total files: {len(generated_files)}")
    for p in generated_files:
        print(f"  - {p}")


if __name__ == "__main__":
    main()


'''
Example usage:

python scripts/generate_amass_gt_from_original.py \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0" \
  --output_dir ./outputs/amass_gt_motions

python scripts/generate_amass_gt_from_original.py \
  --input_pkl /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --motion_ids "0-20" \
  --output_dir ./outputs/amass_gt_motions
'''
