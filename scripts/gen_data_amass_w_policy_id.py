#!/usr/bin/env python3
"""
Simple script to merge policy_id from CSV into AMASS PKL file.
Processes all motions from the PKL, matching with CSV by motion_id.

Usage:
    python scripts/gen_data_amass_w_policy_id.py \
      --amass_pkl /path/to/original.pkl \
      --csv_file /path/to/policy.csv \
      --out_pkl /path/to/output.pkl
"""

import argparse
import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def load_pkl(path: str):
    """Load PKL file using joblib."""
    return joblib.load(path)


def save_pkl(obj, path: str):
    """Save PKL file using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def get_motion_length(entry):
    """Get motion length from entry."""
    for key in ["root_trans_offset", "root_rot", "dof", "poses"]:
        if key in entry:
            arr = np.asarray(entry[key])
            if arr.ndim >= 1 and arr.shape[0] > 0:
                return int(arr.shape[0])
    return None


def parse_id_from_filename(path: str) -> int:
    """Parse numeric ID from filename like saved_desired_states_123_policy_ids.csv"""
    base = Path(path).name
    match = re.search(r'(\d+)', base)
    return int(match.group(1)) if match else None


def test_features():
    """Test function: Read merged PKL and export a single motion to CSV."""
    # pkl_path = "/home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl"
    pkl_path = "/home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl"
    motion_id = 0
    output_csv = "/home/baekdh/dh_workspace/data_phc/data/amass/test_motion_0_original.csv"
    fps = 30.0
    
    # Load PKL
    print(f"Loading PKL: {pkl_path}")
    pkl_obj = load_pkl(pkl_path)
    
    # Get motion entry
    if isinstance(pkl_obj, dict):
        motion_keys = list(pkl_obj.keys())
        if motion_id >= len(motion_keys):
            raise ValueError(f"Motion ID {motion_id} out of range (0-{len(motion_keys)-1})")
        motion_key = motion_keys[motion_id]
        entry = pkl_obj[motion_key]
    else:
        if motion_id >= len(pkl_obj):
            raise ValueError(f"Motion ID {motion_id} out of range (0-{len(pkl_obj)-1})")
        entry = pkl_obj[motion_id]
    
    if not isinstance(entry, dict):
        raise ValueError(f"Motion entry is not a dict")
    
    # Get motion length
    T = get_motion_length(entry)
    if T is None:
        raise ValueError(f"Cannot infer motion length")
    
    print(f"Motion {motion_id}: {T} frames")
    
    # Build CSV data with all features
    csv_data = {
        "motion_id": [motion_id] * T,
        "frame_idx": np.arange(T),
        "time": np.arange(T) / fps,
    }
    
    # Add policy_id if exists
    if "policy_id" in entry:
        csv_data["policy_id"] = np.asarray(entry["policy_id"])[:T]
    
    # Add all other fields
    for key, value in entry.items():
        if key == "policy_id":  # Already added
            continue
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == T:
            if arr.ndim == 1:
                csv_data[key] = arr
            elif arr.ndim == 2:
                # Flatten 2D arrays
                for j in range(arr.shape[1]):
                    csv_data[f"{key}_{j}"] = arr[:, j]
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to: {output_csv}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    

def main():
    parser = argparse.ArgumentParser(description="Merge policy_id from CSV into PKL")
    parser.add_argument("--amass_pkl", type=str, required=True, help="Original AMASS PKL file")
    parser.add_argument("--csv_file", type=str, required=True, help="CSV file with policy_id and motion_id columns")
    parser.add_argument("--out_pkl", type=str, required=True, help="Output PKL file")
    parser.add_argument("--policy_col", type=str, default="policy_id", help="Policy column name in CSV")
    parser.add_argument("--motion_id_col", type=str, default="motion_id", help="Motion ID column name in CSV")
    parser.add_argument("--pad_value", type=int, default=-1, help="Padding value if CSV shorter")
    
    args = parser.parse_args()
    
    # Load PKL
    print(f"Loading PKL: {args.amass_pkl}")
    pkl_obj = load_pkl(args.amass_pkl)
    
    # Check and report number of motions in PKL
    if isinstance(pkl_obj, dict):
        num_motions = len(pkl_obj)
        print(f"Found {num_motions} motions in PKL file (dict format)")
    else:
        num_motions = len(pkl_obj)
        print(f"Found {num_motions} motions in PKL file (list format)")
    
    # Load CSV(s) - handle both single file and directory
    csv_path = Path(args.csv_file)
    if csv_path.is_dir():
        # Load all CSV files from directory
        csv_files = sorted(glob.glob(str(csv_path / "*.csv")))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {args.csv_file}")
        print(f"Loading {len(csv_files)} CSV files from directory: {args.csv_file}")
        
        all_dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Try to extract motion_id from filename if not in CSV
            if args.motion_id_col not in df.columns:
                motion_id = parse_id_from_filename(csv_file)
                if motion_id is not None:
                    df[args.motion_id_col] = motion_id
            all_dfs.append(df)
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        # Single CSV file
        print(f"Loading CSV: {args.csv_file}")
        df = pd.read_csv(args.csv_file)
    
    if args.policy_col not in df.columns:
        raise ValueError(f"CSV missing column '{args.policy_col}'. Available: {list(df.columns)}")
    
    if args.motion_id_col not in df.columns:
        raise ValueError(f"CSV missing column '{args.motion_id_col}'. Available: {list(df.columns)}")
    
    # Group CSV by motion_id
    csv_by_motion = {}
    for motion_id, group in df.groupby(args.motion_id_col):
        policy_seq = group[args.policy_col].to_numpy().astype(np.int64)
        csv_by_motion[int(motion_id)] = policy_seq
    
    print(f"Found {len(csv_by_motion)} motions in CSV")
    
    # Process all motions in PKL
    if isinstance(pkl_obj, dict):
        motion_keys = list(pkl_obj.keys())
        motions_list = [(i, motion_keys[i], pkl_obj[motion_keys[i]]) for i in range(len(motion_keys))]
    else:
        motions_list = [(i, i, pkl_obj[i]) for i in range(len(pkl_obj))]
    
    total_motions = len(motions_list)
    matched = 0
    skipped_no_csv = 0
    skipped_invalid = 0
    
    for motion_id, motion_key, entry in motions_list:
        if not isinstance(entry, dict):
            skipped_invalid += 1
            continue
        
        # Get motion length
        T = get_motion_length(entry)
        if T is None:
            skipped_invalid += 1
            continue
        
        # Check if CSV has data for this motion
        if motion_id not in csv_by_motion:
            skipped_no_csv += 1
            continue
        
        # Get policy_id sequence from CSV
        policy_seq = csv_by_motion[motion_id]
        
        # Align lengths
        if len(policy_seq) > T:
            policy_seq = policy_seq[:T]
        elif len(policy_seq) < T:
            padded = np.full(T, args.pad_value, dtype=np.int64)
            padded[:len(policy_seq)] = policy_seq
            policy_seq = padded
        
        # Add policy_id to entry
        entry[args.policy_col] = policy_seq
        matched += 1
    
    # Save
    print(f"\nSaving to: {args.out_pkl}")
    save_pkl(pkl_obj, args.out_pkl)
    print(f"\nDone! Summary:")
    print(f"  Total motions in PKL: {total_motions}")
    print(f"  Motions with {args.policy_col} added: {matched}")
    if skipped_no_csv > 0:
        print(f"  Motions skipped (no CSV data): {skipped_no_csv}")
    if skipped_invalid > 0:
        print(f"  Motions skipped (invalid): {skipped_invalid}")


if __name__ == "__main__":
    main()
    # Uncomment below to run test_features() instead:
    # test_features()

'''
python scripts/gen_data_amass_w_policy_id.py \
  --amass_pkl /home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --csv_file /home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/single_option_framework \
  --out_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --policy_col "policy_id" 

'''