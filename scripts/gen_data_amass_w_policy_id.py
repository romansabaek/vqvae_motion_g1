#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge per-timestep policy_id CSV logs (30Hz) into an AMASS PKL.

Goal:
- Keep original AMASS pkl structure intact
- Add policy_id sequence for each motion clip (per timestep)
- Save as a new pkl (e.g., amass_policy_id.pkl)

Supported CSV formats:
1) saved_desired_states_{ID}_policy_ids.csv
   - must contain column: policy_id
   - optional: time
   - optional: original_key (string key for matching)
2) Any CSV with columns:
   - policy_id
   - and either:
       - motion_id (int), or
       - original_key (string)

Matching strategy (priority):
A) If CSV contains 'original_key' and PKL provides comparable key -> match by key
B) Else if CSV contains 'motion_id' -> match by motion_id
C) Else parse integer ID from filename with regex patterns

Handling length mismatch:
- By default: truncate to min(T_pkl, T_csv) and pad with -1 to T_pkl if needed
- (This is safest; resampling could be added later if necessary)

Usage example:
python merge_amass_policy_id.py \
  --amass_pkl "/path/to/amass.pkl" \
  --csv_glob "/path/to/policy_logs/*.csv" \
  --out_pkl "/path/to/amass_policy_id.pkl" \
  --policy_col "policy_id" \
  --pad_value -1

Notes:
- This script does NOT modify existing arrays; it only injects 'policy_id' fields.
- For list-based PKL: each element (dict-like) gets policy_id
- For dict-based PKL: it tries common containers: ['motions','clips','data','sequences'].
"""

import os
import re
import glob
import pickle
import argparse
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib


# ---------------------------
# Helpers: load/save PKL
# ---------------------------
def load_pkl(path: str) -> Any:
    """
    Load PKL file. Tries joblib first (common for AMASS data), then falls back to pickle.
    """
    try:
        return joblib.load(path)
    except Exception:
        # Fallback to pickle if joblib fails
        with open(path, "rb") as f:
            return pickle.load(f)


def save_pkl(obj: Any, path: str) -> None:
    """
    Save PKL file using joblib (consistent with AMASS data format).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(obj, path)


# ---------------------------
# Find motions container
# ---------------------------
def find_motion_container(pkl_obj: Any) -> Tuple[str, Union[List[Any], Dict[str, Any]], Optional[List[str]]]:
    """
    Returns (container_type, motions_data, motion_keys)
    container_type:
      - "list"  : pkl_obj is a list of motion dicts/objects
      - "dict:<key>" : pkl_obj is a dict and motions_list = pkl_obj[key]
      - "dict:amass" : pkl_obj is a dict where keys are motion identifiers and values are motion dicts (AMASS format)
    
    motion_keys: list of motion keys/identifiers (None for list format, list of keys for dict formats)
    """
    if isinstance(pkl_obj, list):
        return ("list", pkl_obj, None)

    if isinstance(pkl_obj, dict):
        # Check for nested list structure (common in some datasets)
        candidate_keys = ["motions", "clips", "data", "sequences", "items"]
        for k in candidate_keys:
            if k in pkl_obj and isinstance(pkl_obj[k], list):
                return (f"dict:{k}", pkl_obj[k], None)
        
        # Check if this is AMASS format: dict where values are motion dicts
        # Sample a few values to check if they look like motion dicts
        if len(pkl_obj) > 0:
            sample_values = list(pkl_obj.values())[:3]
            # If all sample values are dicts, assume AMASS format
            if all(isinstance(v, dict) for v in sample_values):
                return ("dict:amass", pkl_obj, list(pkl_obj.keys()))

    raise ValueError(
        "Unsupported PKL structure. Expected a list, a dict containing a list under one of "
        "['motions','clips','data','sequences','items'], or a dict where values are motion dicts (AMASS format)."
    )


# ---------------------------
# Infer length T from a motion entry
# ---------------------------
def infer_T_from_motion_entry(entry: Any) -> Optional[int]:
    """
    Try to infer the number of timesteps T for a single motion clip entry.
    Works best when entry is dict with arrays like 'poses', 'qpos', 'trans', etc.

    Returns T or None if cannot infer.
    """
    if entry is None:
        return None

    # If it's a dict, search arrays with a clear time dimension
    if isinstance(entry, dict):
        # prioritize common AMASS-ish keys
        priority_keys = ["poses", "pose", "qpos", "trans", "translations", "joints", "root_trans"]
        for k in priority_keys:
            if k in entry:
                v = entry[k]
                try:
                    arr = np.asarray(v)
                    if arr.ndim >= 1 and arr.shape[0] > 0:
                        return int(arr.shape[0])
                except Exception:
                    pass

        # fallback: first array-like with ndim>=1
        for k, v in entry.items():
            try:
                arr = np.asarray(v)
                if arr.ndim >= 1 and arr.shape[0] > 0:
                    return int(arr.shape[0])
            except Exception:
                continue

    # If it's an object with attributes
    for attr in ["poses", "pose", "qpos", "trans", "translations"]:
        if hasattr(entry, attr):
            v = getattr(entry, attr)
            try:
                arr = np.asarray(v)
                if arr.ndim >= 1 and arr.shape[0] > 0:
                    return int(arr.shape[0])
            except Exception:
                pass

    return None


def get_entry_key(entry: Any) -> Optional[str]:
    """
    Try to extract a stable key (e.g., 'original_key') from a motion entry.
    """
    if isinstance(entry, dict):
        for k in ["original_key", "key", "name", "seq_name", "clip_name", "id_str"]:
            if k in entry and isinstance(entry[k], str):
                return entry[k]
    return None


def get_entry_motion_id(entry: Any) -> Optional[int]:
    """
    Try to extract motion_id (integer) from a motion entry.
    """
    if isinstance(entry, dict):
        for k in ["motion_id", "id", "clip_id", "seq_id"]:
            if k in entry:
                try:
                    return int(entry[k])
                except Exception:
                    continue
    return None


# ---------------------------
# CSV parsing and mapping
# ---------------------------
def parse_id_from_filename(path: str) -> Optional[int]:
    """
    Parse numeric id from filenames like:
      saved_desired_states_0_policy_ids.csv
      codebook_sequence_000.csv
    """
    base = os.path.basename(path)

    patterns = [
        r"saved_desired_states_(\d+)_policy_ids\.csv",
        r"codebook_sequence_(\d+)\.csv",
        r".*?(\d+).*?\.csv",  # very loose fallback
    ]
    for pat in patterns:
        m = re.match(pat, base)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def load_policy_csv(
    csv_path: str,
    policy_col: str = "policy_id",
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with:
      - policy: np.ndarray (T,)
      - motion_id: Optional[int]
      - original_key: Optional[str]
      - time: Optional[np.ndarray]
    Returns None if the required column is missing.
    """
    df = pd.read_csv(csv_path)
    if policy_col not in df.columns:
        print(f"[WARN] CSV {csv_path} missing required column '{policy_col}'. Available columns: {list(df.columns)}. Skipping.")
        return None

    # policy sequence
    policy = df[policy_col].to_numpy()

    # optional fields
    motion_id = None
    if "motion_id" in df.columns:
        # allow constant or per-row; take first non-nan
        vals = df["motion_id"].dropna().unique().tolist()
        if len(vals) > 0:
            try:
                motion_id = int(vals[0])
            except Exception:
                motion_id = None

    original_key = None
    if "original_key" in df.columns:
        vals = df["original_key"].dropna().unique().tolist()
        if len(vals) > 0 and isinstance(vals[0], str):
            original_key = vals[0]

    time = None
    if "time" in df.columns:
        try:
            time = df["time"].to_numpy()
        except Exception:
            time = None

    return {
        "policy": policy,
        "motion_id": motion_id,
        "original_key": original_key,
        "time": time,
        "csv_path": csv_path,
        "n_rows": len(df),
        "columns": list(df.columns),
    }


def build_policy_maps(
    csv_glob: str,
    policy_col: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Build two maps from CSVs:
      - by_id: int -> csv_info
      - by_key: str -> csv_info
    Skips CSV files that don't have the required column.
    """
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No CSV matched: {csv_glob}")

    by_id: Dict[int, Dict[str, Any]] = {}
    by_key: Dict[str, Dict[str, Any]] = {}
    skipped = 0

    for f in files:
        info = load_policy_csv(f, policy_col=policy_col)
        
        # Skip if file doesn't have required column
        if info is None:
            skipped += 1
            continue

        # key-based map
        if info["original_key"] is not None:
            by_key[info["original_key"]] = info

        # id-based map
        mid = info["motion_id"]
        if mid is None:
            mid = parse_id_from_filename(f)
        if mid is not None:
            by_id[mid] = info

    if skipped > 0:
        print(f"[INFO] Skipped {skipped} CSV file(s) missing required column '{policy_col}'")
    
    if len(by_id) == 0 and len(by_key) == 0:
        raise ValueError(f"No valid CSV files found with column '{policy_col}'. Check your CSV files and --policy_col argument.")

    return by_id, by_key


# ---------------------------
# Merge logic
# ---------------------------
def align_and_inject_policy(
    entry: Any,
    policy_seq: np.ndarray,
    pad_value: int = -1,
    field_name: str = "policy_id",
) -> Dict[str, Any]:
    """
    Inject policy into entry (dict-like assumed).
    Returns a small report dict for logging.
    """
    T = infer_T_from_motion_entry(entry)
    if T is None:
        # If we can't infer T, just store the raw sequence
        T = len(policy_seq)

    policy_seq = np.asarray(policy_seq).astype(np.int64)

    if len(policy_seq) == T:
        aligned = policy_seq
        mode = "exact"
    elif len(policy_seq) > T:
        aligned = policy_seq[:T]
        mode = "truncate_csv"
    else:
        # policy shorter than motion -> pad to T
        aligned = np.full((T,), int(pad_value), dtype=np.int64)
        aligned[: len(policy_seq)] = policy_seq
        mode = "pad_csv"

    if not isinstance(entry, dict):
        raise ValueError("Motion entry is not dict-like; please convert your PKL motions to dict entries first.")

    entry[field_name] = aligned
    return {
        "T_motion": T,
        "T_policy": int(len(policy_seq)),
        "mode": mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass_pkl", type=str, required=True, help="Path to original AMASS pkl")
    ap.add_argument("--csv_glob", type=str, required=True, help="Glob for per-motion policy csv files")
    ap.add_argument("--out_pkl", type=str, required=True, help="Output pkl path (will keep original content + policy_id)")
    ap.add_argument("--policy_col", type=str, default="policy_id", help="Column name in CSV for policy id")
    ap.add_argument("--field_name", type=str, default="policy_id", help="Field name to insert into PKL entries")
    ap.add_argument("--pad_value", type=int, default=-1, help="Padding value when CSV shorter than motion length")
    ap.add_argument("--dry_run", action="store_true", help="If set, do not write output pkl")
    args = ap.parse_args()

    # Load
    pkl_obj = load_pkl(args.amass_pkl)
    container_type, motions_data, motion_keys = find_motion_container(pkl_obj)
    print(f"[OK] Loaded PKL: {args.amass_pkl}")
    
    # Handle different container types
    if container_type == "dict:amass":
        # AMASS format: motions_data is the dict itself, motion_keys are the keys
        num_motions = len(motions_data)
        print(f"[OK] Motion container: {container_type}, num_motions={num_motions}")
        # Convert to list of (key, entry) tuples for iteration
        motions_list = [(key, motions_data[key]) for key in motion_keys]
    else:
        # List or nested dict format
        motions_list = [(i, entry) for i, entry in enumerate(motions_data)]
        motion_keys = None
        num_motions = len(motions_list)
        print(f"[OK] Motion container: {container_type}, num_motions={num_motions}")

    # Build CSV maps
    by_id, by_key = build_policy_maps(args.csv_glob, args.policy_col)
    print(f"[OK] Loaded policy CSVs: by_id={len(by_id)}, by_key={len(by_key)}")

    matched = 0
    unmatched = 0

    reports = []
    for i, (motion_identifier, entry) in enumerate(motions_list):
        if not isinstance(entry, dict):
            # You can extend here if your motion entries are objects, but most AMASS PKLs use dicts.
            print(f"[WARN] motion[{i}] is not a dict; skipping.")
            unmatched += 1
            continue

        # Try key match first
        # For AMASS format, motion_identifier is the dict key (string)
        # For other formats, try to get key from entry
        key = None
        if container_type == "dict:amass" and isinstance(motion_identifier, str):
            # Use the dict key directly for AMASS format
            key = motion_identifier
            # Also check if entry has an original_key that matches
            entry_key = get_entry_key(entry)
            if entry_key is not None:
                key = entry_key  # Prefer entry's original_key if available
        else:
            key = get_entry_key(entry)
        
        info = None
        if key is not None and key in by_key:
            info = by_key[key]

        # Then try motion_id match
        if info is None:
            mid = get_entry_motion_id(entry)
            if mid is None:
                # For AMASS format, try parsing ID from the key
                if container_type == "dict:amass" and isinstance(motion_identifier, str):
                    mid = parse_id_from_filename(motion_identifier)
                if mid is None:
                    mid = i  # fallback: index-based
            if mid in by_id:
                info = by_id[mid]

        if info is None:
            unmatched += 1
            continue

        rep = align_and_inject_policy(
            entry=entry,
            policy_seq=info["policy"],
            pad_value=args.pad_value,
            field_name=args.field_name,
        )
        rep.update({
            "motion_index": i,
            "motion_identifier": motion_identifier if container_type == "dict:amass" else i,
            "matched_by": "key" if (key is not None and key in by_key) else "id_or_index",
            "csv": os.path.basename(info["csv_path"]),
            "original_key": info["original_key"],
            "csv_motion_id": info["motion_id"],
        })
        reports.append(rep)
        matched += 1

    print(f"[DONE] matched={matched}, unmatched={unmatched}")

    # Print a small summary of alignment modes
    if reports:
        modes = {}
        for r in reports:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        print("[Summary] alignment modes:", modes)

        # show a few examples
        print("[Examples] first 5 merged entries:")
        for r in reports[:5]:
            print("  ", r)

    if args.dry_run:
        print("[DRY RUN] Not saving output.")
        return

    # Save
    save_pkl(pkl_obj, args.out_pkl)
    print(f"[OK] Saved merged PKL to: {args.out_pkl}")


if __name__ == "__main__":
    main()


'''

python scripts/gen_data_amass_w_policy_id.py \
  --amass_pkl /home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
  --csv_glob "/home/baekdh/dh_workspace/hrl/humanoidverse/data/motions/single_option_framework/*.csv" \
  --out_pkl /home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl \
  --policy_col "policy_id" \
  --pad_value -1

'''