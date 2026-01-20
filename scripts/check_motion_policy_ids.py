import sys
from pathlib import Path
import numpy as np
import argparse
import joblib

# Add parent dir so imports work similarly to your training script
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.data.motion_data_adapter import MotionDataAdapter

def check_policy_ids_fast(pkl_file: str, motion_id: int = None):
    """
    Fast check of policy IDs without loading/extracting motion features.
    Only loads the PKL file and checks for policy_id keys.
    """
    print(f"Loading PKL file (fast check, no feature extraction)...")
    motion_data_dict = joblib.load(pkl_file)
    motion_keys_all = list(motion_data_dict.keys())
    
    print(f"Total motions in file: {len(motion_keys_all)}")
    
    # Check policy IDs for all motions (fast - just checking keys)
    all_policy_ids = []
    for motion_key in motion_keys_all:
        motion_data = motion_data_dict[motion_key]
        if "policy_id" in motion_data and motion_data["policy_id"] is not None:
            policy_id_seq = np.asarray(motion_data["policy_id"], dtype=np.int64)
            all_policy_ids.append(policy_id_seq)
        else:
            all_policy_ids.append(None)
    
    # Find motions without policy IDs
    motions_without_policy_ids = [i for i, pid in enumerate(all_policy_ids) if pid is None]
    motions_with_policy_ids = [i for i, pid in enumerate(all_policy_ids) if pid is not None]
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total motions: {len(all_policy_ids)}")
    print(f"  Motions WITH policy IDs: {len(motions_with_policy_ids)}")
    print(f"  Motions WITHOUT policy IDs: {len(motions_without_policy_ids)}")
    print(f"{'='*60}")
    
    if motions_without_policy_ids:
        print(f"\nMotion IDs WITHOUT policy IDs ({len(motions_without_policy_ids)} total):")
        # Print in chunks for readability
        chunk_size = 20
        for i in range(0, len(motions_without_policy_ids), chunk_size):
            chunk = motions_without_policy_ids[i:i+chunk_size]
            print(f"  {chunk}")
    else:
        print("\nAll motions have policy IDs!")
    
    # If specific motion_id requested, show details
    if motion_id is not None:
        if motion_id < len(all_policy_ids):
            motion_policy_ids = all_policy_ids[motion_id]
            if motion_policy_ids is not None:
                unique_ids = np.unique(motion_policy_ids)
                print(f"\n{'='*60}")
                print(f"Motion {motion_id} details:")
                print(f"  HAS policy IDs: shape={motion_policy_ids.shape}, unique={unique_ids}")
                print(f"  Policy ID counts: {dict(zip(*np.unique(motion_policy_ids, return_counts=True)))}")
            else:
                print(f"\n{'='*60}")
                print(f"Motion {motion_id} details:")
                print(f"  DOES NOT have policy IDs (None)")
        else:
            print(f"\nMotion {motion_id} is out of bounds (valid range: 0-{len(all_policy_ids)-1})")
    
    return all_policy_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, 
                       default='/home/baekdh/dh_workspace/data_phc/data/amass/amass_train_w_policy_id/amass_train_w_policy_id.pkl')
    parser.add_argument('--motion_id', type=int, default=None, help='Check specific motion ID (optional)')
    parser.add_argument('--fast', action='store_true', help='Use fast mode (skip feature extraction, default)')
    parser.add_argument('--full', action='store_true', help='Use full mode (load all motion data with features)')
    args = parser.parse_args()
    
    pkl_file = args.pkl_file
    
    # Default to fast mode unless --full is specified
    if args.full:
        print("Using FULL mode (loading all motion data with features)...")
        # Use CPU for faster startup and small checks
        config = {'device': 'cpu'}
        adapter = MotionDataAdapter(config)
        
        print("Loading ALL motions (this may take a while)...")
        motions, end_indices, frame_size = adapter.load_motion_data(pkl_file, motion_ids=None)
        print(f"Total motions loaded: {len(end_indices)}")

        # Check per-motion policy IDs
        if hasattr(adapter, 'all_policy_ids_per_motion'):
            all_policy_ids = adapter.all_policy_ids_per_motion
            print(f"\nPer-motion policy IDs list length: {len(all_policy_ids)}")
            
            # Find motions without policy IDs
            motions_without_policy_ids = [i for i, pid in enumerate(all_policy_ids) if pid is None]
            motions_with_policy_ids = [i for i, pid in enumerate(all_policy_ids) if pid is not None]
            
            print(f"\n{'='*60}")
            print(f"Summary:")
            print(f"  Total motions: {len(all_policy_ids)}")
            print(f"  Motions WITH policy IDs: {len(motions_with_policy_ids)}")
            print(f"  Motions WITHOUT policy IDs: {len(motions_without_policy_ids)}")
            print(f"{'='*60}")
            
            if motions_without_policy_ids:
                print(f"\nMotion IDs WITHOUT policy IDs ({len(motions_without_policy_ids)} total):")
                # Print in chunks for readability
                chunk_size = 20
                for i in range(0, len(motions_without_policy_ids), chunk_size):
                    chunk = motions_without_policy_ids[i:i+chunk_size]
                    print(f"  {chunk}")
            else:
                print("\nAll motions have policy IDs!")
            
            # If specific motion_id requested, show details
            if args.motion_id is not None:
                motion_id = args.motion_id
                if motion_id < len(all_policy_ids):
                    motion_policy_ids = all_policy_ids[motion_id]
                    if motion_policy_ids is not None:
                        unique_ids = np.unique(motion_policy_ids)
                        print(f"\n{'='*60}")
                        print(f"Motion {motion_id} details:")
                        print(f"  HAS policy IDs: shape={motion_policy_ids.shape}, unique={unique_ids}")
                        print(f"  Policy ID counts: {dict(zip(*np.unique(motion_policy_ids, return_counts=True)))}")
                    else:
                        print(f"\n{'='*60}")
                        print(f"Motion {motion_id} details:")
                        print(f"  DOES NOT have policy IDs (None)")
                else:
                    print(f"\nMotion {motion_id} is out of bounds (valid range: 0-{len(all_policy_ids)-1})")
        else:
            print("all_policy_ids_per_motion not found in adapter")
    else:
        # Fast mode (default)
        check_policy_ids_fast(pkl_file, args.motion_id)

if __name__ == '__main__':
    main()



'''
Usage:
    # Fast mode (default) - only checks policy IDs without loading motion features
    python scripts/check_motion_policy_ids.py 
    
    # Fast mode with specific motion ID
    python scripts/check_motion_policy_ids.py --motion_id 42
    
    # Full mode - loads all motion data with features (slower, but matches training behavior)
    python scripts/check_motion_policy_ids.py --full
'''