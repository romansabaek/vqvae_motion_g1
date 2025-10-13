#!/usr/bin/env python3
"""
Check the actual frame size of motion data
"""

import sys
import joblib
import numpy as np
from pathlib import Path

def check_frame_size(pkl_file: str, motion_id: int = 0):
    """Check the frame size of motion data."""
    
    print(f"🔍 Checking frame size for motion data")
    print(f"PKL file: {pkl_file}")
    print(f"Motion ID: {motion_id}")
    print("=" * 50)
    
    try:
        # Load PKL file
        motion_data = joblib.load(pkl_file)
        motion_data_keys = list(motion_data.keys())
        
        print(f"✅ PKL file loaded successfully!")
        print(f"📊 Total motions in file: {len(motion_data_keys)}")
        
        if motion_id >= len(motion_data_keys):
            print(f"❌ Motion ID {motion_id} out of range (max: {len(motion_data_keys)-1})")
            return
        
        # Get motion data
        motion_key = motion_data_keys[motion_id]
        motion_data_single = motion_data[motion_key]
        
        print(f"\n📋 Motion ID {motion_id}: {motion_key}")
        print(f"Keys: {list(motion_data_single.keys())}")
        
        # Check each component
        root_pos = motion_data_single['root_trans_offset']  # 3D
        root_rot = motion_data_single['root_rot']  # 4D quaternion
        dof_pos = motion_data_single['dof']  # N×D joints
        
        print(f"\n📊 Motion Data Components:")
        print(f"  root_trans_offset: {root_pos.shape} (3D position)")
        print(f"  root_rot: {root_rot.shape} (4D quaternion)")
        print(f"  dof: {dof_pos.shape} (joint positions)")
        
        # Calculate frame size
        root_pos_dim = root_pos.shape[1]  # 3
        root_rot_dim = root_rot.shape[1]  # 4
        dof_dim = dof_pos.shape[1]  # N
        
        # With velocities (computed during motion processing)
        root_vel_dim = root_pos_dim  # 3
        root_ang_vel_dim = root_rot_dim  # 4 (or 3 if converted to expmap)
        dof_vel_dim = dof_dim  # N
        
        # Total frame size
        frame_size_with_quat = root_pos_dim + root_rot_dim + root_vel_dim + root_ang_vel_dim + dof_dim + dof_vel_dim
        frame_size_with_expmap = root_pos_dim + 3 + root_vel_dim + 3 + dof_dim + dof_vel_dim  # expmap is 3D
        
        print(f"\n🎯 Frame Size Calculation:")
        print(f"  Root position: {root_pos_dim}D")
        print(f"  Root rotation (quat): {root_rot_dim}D")
        print(f"  Root velocity: {root_vel_dim}D")
        print(f"  Root angular velocity: {root_ang_vel_dim}D")
        print(f"  DOF positions: {dof_dim}D")
        print(f"  DOF velocities: {dof_vel_dim}D")
        print(f"  Total (with quaternion): {frame_size_with_quat}D")
        print(f"  Total (with expmap): {frame_size_with_expmap}D")
        
        print(f"\n💡 Expected frame size: {frame_size_with_quat}D (with quaternion)")
        print(f"💡 Or {frame_size_with_expmap}D (with expmap conversion)")
        
    except Exception as e:
        print(f"❌ Error checking frame size: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_frame_size.py <path_to_pkl_file> [motion_id]")
        print("Example: python check_frame_size.py data/motions.pkl")
        print("Example: python check_frame_size.py data/motions.pkl 0")
        return
    
    pkl_file = sys.argv[1]
    motion_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    if not Path(pkl_file).exists():
        print(f"❌ File not found: {pkl_file}")
        return
    
    check_frame_size(pkl_file, motion_id)

if __name__ == "__main__":
    main()
