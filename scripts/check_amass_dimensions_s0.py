#!/usr/bin/env python3
"""
Check dimensions of all keys in AMASS retargeted data for G1 humanoid.
Analyzes the structure and dimensions of motion data to understand the format.
"""

import numpy as np
import joblib
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd


class AMASSDimensionChecker:
    """Check dimensions and structure of AMASS motion data."""
    
    def __init__(self, pkl_file_path: str):
        """Initialize with AMASS PKL file path."""
        self.pkl_file_path = pkl_file_path
        self.motion_data = None
        self.motion_keys = None
        
    def load_and_analyze(self, motion_ids: list = None, max_motions: int = 10):
        """Load and analyze AMASS motion data dimensions."""
        print(f"🔍 Loading AMASS data from: {self.pkl_file_path}")
        
        try:
            # Load PKL file
            self.motion_data = joblib.load(self.pkl_file_path)
            self.motion_keys = list(self.motion_data.keys())
            
            print(f"✅ Successfully loaded {len(self.motion_keys)} motions")
            
            # Analyze dimensions
            if motion_ids is None:
                motion_ids = list(range(min(max_motions, len(self.motion_keys))))
            
            print(f"📊 Analyzing dimensions for {len(motion_ids)} motions...")
            
            # Analyze each motion
            all_analyses = []
            for motion_id in motion_ids:
                if motion_id >= len(self.motion_keys):
                    print(f"⚠️  Motion ID {motion_id} out of range. Skipping.")
                    continue
                
                motion_key = self.motion_keys[motion_id]
                motion_dict = self.motion_data[motion_key]
                
                print(f"\n📋 Motion {motion_id}: {motion_key}")
                analysis = self._analyze_single_motion(motion_dict, motion_id, motion_key)
                all_analyses.append(analysis)
            
            # Generate summary report
            self._generate_summary_report(all_analyses)
            
            return all_analyses
            
        except Exception as e:
            print(f"❌ Error loading AMASS data: {e}")
            return None
    
    def _analyze_single_motion(self, motion_dict: Dict[str, Any], motion_id: int, motion_key: str) -> Dict:
        """Analyze dimensions of a single motion."""
        print(f"  Keys: {list(motion_dict.keys())}")
        
        analysis = {
            'motion_id': motion_id,
            'motion_key': motion_key,
            'keys': list(motion_dict.keys()),
            'dimensions': {},
            'data_types': {},
            'frame_count': 0
        }
        
        for key, data in motion_dict.items():
            try:
                # Convert to numpy array if possible
                if isinstance(data, (list, tuple)):
                    data_array = np.array(data)
                elif isinstance(data, np.ndarray):
                    data_array = data
                else:
                    data_array = np.array([data])  # Single value
                
                # Analyze dimensions
                shape = data_array.shape
                dtype = data_array.dtype
                
                analysis['dimensions'][key] = shape
                analysis['data_types'][key] = str(dtype)
                
                # Determine frame count (use first numeric array with time dimension)
                if analysis['frame_count'] == 0 and data_array.ndim >= 1 and np.issubdtype(dtype, np.number):
                    analysis['frame_count'] = data_array.shape[0]
                
                print(f"    {key}: {shape} ({dtype})")
                
                # Analyze quaternion order if this is root_rot
                if key == 'root_rot' and data_array.ndim == 2 and data_array.shape[1] == 4:
                    self._analyze_quaternion_order(data_array)
                
            except Exception as e:
                print(f"    {key}: Error analyzing - {e}")
                analysis['dimensions'][key] = f"Error: {e}"
                analysis['data_types'][key] = str(type(data))
        
        print(f"  📏 Total frames: {analysis['frame_count']}")
        
        return analysis
    
    def _analyze_quaternion_order(self, quat_data: np.ndarray):
        """Simple quaternion order analysis."""
        if quat_data.shape[1] != 4:
            print(f"      ⚠️  Invalid quaternion dimensions: {quat_data.shape[1]} (expected 4)")
            return
        
        # Sample a few quaternions
        sample_size = min(5, quat_data.shape[0])
        sample_quats = quat_data[:sample_size]
        
        print(f"      🔍 Quaternion Order Analysis (sample of {sample_size}):")
        
        # Check component magnitudes
        w0_mag = np.mean(np.abs(sample_quats[:, 0]))  # First component
        w3_mag = np.mean(np.abs(sample_quats[:, 3]))  # Last component
        
        print(f"        First component (index 0) avg magnitude: {w0_mag:.4f}")
        print(f"        Last component (index 3) avg magnitude: {w3_mag:.4f}")
        
        # Simple heuristic: which component is closer to 1?
        if w0_mag > 0.8:
            print(f"        🎯 Likely format: WXYZ [w, x, y, z]")
        elif w3_mag > 0.8:
            print(f"        🎯 Likely format: XYZW [x, y, z, w]")
        else:
            print(f"        ⚠️  Cannot determine format clearly")
        
        # Show sample quaternions
        print(f"        Sample quaternions:")
        for i in range(min(3, sample_size)):
            q = sample_quats[i]
            print(f"          [{i}]: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")

    def _analyze_quaternion_format(self, quat_data: np.ndarray, key: str):
        """Analyze quaternion format and order."""
        if quat_data.shape[1] != 4:
            print(f"      ⚠️  Invalid quaternion dimensions: {quat_data.shape[1]} (expected 4)")
            return
        
        # Sample a few quaternions for analysis
        sample_size = min(10, quat_data.shape[0])
        sample_indices = np.linspace(0, quat_data.shape[0]-1, sample_size, dtype=int)
        sample_quats = quat_data[sample_indices]
        
        print(f"      🔍 Quaternion Analysis (sample of {sample_size} quaternions):")
        
        # Check quaternion normalization
        norms = np.linalg.norm(sample_quats, axis=1)
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        print(f"        └─ Norm: {norm_mean:.4f} ± {norm_std:.4f} (expected: 1.0)")
        
        if abs(norm_mean - 1.0) > 0.1:
            print(f"        ⚠️  Quaternions are not properly normalized!")
        
        # Analyze quaternion component ranges
        w_range = [np.min(sample_quats[:, 0]), np.max(sample_quats[:, 0])]
        x_range = [np.min(sample_quats[:, 1]), np.max(sample_quats[:, 1])]
        y_range = [np.min(sample_quats[:, 2]), np.max(sample_quats[:, 2])]
        z_range = [np.min(sample_quats[:, 3]), np.max(sample_quats[:, 3])]
        
        print(f"        └─ Component ranges:")
        print(f"           W: [{w_range[0]:.4f}, {w_range[1]:.4f}]")
        print(f"           X: [{x_range[0]:.4f}, {x_range[1]:.4f}]")
        print(f"           Y: [{y_range[0]:.4f}, {y_range[1]:.4f}]")
        print(f"           Z: [{z_range[0]:.4f}, {z_range[1]:.4f}]")
        
        # Determine likely quaternion order using a better heuristic
        # For small rotations, w should be close to 1 and xyz should be small
        # Check which component is closest to 1 on average
        w0_abs_mean = np.mean(np.abs(sample_quats[:, 0]))  # First component
        w3_abs_mean = np.mean(np.abs(sample_quats[:, 3]))  # Last component
        xyz_middle_mean = np.mean(np.abs(sample_quats[:, 1:3]))  # Middle components
        
        print(f"        └─ Quaternion Order Analysis:")
        print(f"           First component (index 0) avg magnitude: {w0_abs_mean:.4f}")
        print(f"           Last component (index 3) avg magnitude: {w3_abs_mean:.4f}")
        print(f"           Middle components (1:3) avg magnitude: {xyz_middle_mean:.4f}")
        
        # Better heuristic: which end component is closer to 1?
        # For WXYZ: w (index 0) should be ~1, xyz (1:3) should be small
        # For XYZW: w (index 3) should be ~1, xyz (0:2) should be small
        if w0_abs_mean > w3_abs_mean and w0_abs_mean > xyz_middle_mean:
            print(f"           🎯 Likely format: WXYZ (scalar-first)")
            print(f"           📝 Format: [w, x, y, z] where w is scalar component")
        elif w3_abs_mean > w0_abs_mean and w3_abs_mean > xyz_middle_mean:
            print(f"           🎯 Likely format: XYZW (scalar-last)")
            print(f"           📝 Format: [x, y, z, w] where w is scalar component")
        else:
            print(f"           ⚠️  Cannot determine format clearly - ambiguous quaternion data")
        
        # Check for common quaternion patterns
        # WXYZ: w should typically be larger magnitude for small rotations
        # XYZW: last component should typically be larger magnitude for small rotations
        
        # Additional analysis: check if quaternions represent valid rotations
        try:
            # Convert to rotation matrices to validate
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_quat(sample_quats)
            rotation_matrices = rotation.as_matrix()
            
            # Check if rotation matrices are orthogonal (det should be ~1)
            determinants = np.linalg.det(rotation_matrices)
            det_mean = np.mean(determinants)
            det_std = np.std(determinants)
            
            print(f"        └─ Rotation Matrix Validation:")
            print(f"           Determinant: {det_mean:.4f} ± {det_std:.4f} (expected: 1.0)")
            
            if abs(det_mean - 1.0) < 0.1:
                print(f"           ✅ Valid rotation matrices")
            else:
                print(f"           ⚠️  Invalid rotation matrices (determinant far from 1)")
                
        except ImportError:
            print(f"        └─ ⚠️  Cannot validate rotation matrices (scipy not available)")
        except Exception as e:
            print(f"        └─ ⚠️  Error validating rotation matrices: {e}")
        
        # Check for identity quaternions
        identity_count = np.sum(np.allclose(sample_quats, [1, 0, 0, 0], atol=0.01))
        if identity_count > 0:
            print(f"        └─ Identity quaternions found: {identity_count}/{sample_size}")
        
        # Check for zero quaternions
        zero_count = np.sum(np.allclose(sample_quats, [0, 0, 0, 0], atol=0.01))
        if zero_count > 0:
            print(f"        └─ ⚠️  Zero quaternions found: {zero_count}/{sample_size}")
    
    def _generate_summary_report(self, analyses: list):
        """Generate simple summary report."""
        print(f"\n" + "="*60)
        print(f"📊 AMASS DATA SUMMARY")
        print(f"="*60)
        
        if not analyses:
            print("❌ No analyses to summarize")
            return
        
        # Collect all unique keys
        all_keys = set()
        for analysis in analyses:
            all_keys.update(analysis['keys'])
        
        print(f"\n🔑 All Keys Found: {sorted(all_keys)}")
        
        # Show dimensions for each key
        print(f"\n📋 Key Dimensions:")
        for key in sorted(all_keys):
            dimensions = []
            for analysis in analyses:
                if key in analysis['dimensions']:
                    dimensions.append(analysis['dimensions'][key])
            
            # Show unique dimensions
            unique_dims = list(set(str(d) for d in dimensions))
            if len(unique_dims) == 1:
                print(f"  ✅ {key}: {unique_dims[0]}")
            else:
                print(f"  ⚠️  {key}: {unique_dims}")
        
        # Frame count analysis
        frame_counts = [analysis['frame_count'] for analysis in analyses]
        print(f"\n⏱️  Frame Count Analysis:")
        print(f"  Average frames: {np.mean(frame_counts):.1f}")
        print(f"  Frame range: {min(frame_counts)} - {max(frame_counts)}")
        
        print(f"\n🎉 Analysis complete! Analyzed {len(analyses)} motions.")


def main():
    parser = argparse.ArgumentParser(description='Check dimensions of AMASS retargeted data for G1 humanoid')
    parser.add_argument('--pkl_file', type=str, required=True, help='Path to AMASS PKL file')
    parser.add_argument('--motion_ids', type=str, default=None, help='Comma-separated motion IDs to analyze (e.g., "0,1,2" or "0-5")')
    parser.add_argument('--max_motions', type=int, default=10, help='Maximum number of motions to analyze')
    
    args = parser.parse_args()
    
    # Parse motion IDs
    motion_ids = None
    if args.motion_ids:
        motion_ids = []
        for x in args.motion_ids.split(','):
            x = x.strip()
            if '-' in x:
                start, end = map(int, x.split('-'))
                motion_ids.extend(range(start, end + 1))
            else:
                motion_ids.append(int(x))
    
    # Initialize checker
    checker = AMASSDimensionChecker(args.pkl_file)
    
    # Analyze dimensions
    results = checker.load_and_analyze(motion_ids, args.max_motions)
    
    if results:
        print(f"\n🎉 Dimension analysis complete!")
        print(f"Analyzed {len(results)} motions")
    else:
        print(f"\n❌ Dimension analysis failed!")


if __name__ == "__main__":
    main()


'''
# Check dimensions of specific motions
python scripts/check_amass_dimensions_s0.py \
    --pkl_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "0-5" \
    --max_motions 10

# Check dimensions of all motions (first 10)
python scripts/check_amass_dimensions_s0.py \
    --pkl_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --max_motions 10

# Check dimensions of single motion
python scripts/check_amass_dimensions_s0.py \
    --pkl_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "1"
'''
