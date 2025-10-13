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
            'ranges': {},
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
                
                # Calculate ranges for numeric data
                if np.issubdtype(dtype, np.number) and data_array.size > 0:
                    if data_array.ndim == 1:
                        analysis['ranges'][key] = {
                            'min': float(np.min(data_array)),
                            'max': float(np.max(data_array)),
                            'mean': float(np.mean(data_array)),
                            'std': float(np.std(data_array))
                        }
                    elif data_array.ndim == 2:
                        analysis['ranges'][key] = {
                            'shape': shape,
                            'min_per_dim': [float(np.min(data_array[:, i])) for i in range(min(3, shape[1]))],
                            'max_per_dim': [float(np.max(data_array[:, i])) for i in range(min(3, shape[1]))],
                            'mean_per_dim': [float(np.mean(data_array[:, i])) for i in range(min(3, shape[1]))],
                            'std_per_dim': [float(np.std(data_array[:, i])) for i in range(min(3, shape[1]))]
                        }
                    else:
                        analysis['ranges'][key] = {
                            'shape': shape,
                            'min': float(np.min(data_array)),
                            'max': float(np.max(data_array))
                        }
                else:
                    analysis['ranges'][key] = {'type': 'non_numeric', 'sample': str(data)[:100]}
                
                # Determine frame count (use first numeric array with time dimension)
                if analysis['frame_count'] == 0 and data_array.ndim >= 1 and np.issubdtype(dtype, np.number):
                    analysis['frame_count'] = data_array.shape[0]
                
                print(f"    {key}: {shape} ({dtype})")
                
                # Print detailed info for key motion components
                if key in ['root_trans_offset', 'root_rot', 'dof']:
                    if data_array.ndim == 2:
                        print(f"      └─ {key} range: [{np.min(data_array):.4f}, {np.max(data_array):.4f}]")
                        if key == 'root_trans_offset':
                            print(f"      └─ Position range (m): X[{np.min(data_array[:, 0]):.3f}, {np.max(data_array[:, 0]):.3f}], "
                                  f"Y[{np.min(data_array[:, 1]):.3f}, {np.max(data_array[:, 1]):.3f}], "
                                  f"Z[{np.min(data_array[:, 2]):.3f}, {np.max(data_array[:, 2]):.3f}]")
                        elif key == 'root_rot':
                            print(f"      └─ Quaternion range: [{np.min(data_array):.4f}, {np.max(data_array):.4f}]")
                        elif key == 'dof':
                            print(f"      └─ DOF range (rad): [{np.min(data_array):.4f}, {np.max(data_array):.4f}]")
                
            except Exception as e:
                print(f"    {key}: Error analyzing - {e}")
                analysis['dimensions'][key] = f"Error: {e}"
                analysis['data_types'][key] = str(type(data))
                analysis['ranges'][key] = {'error': str(e)}
        
        print(f"  📏 Total frames: {analysis['frame_count']}")
        
        return analysis
    
    def _generate_summary_report(self, analyses: list):
        """Generate comprehensive summary report."""
        print(f"\n" + "="*80)
        print(f"📊 AMASS DATA DIMENSION SUMMARY REPORT")
        print(f"="*80)
        
        if not analyses:
            print("❌ No analyses to summarize")
            return
        
        # Collect all unique keys
        all_keys = set()
        for analysis in analyses:
            all_keys.update(analysis['keys'])
        
        print(f"\n🔑 All Keys Found: {sorted(all_keys)}")
        
        # Analyze consistency across motions
        key_consistency = {}
        for key in all_keys:
            dimensions = []
            data_types = []
            for analysis in analyses:
                if key in analysis['dimensions']:
                    dimensions.append(analysis['dimensions'][key])
                    data_types.append(analysis['data_types'][key])
            
            key_consistency[key] = {
                'dimensions': dimensions,
                'data_types': data_types,
                'consistent_dimensions': len(set(str(d) for d in dimensions)) == 1,
                'consistent_types': len(set(data_types)) == 1
            }
        
        print(f"\n📋 Key Consistency Analysis:")
        for key, consistency in key_consistency.items():
            status = "✅" if consistency['consistent_dimensions'] and consistency['consistent_types'] else "⚠️ "
            print(f"  {status} {key}:")
            print(f"    Dimensions: {consistency['dimensions']}")
            print(f"    Types: {consistency['data_types']}")
            if not consistency['consistent_dimensions']:
                print(f"    ⚠️  Inconsistent dimensions across motions!")
            if not consistency['consistent_types']:
                print(f"    ⚠️  Inconsistent data types across motions!")
        
        # Frame count analysis
        frame_counts = [analysis['frame_count'] for analysis in analyses]
        print(f"\n⏱️  Frame Count Analysis:")
        print(f"  Average frames: {np.mean(frame_counts):.1f}")
        print(f"  Frame range: {min(frame_counts)} - {max(frame_counts)}")
        print(f"  Frame counts: {frame_counts}")
        
        # G1 Humanoid specific analysis
        self._analyze_g1_specifics(analyses, key_consistency)
        
        # Generate CSV report
        self._save_csv_report(analyses)
    
    def _analyze_g1_specifics(self, analyses: list, key_consistency: dict):
        """Analyze G1 humanoid specific aspects."""
        print(f"\n🤖 G1 Humanoid Specific Analysis:")
        
        # Check for expected G1 components
        g1_expected_keys = ['root_trans_offset', 'root_rot', 'dof']
        g1_optional_keys = ['pose_aa', 'smpl_joints', 'contact_mask', 'fps']
        
        print(f"  Required keys for G1:")
        for key in g1_expected_keys:
            if key in key_consistency:
                status = "✅" if key_consistency[key]['consistent_dimensions'] else "⚠️ "
                dims = key_consistency[key]['dimensions'][0] if key_consistency[key]['dimensions'] else "Unknown"
                print(f"    {status} {key}: {dims}")
            else:
                print(f"    ❌ {key}: Missing!")
        
        print(f"  Optional keys:")
        for key in g1_optional_keys:
            if key in key_consistency:
                status = "✅" if key_consistency[key]['consistent_dimensions'] else "⚠️ "
                dims = key_consistency[key]['dimensions'][0] if key_consistency[key]['dimensions'] else "Unknown"
                print(f"    {status} {key}: {dims}")
            else:
                print(f"    ⚪ {key}: Not present")
        
        # Analyze DOF dimensions (should be 23 for G1)
        if 'dof' in key_consistency and key_consistency['dof']['dimensions']:
            dof_dims = key_consistency['dof']['dimensions'][0]
            if len(dof_dims) == 2 and dof_dims[1] == 23:
                print(f"  ✅ DOF dimension correct for G1 (23 DOF)")
            else:
                print(f"  ⚠️  DOF dimension unusual for G1: {dof_dims} (expected: [frames, 23])")
        
        # Analyze root position dimensions (should be [frames, 3])
        if 'root_trans_offset' in key_consistency and key_consistency['root_trans_offset']['dimensions']:
            root_dims = key_consistency['root_trans_offset']['dimensions'][0]
            if len(root_dims) == 2 and root_dims[1] == 3:
                print(f"  ✅ Root position dimension correct: {root_dims}")
            else:
                print(f"  ⚠️  Root position dimension unusual: {root_dims} (expected: [frames, 3])")
        
        # Analyze root rotation dimensions (should be [frames, 4] for quaternions)
        if 'root_rot' in key_consistency and key_consistency['root_rot']['dimensions']:
            rot_dims = key_consistency['root_rot']['dimensions'][0]
            if len(rot_dims) == 2 and rot_dims[1] == 4:
                print(f"  ✅ Root rotation dimension correct (quaternions): {rot_dims}")
            else:
                print(f"  ⚠️  Root rotation dimension unusual: {rot_dims} (expected: [frames, 4])")
    
    def _save_csv_report(self, analyses: list):
        """Save detailed CSV report."""
        # Create detailed report
        detailed_rows = []
        for analysis in analyses:
            for key, dims in analysis['dimensions'].items():
                detailed_rows.append({
                    'motion_id': analysis['motion_id'],
                    'motion_key': analysis['motion_key'],
                    'data_key': key,
                    'dimensions': str(dims),
                    'data_type': analysis['data_types'].get(key, 'unknown'),
                    'frame_count': analysis['frame_count']
                })
        
        # Save detailed report
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_file = Path(self.pkl_file_path).parent / "amass_dimensions_detailed.csv"
        detailed_df.to_csv(detailed_file, index=False)
        
        # Create summary report
        summary_rows = []
        all_keys = set()
        for analysis in analyses:
            all_keys.update(analysis['keys'])
        
        for key in all_keys:
            dims_list = []
            types_list = []
            for analysis in analyses:
                if key in analysis['dimensions']:
                    dims_list.append(str(analysis['dimensions'][key]))
                    types_list.append(analysis['data_types'][key])
            
            summary_rows.append({
                'data_key': key,
                'dimensions': '; '.join(set(dims_list)),
                'data_types': '; '.join(set(types_list)),
                'frequency': len(dims_list),
                'consistent': len(set(dims_list)) == 1
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = Path(self.pkl_file_path).parent / "amass_dimensions_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n📄 Reports saved:")
        print(f"  - Detailed: {detailed_file}")
        print(f"  - Summary: {summary_file}")


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
python scripts/check_amass_dimensions.py \
    --pkl_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "0-5" \
    --max_motions 10

# Check dimensions of all motions (first 10)
python scripts/check_amass_dimensions.py \
    --pkl_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --max_motions 10

# Check dimensions of single motion
python scripts/check_amass_dimensions.py \
    --pkl_file /home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl \
    --motion_ids "1"
'''
