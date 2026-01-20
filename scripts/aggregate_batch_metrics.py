#!/usr/bin/env python3
"""
Aggregate metrics from batch evaluation JSON files and calculate averages.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def load_metrics_from_json(json_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {json_path}: {e}")
        return None


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics by calculating mean, std, min, max for numeric values."""
    if not metrics_list:
        return {}
    
    # Collect all keys
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = []
        for m in metrics_list:
            if key in m:
                val = m[key]
                # Handle different types
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, list):
                    # For lists, we'll compute element-wise statistics
                    values.append(val)
                elif isinstance(val, dict):
                    # Skip nested dicts for now
                    continue
        
        if not values:
            continue
        
        # Check if all values are scalars
        if all(isinstance(v, (int, float)) for v in values):
            values_array = np.array(values)
            aggregated[key] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "count": len(values)
            }
        # Check if all values are lists of the same length
        elif all(isinstance(v, list) and len(v) == len(values[0]) for v in values):
            # Element-wise statistics for lists
            values_array = np.array(values)
            aggregated[key] = {
                "mean": np.mean(values_array, axis=0).tolist(),
                "std": np.std(values_array, axis=0).tolist(),
                "min": np.min(values_array, axis=0).tolist(),
                "max": np.max(values_array, axis=0).tolist(),
                "count": len(values)
            }
    
    return aggregated


def main():
    if len(sys.argv) < 4:
        print("Usage: python aggregate_batch_metrics.py <output_dir_base> <start_id> <end_id>")
        sys.exit(1)
    
    output_dir_base = Path(sys.argv[1])
    start_id = int(sys.argv[2])
    end_id = int(sys.argv[3])
    
    print(f"Aggregating metrics from motion IDs {start_id} to {end_id}...")
    
    metrics_list = []
    missing_count = 0
    
    for motion_id in range(start_id, end_id + 1):
        metrics_path = output_dir_base / f"motion_{motion_id}" / f"motion_{motion_id}_metrics.json"
        if metrics_path.exists():
            metrics = load_metrics_from_json(metrics_path)
            if metrics:
                metrics_list.append(metrics)
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    if not metrics_list:
        print(f"Error: No metrics files found in {output_dir_base}")
        sys.exit(1)
    
    print(f"Found {len(metrics_list)} metrics files ({missing_count} missing)")
    
    # Aggregate metrics
    aggregated = aggregate_metrics(metrics_list)
    
    # Add summary info
    aggregated["_summary"] = {
        "total_motions": len(metrics_list),
        "missing_motions": missing_count,
        "motion_id_range": [start_id, end_id]
    }
    
    # Save aggregated metrics
    output_path = output_dir_base / "aggregated_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated metrics saved to: {output_path}")
    print("\nAverage Performance Summary:")
    print("=" * 80)
    
    # Print key metrics in a readable format
    key_metrics = [
        "overall_rmse", "root_deltas_rmse", "dof_pos_rmse", "dof_vel_rmse",
        "root_dx_rmse", "root_dy_rmse", "root_dz_rmse", "root_dyaw_rmse",
        "codebook_acc", "codebook_top3_acc", "policy_acc"
    ]
    
    for key in key_metrics:
        if key in aggregated:
            stats = aggregated[key]
            if isinstance(stats, dict) and "mean" in stats:
                print(f"{key:25s}: {stats['mean']:.6f} ± {stats['std']:.6f} (min={stats['min']:.6f}, max={stats['max']:.6f}, n={stats['count']})")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

