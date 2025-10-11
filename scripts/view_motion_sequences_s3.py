#!/usr/bin/env python3
"""
View motion block sequences for specific motion IDs.
Shows the exact sequence of motion blocks that represent each motion.
"""

import pandas as pd
import argparse
from pathlib import Path


def view_motion_sequences(csv_file: str, motion_ids: list = None, max_sequences: int = 10):
    """View motion block sequences for specified motion IDs."""
    
    # Load the analysis results
    df = pd.read_csv(csv_file)
    
    print(f"📊 Motion Block Sequences Analysis")
    print(f"=================================")
    print(f"Total motions in dataset: {len(df)}")
    print()
    
    # If no specific motion IDs provided, show first few
    if motion_ids is None:
        motion_ids = df['motion_id'].head(max_sequences).tolist()
    
    # Filter for requested motion IDs
    available_ids = df['motion_id'].tolist()
    requested_ids = [mid for mid in motion_ids if mid in available_ids]
    missing_ids = [mid for mid in motion_ids if mid not in available_ids]
    
    if missing_ids:
        print(f"⚠️  Motion IDs not found: {missing_ids}")
        print()
    
    if not requested_ids:
        print("❌ No valid motion IDs found!")
        return
    
    print(f"🎯 Showing sequences for {len(requested_ids)} motions:")
    print()
    
    # Display sequences for each motion
    for motion_id in requested_ids:
        motion_data = df[df['motion_id'] == motion_id].iloc[0]
        
        print(f"Motion ID: {motion_id}")
        print(f"Motion Key: {motion_data['motion_key']}")
        print(f"Duration: {motion_data['duration']:.2f}s")
        print(f"Total Blocks: {motion_data['total_blocks']}")
        print(f"Unique Blocks: {motion_data['unique_blocks']}")
        print(f"Diversity: {motion_data['block_diversity']:.3f}")
        print(f"Most Common Block: {motion_data['most_common_block']} (appears {motion_data['most_common_block_count']} times)")
        
        # Parse and display the sequence
        sequence_str = motion_data['codebook_sequence']
        sequence = [int(x.strip()) for x in sequence_str.split(',')]
        
        print(f"Motion Block Sequence ({len(sequence)} blocks):")
        
        # Display sequence in chunks for readability
        chunk_size = 20
        for i in range(0, len(sequence), chunk_size):
            chunk = sequence[i:i+chunk_size]
            chunk_str = ', '.join(f"{block:3d}" for block in chunk)
            print(f"  Blocks {i:3d}-{i+len(chunk)-1:3d}: {chunk_str}")
        
        print()
        print("-" * 80)
        print()




def main():
    parser = argparse.ArgumentParser(description='View motion block sequences')
    parser.add_argument('--csv_file', type=str, default='./outputs/analysis/individual_motion_analysis.csv', 
                       help='Path to the analysis CSV file')
    parser.add_argument('--motion_ids', type=str, default=None, 
                       help='Comma-separated motion IDs to view (e.g., "0,1,2" or "0-5")')
    parser.add_argument('--max_sequences', type=int, default=10, 
                       help='Maximum number of sequences to show if no IDs specified')
    parser.add_argument('--find_patterns', action='store_true', 
                       help='Find common motion block patterns')
    parser.add_argument('--pattern_length', type=int, default=3, 
                       help='Minimum length for pattern finding')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare sequences between specified motions')
    
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
    
    # Check if CSV file exists
    if not Path(args.csv_file).exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        print("Run the motion block analysis first!")
        return
    
    view_motion_sequences(args.csv_file, motion_ids, args.max_sequences)


if __name__ == "__main__":
    main()

'''
python scripts/view_motion_sequences.py \
    --csv_file ./outputs/analysis/individual_motion_analysis.csv \
    --motion_ids "0-300"
'''