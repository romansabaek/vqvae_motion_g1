#!/usr/bin/env python3
"""
Analyze what types of motions (walking, running, turning, etc.) are in each cluster.
This helps understand if the VQVAE learned meaningful motion categories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import argparse
from pathlib import Path
import re


def extract_motion_type_from_key(motion_key: str) -> str:
    """Extract motion type from motion key using keyword matching."""
    
    motion_key_lower = motion_key.lower()
    
    # Define motion type patterns
    motion_patterns = {
        'walking': ['walk', 'stride', 'step'],
        'running': ['run', 'jog', 'sprint'],
        'turning': ['turn', 'rotate', 'spin'],
        'jumping': ['jump', 'hop', 'leap'],
        'sitting': ['sit', 'chair', 'seat'],
        'standing': ['stand', 'rise', 'up'],
        'bending': ['bend', 'lean', 'forward', 'backward'],
        'reaching': ['reach', 'grab', 'pick', 'put'],
        'dancing': ['dance', 'rhythm', 'beat'],
        'sports': ['throw', 'catch', 'kick', 'ball', 'tennis', 'golf'],
        'climbing': ['climb', 'stairs', 'ladder'],
        'falling': ['fall', 'drop', 'down'],
        'idle': ['idle', 'rest', 'pause', 'still'],
        'transition': ['transition', 'change', 'switch', 'to', 'from']
    }
    
    # Check for motion patterns
    for motion_type, keywords in motion_patterns.items():
        for keyword in keywords:
            if keyword in motion_key_lower:
                return motion_type
    
    # If no pattern matches, return 'unknown'
    return 'unknown'


def analyze_cluster_motion_types(csv_file: str, cluster_file: str = None):
    """Analyze what types of motions are in each cluster."""
    
    print("🔍 Analyzing motion types in each cluster...")
    
    # Load motion analysis data
    df_motions = pd.read_csv(csv_file)
    
    # Load cluster data if provided
    if cluster_file and Path(cluster_file).exists():
        df_clusters = pd.read_csv(cluster_file)
        # Merge with motion data
        df_combined = df_motions.merge(df_clusters[['motion_id', 'cluster']], on='motion_id')
        print(f"✅ Loaded cluster data from: {cluster_file}")
    else:
        # Create dummy clusters (all in cluster 0) for testing
        df_combined = df_motions.copy()
        df_combined['cluster'] = 0
        print("⚠️  No cluster file provided, using single cluster")
    
    # Extract motion types from motion keys
    print("🏷️  Extracting motion types from motion keys...")
    df_combined['motion_type'] = df_combined['motion_key'].apply(extract_motion_type_from_key)
    
    # Analyze motion types per cluster
    print("\n📊 Motion Type Analysis by Cluster:")
    print("=" * 50)
    
    cluster_analysis = {}
    
    for cluster_id in sorted(df_combined['cluster'].unique()):
        cluster_data = df_combined[df_combined['cluster'] == cluster_id]
        
        print(f"\n🎯 Cluster {cluster_id} ({len(cluster_data)} motions):")
        print("-" * 30)
        
        # Count motion types in this cluster
        motion_type_counts = cluster_data['motion_type'].value_counts()
        
        print("Motion Types:")
        for motion_type, count in motion_type_counts.items():
            percentage = (count / len(cluster_data)) * 100
            print(f"  {motion_type}: {count} motions ({percentage:.1f}%)")
        
        # Show some example motion keys for each type
        print("\nExample Motion Keys:")
        for motion_type in motion_type_counts.index[:3]:  # Top 3 types
            examples = cluster_data[cluster_data['motion_type'] == motion_type]['motion_key'].head(3)
            print(f"  {motion_type}:")
            for example in examples:
                print(f"    - {example}")
        
        # Store analysis for visualization
        cluster_analysis[cluster_id] = {
            'motion_type_counts': motion_type_counts,
            'total_motions': len(cluster_data),
            'avg_duration': cluster_data['duration'].mean(),
            'avg_diversity': cluster_data['block_diversity'].mean()
        }
    
    return df_combined, cluster_analysis


def create_motion_type_visualizations(df_combined, cluster_analysis, output_dir: Path):
    """Create visualizations showing motion types in each cluster."""
    
    print("📊 Creating motion type visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Motion type distribution across clusters
    ax1 = axes[0, 0]
    motion_type_cluster_matrix = df_combined.groupby(['cluster', 'motion_type']).size().unstack(fill_value=0)
    
    # Normalize by cluster size
    motion_type_cluster_normalized = motion_type_cluster_matrix.div(motion_type_cluster_matrix.sum(axis=1), axis=0)
    
    sns.heatmap(motion_type_cluster_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Motion Type Distribution Across Clusters')
    ax1.set_xlabel('Motion Type')
    ax1.set_ylabel('Cluster')
    
    # 2. Cluster characteristics
    ax2 = axes[0, 1]
    cluster_chars = []
    for cluster_id, analysis in cluster_analysis.items():
        cluster_chars.append({
            'cluster': cluster_id,
            'total_motions': analysis['total_motions'],
            'avg_duration': analysis['avg_duration'],
            'avg_diversity': analysis['avg_diversity']
        })
    
    df_chars = pd.DataFrame(cluster_chars)
    
    scatter = ax2.scatter(df_chars['avg_duration'], df_chars['avg_diversity'], 
                         c=df_chars['cluster'], cmap='tab10', s=df_chars['total_motions']*3, 
                         alpha=0.7, edgecolors='black')
    
    for i, row in df_chars.iterrows():
        ax2.annotate(f'C{row["cluster"]}', (row['avg_duration'], row['avg_diversity']),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax2.set_xlabel('Average Duration (seconds)')
    ax2.set_ylabel('Average Block Diversity')
    ax2.set_title('Cluster Characteristics\n(Size = Number of Motions)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Motion type frequency
    ax3 = axes[1, 0]
    motion_type_counts = df_combined['motion_type'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(motion_type_counts)))
    
    wedges, texts, autotexts = ax3.pie(motion_type_counts.values, 
                                      labels=motion_type_counts.index,
                                      autopct='%1.1f%%', 
                                      colors=colors,
                                      startangle=90)
    ax3.set_title('Overall Motion Type Distribution')
    
    # 4. Duration vs Diversity by Motion Type
    ax4 = axes[1, 1]
    motion_types = df_combined['motion_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(motion_types)))
    
    for i, motion_type in enumerate(motion_types):
        type_data = df_combined[df_combined['motion_type'] == motion_type]
        ax4.scatter(type_data['duration'], type_data['block_diversity'], 
                   c=[colors[i]], label=motion_type, alpha=0.6, s=50)
    
    ax4.set_xlabel('Duration (seconds)')
    ax4.set_ylabel('Block Diversity')
    ax4.set_title('Duration vs Diversity by Motion Type')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "motion_type_cluster_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Motion type visualizations saved to: {output_dir / 'motion_type_cluster_analysis.png'}")


def create_detailed_cluster_report(df_combined, cluster_analysis, output_dir: Path):
    """Create a detailed report of motion types in each cluster."""
    
    print("📄 Creating detailed cluster report...")
    
    report = "# Motion Type Cluster Analysis Report\n\n"
    report += "This report analyzes what types of motions (walking, running, turning, etc.) are grouped together in each cluster.\n\n"
    
    # Overall statistics
    total_motions = len(df_combined)
    unique_motion_types = df_combined['motion_type'].nunique()
    
    report += f"## Overall Statistics\n"
    report += f"- **Total Motions**: {total_motions}\n"
    report += f"- **Unique Motion Types**: {unique_motion_types}\n"
    report += f"- **Number of Clusters**: {len(cluster_analysis)}\n\n"
    
    # Motion type distribution
    motion_type_counts = df_combined['motion_type'].value_counts()
    report += f"## Motion Type Distribution\n"
    for motion_type, count in motion_type_counts.items():
        percentage = (count / total_motions) * 100
        report += f"- **{motion_type}**: {count} motions ({percentage:.1f}%)\n"
    report += "\n"
    
    # Detailed cluster analysis
    for cluster_id in sorted(cluster_analysis.keys()):
        analysis = cluster_analysis[cluster_id]
        cluster_data = df_combined[df_combined['cluster'] == cluster_id]
        
        report += f"## Cluster {cluster_id}\n"
        report += f"- **Total Motions**: {analysis['total_motions']}\n"
        report += f"- **Average Duration**: {analysis['avg_duration']:.2f} seconds\n"
        report += f"- **Average Block Diversity**: {analysis['avg_diversity']:.3f}\n\n"
        
        # Motion types in this cluster
        motion_type_counts = analysis['motion_type_counts']
        report += f"### Motion Types in Cluster {cluster_id}\n"
        
        for motion_type, count in motion_type_counts.items():
            percentage = (count / analysis['total_motions']) * 100
            report += f"- **{motion_type}**: {count} motions ({percentage:.1f}%)\n"
        
        report += "\n### Example Motions\n"
        
        # Show examples for each motion type
        for motion_type in motion_type_counts.index[:3]:  # Top 3 types
            examples = cluster_data[cluster_data['motion_type'] == motion_type]['motion_key'].head(5)
            report += f"**{motion_type}:**\n"
            for example in examples:
                # Clean up the motion key for readability
                clean_key = example.replace('_poses', '').replace('0-ACCAD_ACCAD_', '')
                report += f"  - {clean_key}\n"
            report += "\n"
        
        report += "---\n\n"
    
    # Cluster interpretation
    report += f"## Cluster Interpretation\n\n"
    report += f"Based on the motion type analysis, here's what each cluster likely represents:\n\n"
    
    for cluster_id in sorted(cluster_analysis.keys()):
        analysis = cluster_analysis[cluster_id]
        dominant_type = analysis['motion_type_counts'].index[0]
        dominant_percentage = (analysis['motion_type_counts'].iloc[0] / analysis['total_motions']) * 100
        
        report += f"### Cluster {cluster_id}\n"
        report += f"- **Dominant Type**: {dominant_type} ({dominant_percentage:.1f}%)\n"
        report += f"- **Characteristics**: "
        
        if analysis['avg_duration'] < 3:
            report += "Short duration, "
        elif analysis['avg_duration'] > 8:
            report += "Long duration, "
        else:
            report += "Medium duration, "
            
        if analysis['avg_diversity'] < 0.4:
            report += "low complexity (repetitive motions)\n"
        elif analysis['avg_diversity'] > 0.7:
            report += "high complexity (varied motions)\n"
        else:
            report += "medium complexity\n"
        
        report += f"- **Likely represents**: "
        
        # Interpret based on motion types and characteristics
        if dominant_type == 'walking':
            if analysis['avg_diversity'] < 0.5:
                report += "Simple walking patterns (straight walking)\n"
            else:
                report += "Complex walking patterns (walking with variations)\n"
        elif dominant_type == 'turning':
            report += "Turning and rotation movements\n"
        elif dominant_type == 'transition':
            report += "Transition movements between different poses\n"
        elif dominant_type == 'running':
            report += "Running and fast locomotion\n"
        elif dominant_type == 'jumping':
            report += "Jumping and hopping movements\n"
        else:
            report += f"{dominant_type.capitalize()} movements\n"
        
        report += "\n"
    
    # Save report
    with open(output_dir / "motion_type_cluster_report.md", 'w') as f:
        f.write(report)
    
    print(f"📄 Detailed report saved to: {output_dir / 'motion_type_cluster_report.md'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze motion types in clusters')
    parser.add_argument('--csv_file', type=str, default='./outputs/analysis/individual_motion_analysis.csv',
                       help='Path to the motion analysis CSV file')
    parser.add_argument('--cluster_file', type=str, default='./outputs/clustering/motion_clusters.csv',
                       help='Path to the cluster assignments CSV file')
    parser.add_argument('--output_dir', type=str, default='./outputs/motion_type_analysis',
                       help='Output directory for motion type analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if CSV file exists
    if not Path(args.csv_file).exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        print("Run the motion block analysis first!")
        return
    
    print(f"🎯 Motion Type Cluster Analysis")
    print(f"==============================")
    print(f"Motion CSV: {args.csv_file}")
    print(f"Cluster CSV: {args.cluster_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Analyze motion types in clusters
    df_combined, cluster_analysis = analyze_cluster_motion_types(args.csv_file, args.cluster_file)
    
    # Create visualizations
    create_motion_type_visualizations(df_combined, cluster_analysis, output_path)
    
    # Create detailed report
    create_detailed_cluster_report(df_combined, cluster_analysis, output_path)
    
    print(f"\n🎉 Motion type analysis complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Analyzed {len(df_combined)} motions across {len(cluster_analysis)} clusters")


if __name__ == "__main__":
    main()


'''
python scripts/analyze_motion_cluster_types.py \
    --csv_file ./outputs/analysis/individual_motion_analysis.csv \
    --cluster_file ./outputs/clustering/motion_clusters.csv \
    --output_dir ./outputs/motion_type_analysis
'''