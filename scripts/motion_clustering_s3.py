#!/usr/bin/env python3
"""
Simple motion block clustering - just basic clustering with scatter plots and grouping.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path


def extract_motion_block_features(df):
    """Extract features based on motion block content and patterns."""
    
    print("🔧 Extracting motion block content features...")
    
    # Get all unique block IDs
    all_blocks = set()
    for _, row in df.iterrows():
        sequence = [int(x.strip()) for x in row['codebook_sequence'].split(',')]
        all_blocks.update(sequence)
    
    max_block_id = max(all_blocks)
    print(f"📊 Found {len(all_blocks)} unique blocks (max ID: {max_block_id})")
    
    # Analyze which blocks are most commonly used across all motions
    global_block_counts = {}
    for _, row in df.iterrows():
        sequence = [int(x.strip()) for x in row['codebook_sequence'].split(',')]
        for block_id in sequence:
            global_block_counts[block_id] = global_block_counts.get(block_id, 0) + 1
    
    # Get top blocks that appear frequently across motions
    top_blocks = sorted(global_block_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    top_block_ids = [block_id for block_id, _ in top_blocks]
    
    print(f"🎯 Using top {len(top_block_ids)} most frequent blocks for clustering")
    print(f"   Top blocks: {top_block_ids[:10]}...")
    
    # Create feature matrix focused on motion block content
    motion_features = []
    
    for _, row in df.iterrows():
        sequence = [int(x.strip()) for x in row['codebook_sequence'].split(',')]
        
        # 1. Block presence vector (which important blocks are used)
        block_presence = np.zeros(len(top_block_ids))
        for i, block_id in enumerate(top_block_ids):
            if block_id in sequence:
                block_presence[i] = 1.0
        
        # 2. Block frequency vector (how often each important block appears)
        block_frequency = np.zeros(len(top_block_ids))
        block_counts = {}
        for block_id in sequence:
            block_counts[block_id] = block_counts.get(block_id, 0) + 1
        
        for i, block_id in enumerate(top_block_ids):
            if block_id in block_counts:
                block_frequency[i] = block_counts[block_id] / len(sequence)
        
        # 3. Block combination patterns (which blocks appear together)
        block_combinations = np.zeros(len(top_block_ids))
        for block_id in sequence:
            if block_id in top_block_ids:
                idx = top_block_ids.index(block_id)
                block_combinations[idx] += 1
        
        # Normalize combinations
        if len(sequence) > 0:
            block_combinations = block_combinations / len(sequence)
        
        # 4. Motion block signature (unique pattern of this motion)
        motion_signature = np.concatenate([
            block_presence,      # Which blocks are used
            block_frequency,     # How often each block appears
            block_combinations   # Block combination patterns
        ])
        
        motion_features.append(motion_signature)
    
    motion_features = np.array(motion_features)
    print(f"✅ Created motion block feature matrix: {motion_features.shape}")
    print(f"   - Block presence features: {len(top_block_ids)}")
    print(f"   - Block frequency features: {len(top_block_ids)}")
    print(f"   - Block combination features: {len(top_block_ids)}")
    print(f"   - Total features: {motion_features.shape[1]}")
    
    return motion_features, top_block_ids


def extract_vqvae_features(df):
    """Extract VQVAE-based features from motion block sequences."""
    
    print("🔧 Extracting VQVAE-based features...")
    
    # Get all unique block IDs to create consistent feature vectors
    all_blocks = set()
    for _, row in df.iterrows():
        sequence = [int(x.strip()) for x in row['codebook_sequence'].split(',')]
        all_blocks.update(sequence)
    
    max_block_id = max(all_blocks)
    print(f"📊 Found {len(all_blocks)} unique blocks (max ID: {max_block_id})")
    
    # Create feature matrix based on block frequencies and patterns
    vqvae_features = []
    
    for _, row in df.iterrows():
        sequence = [int(x.strip()) for x in row['codebook_sequence'].split(',')]
        
        # 1. Block frequency vector (normalized)
        block_counts = {}
        for block_id in sequence:
            block_counts[block_id] = block_counts.get(block_id, 0) + 1
        
        frequency_vector = np.zeros(max_block_id + 1)
        for block_id, count in block_counts.items():
            frequency_vector[block_id] = count / len(sequence)  # Normalize by sequence length
        
        # 2. Block transition patterns (simplified)
        transitions = {}
        for i in range(len(sequence) - 1):
            transition = (sequence[i], sequence[i + 1])
            transitions[transition] = transitions.get(transition, 0) + 1
        
        # Create transition vector (top transitions only)
        top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:20]
        transition_vector = np.zeros(20)
        for i, (_, count) in enumerate(top_transitions):
            if i < 20:
                transition_vector[i] = count / len(sequence)
        
        # 3. Block pattern statistics
        pattern_stats = [
            len(set(sequence)) / len(sequence),  # Block diversity
            np.std(frequency_vector),           # Frequency variation
            max(frequency_vector),              # Max frequency
            len(sequence)                       # Sequence length
        ]
        
        # Combine all features
        combined_features = np.concatenate([
            frequency_vector,      # Block frequencies
            transition_vector,     # Top transitions
            pattern_stats          # Pattern statistics
        ])
        
        vqvae_features.append(combined_features)
    
    vqvae_features = np.array(vqvae_features)
    print(f"✅ Created VQVAE feature matrix: {vqvae_features.shape}")
    print(f"   - Block frequency features: {max_block_id + 1}")
    print(f"   - Transition features: 20")
    print(f"   - Pattern statistics: 4")
    
    return vqvae_features


def simple_clustering_analysis(csv_file: str, output_dir: str = "./outputs/simple_clustering", n_clusters: int = 5, feature_type: str = "motion_blocks"):
    """Simple clustering analysis with different feature types."""
    
    print(f"🎯 Motion Block Clustering ({feature_type.upper()} Features)")
    print(f"===============================================================================")
    print(f"Input CSV: {csv_file}")
    print(f"Output directory: {output_dir}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Feature type: {feature_type}")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load motion data
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} motions")
    
    # Create feature matrix based on feature type
    if feature_type == "motion_blocks":
        features, top_block_ids = extract_motion_block_features(df)
        feature_description = "Motion Block Content"
    elif feature_type == "vqvae":
        features = extract_vqvae_features(df)
        top_block_ids = None
        feature_description = "VQVAE-based"
    else:  # basic
        features = df[['duration', 'block_diversity', 'unique_blocks', 'total_blocks']].values
        top_block_ids = None
        feature_description = "Basic Statistics"
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    print(f"🎯 Performing K-means clustering with {n_clusters} clusters using {feature_description} features...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Print cluster summary with motion block analysis
    print(f"\n📊 Cluster Summary:")
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"Cluster {cluster_id}: {len(cluster_data)} motions")
        print(f"  - Average duration: {cluster_data['duration'].mean():.2f}s")
        print(f"  - Average diversity: {cluster_data['block_diversity'].mean():.3f}")
        
        # Analyze motion blocks in this cluster
        if feature_type == "motion_blocks" and top_block_ids:
            cluster_blocks = []
            for _, row in cluster_data.iterrows():
                sequence = [int(x.strip()) for x in row['codebook_sequence'].split(',')]
                cluster_blocks.extend(sequence)
            
            # Find most common blocks in this cluster
            from collections import Counter
            block_counts = Counter(cluster_blocks)
            top_cluster_blocks = block_counts.most_common(5)
            print(f"  - Top blocks: {[f'{block}({count})' for block, count in top_cluster_blocks]}")
        
        print(f"  - Motion IDs: {sorted(cluster_data['motion_id'].tolist())[:10]}{'...' if len(cluster_data) > 10 else ''}")
        print()
    
    # Create single clustering plot
    print("📊 Creating main clustering plot...")
    
    # Use PCA to reduce features to 2D for visualization
    if feature_type in ["motion_blocks", "vqvae"]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)'
        y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
        title = f'Motion Clustering ({feature_description}) - {n_clusters} Clusters'
    else:  # basic features
        features_2d = features_scaled[:, :2]  # Use first 2 basic features
        x_label = 'Duration (normalized)'
        y_label = 'Block Diversity (normalized)'
        title = f'Motion Clustering ({feature_description}) - {n_clusters} Clusters'
    
    # Create main clustering plot
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster with different colors and markers
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        cluster_mask = df['cluster'] == cluster_id
        cluster_data_2d = features_2d[cluster_mask]
        
        plt.scatter(cluster_data_2d[:, 0], cluster_data_2d[:, 1], 
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id} ({cluster_mask.sum()} motions)', 
                   alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Plot cluster centers if using high-dimensional features
    if feature_type in ["motion_blocks", "vqvae"]:
        cluster_centers_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], 
                   c='red', marker='x', s=200, linewidth=3, label='Cluster Centers')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Total Motions: {len(df)}\n'
    for cluster_id in range(n_clusters):
        cluster_size = (df['cluster'] == cluster_id).sum()
        stats_text += f'Cluster {cluster_id}: {cluster_size} motions\n'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / "motion_clustering_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster assignments
    df.to_csv(output_path / "simple_clusters.csv", index=False)
    
    print(f"✅ Motion clustering complete!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Created {n_clusters} clusters from {len(df)} motions")
    print(f"📈 Main clustering plot saved to: {output_path / 'motion_clustering_plot.png'}")
    print(f"📋 Cluster assignments saved to: {output_path / 'simple_clusters.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Motion block clustering with different feature types')
    parser.add_argument('--csv_file', type=str, default='./outputs/analysis/individual_motion_analysis.csv',
                       help='Path to the motion analysis CSV file')
    parser.add_argument('--output_dir', type=str, default='./outputs/motion_block_clustering',
                       help='Output directory for clustering results')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters')
    parser.add_argument('--feature_type', type=str, default='motion_blocks', 
                       choices=['motion_blocks', 'vqvae', 'basic'],
                       help='Feature type: motion_blocks (recommended), vqvae, or basic')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not Path(args.csv_file).exists():
        print(f"❌ CSV file not found: {args.csv_file}")
        print("Run the motion block analysis first!")
        return
    
    # Run clustering
    simple_clustering_analysis(args.csv_file, args.output_dir, args.n_clusters, args.feature_type)


if __name__ == "__main__":
    main()

'''
python scripts/motion_clustering.py \
    --csv_file ./outputs/analysis/individual_motion_analysis.csv \
    --output_dir ./outputs/clustering
'''