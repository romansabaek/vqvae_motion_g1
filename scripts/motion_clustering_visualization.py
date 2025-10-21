#!/usr/bin/env python3
"""
Motion Clustering Visualization Script
Groups similar motions based on encoder output features (latent space representations) using K-means clustering.
Creates a single t-SNE visualization showing motion clusters in 2D space.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import Counter
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def load_codebook_sequences(base_dir: str, max_motions: int = None) -> pd.DataFrame:
    """
    Load all codebook sequence CSV files and create a unified dataset.
    """
    print("🔧 Loading codebook sequences...")
    
    all_data = []
    csv_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.csv')])
    
    if max_motions:
        csv_files = csv_files[:max_motions]
    
    print(f"📊 Found {len(csv_files)} codebook sequence files")
    
    for i, csv_file in enumerate(csv_files):
        if i % 50 == 0:
            print(f"   Loading {i+1}/{len(csv_files)}: {csv_file}")
            
        csv_path = os.path.join(base_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            
            # Extract motion ID from filename
            motion_id = int(csv_file.split('_')[-1].split('.')[0])
            
            # Create motion summary
            motion_summary = {
                'motion_id': motion_id,
                'file_name': csv_file,
                'total_frames': len(df),
                'unique_codebooks': df['codebook_idx'].nunique(),
                'codebook_range_min': df['codebook_idx'].min(),
                'codebook_range_max': df['codebook_idx'].max(),
                'codebook_changes': df['codebook_changed'].sum() if 'codebook_changed' in df.columns else 0,
                'avg_reconstruction_error': df['total_reconstruction_error'].mean() if 'total_reconstruction_error' in df.columns else 0,
                'motion_description': df['original_key'].iloc[0] if 'original_key' in df.columns else f"Motion_{motion_id}",
                'codebook_sequence': ','.join(df['codebook_idx'].astype(str).tolist())
            }
            
            all_data.append(motion_summary)
            
        except Exception as e:
            print(f"   Warning: Could not load {csv_file}: {e}")
            continue
    
    motion_df = pd.DataFrame(all_data)
    print(f"✅ Loaded {len(motion_df)} motion sequences")
    
    return motion_df

def extract_encoder_features(motion_df: pd.DataFrame, base_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract encoder output features (latent space representations) from codebook sequences.
    """
    print("🔧 Extracting encoder output features...")
    
    features = []
    feature_names = []
    
    for _, row in motion_df.iterrows():
        csv_path = os.path.join(base_dir, row['file_name'])
        try:
            df = pd.read_csv(csv_path)
            
            # Extract reconstructed features (encoder outputs)
            reconstructed_cols = [col for col in df.columns if col.startswith('reconstructed_feat_')]
            reconstructed_features = df[reconstructed_cols].values  # Shape: (frames, features)
            
            # Compute motion-level features from encoder outputs
            # 1. Mean encoder output across all frames
            mean_encoder = np.mean(reconstructed_features, axis=0)
            
            # 2. Standard deviation of encoder outputs
            std_encoder = np.std(reconstructed_features, axis=0)
            
            # 3. Min and max values
            min_encoder = np.min(reconstructed_features, axis=0)
            max_encoder = np.max(reconstructed_features, axis=0)
            
            # 4. First and last frame features (motion boundaries)
            first_frame = reconstructed_features[0] if len(reconstructed_features) > 0 else np.zeros(len(reconstructed_cols))
            last_frame = reconstructed_features[-1] if len(reconstructed_features) > 0 else np.zeros(len(reconstructed_cols))
            
            # 5. Motion dynamics (frame-to-frame differences)
            if len(reconstructed_features) > 1:
                frame_diffs = np.diff(reconstructed_features, axis=0)
                mean_diffs = np.mean(frame_diffs, axis=0)
                std_diffs = np.std(frame_diffs, axis=0)
            else:
                mean_diffs = np.zeros(len(reconstructed_cols))
                std_diffs = np.zeros(len(reconstructed_cols))
            
            # Combine all encoder-based features
            motion_features = np.concatenate([
                mean_encoder,      # Mean encoder output
                std_encoder,       # Standard deviation
                min_encoder,       # Minimum values
                max_encoder,       # Maximum values
                first_frame,       # First frame
                last_frame,        # Last frame
                mean_diffs,        # Mean frame differences
                std_diffs          # Std frame differences
            ])
            
            features.append(motion_features)
            
        except Exception as e:
            print(f"   Warning: Could not load encoder features from {row['file_name']}: {e}")
            # Create zero features if file can't be loaded
            n_features = len(reconstructed_cols) if 'reconstructed_cols' in locals() else 50  # Default to 50 features
            zero_features = np.zeros(n_features * 8)  # 8 different feature types
            features.append(zero_features)
    
    # Create feature names
    n_encoder_dims = len(reconstructed_cols) if 'reconstructed_cols' in locals() else 50
    feature_names = (
        [f'mean_encoder_{i}' for i in range(n_encoder_dims)] +
        [f'std_encoder_{i}' for i in range(n_encoder_dims)] +
        [f'min_encoder_{i}' for i in range(n_encoder_dims)] +
        [f'max_encoder_{i}' for i in range(n_encoder_dims)] +
        [f'first_frame_{i}' for i in range(n_encoder_dims)] +
        [f'last_frame_{i}' for i in range(n_encoder_dims)] +
        [f'mean_diff_{i}' for i in range(n_encoder_dims)] +
        [f'std_diff_{i}' for i in range(n_encoder_dims)]
    )
    
    features = np.array(features)
    print(f"✅ Created encoder feature matrix: {features.shape}")
    print(f"   - Encoder dimensions: {n_encoder_dims}")
    print(f"   - Feature types: 8 (mean, std, min, max, first, last, mean_diff, std_diff)")
    print(f"   - Total features: {features.shape[1]}")
    
    return features, feature_names

def find_optimal_clusters(features: np.ndarray, max_clusters: int = 10) -> int:
    """
    Find optimal number of clusters using elbow method and silhouette score.
    """
    print("🔧 Finding optimal number of clusters...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Test different numbers of clusters
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_clusters + 1, len(features)))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features_scaled, cluster_labels))
    
    # Find optimal k (elbow method + silhouette score)
    # Use silhouette score as primary metric
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    print(f"📊 Optimal number of clusters: {optimal_k}")
    print(f"   - Silhouette score: {max(silhouette_scores):.3f}")
    
    return optimal_k

def perform_clustering(motion_df: pd.DataFrame, features: np.ndarray, n_clusters: int) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Perform K-means clustering on motion features.
    """
    print(f"🎯 Performing K-means clustering with {n_clusters} clusters...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to dataframe
    motion_df = motion_df.copy()
    motion_df['cluster'] = cluster_labels
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    print(f"✅ Clustering complete! Silhouette score: {silhouette_avg:.3f}")
    
    return motion_df, kmeans, scaler

def create_tsne_visualization(motion_df: pd.DataFrame, features: np.ndarray, 
                             kmeans: KMeans, scaler: StandardScaler, 
                             output_dir: Path):
    """
    Create a single t-SNE visualization of motion clustering.
    """
    print("📊 Creating t-SNE visualization...")
    
    # Standardize features for visualization
    features_scaled = scaler.transform(features)
    
    # Reduce to 2D using t-SNE
    print("   Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
    features_2d = tsne.fit_transform(features_scaled)
    
    # Create main clustering plot
    plt.figure(figsize=(12, 10))
    
    n_clusters = len(set(motion_df['cluster']))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        cluster_mask = motion_df['cluster'] == cluster_id
        cluster_data_2d = features_2d[cluster_mask]
        
        plt.scatter(cluster_data_2d[:, 0], cluster_data_2d[:, 1], 
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id} ({cluster_mask.sum()} motions)', 
                   alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'Motion Clustering - {n_clusters} Clusters (t-SNE Visualization)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motion_clustering_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ t-SNE visualization saved!")


def print_cluster_summary(motion_df: pd.DataFrame):
    """
    Print detailed cluster summary.
    """
    print(f"\n📊 CLUSTER SUMMARY")
    print("=" * 80)
    
    n_clusters = len(set(motion_df['cluster']))
    
    for cluster_id in range(n_clusters):
        cluster_data = motion_df[motion_df['cluster'] == cluster_id]
        
        print(f"\n🎯 Cluster {cluster_id} ({len(cluster_data)} motions):")
        print(f"   - Average duration: {cluster_data['total_frames'].mean():.1f} frames")
        print(f"   - Average unique codebooks: {cluster_data['unique_codebooks'].mean():.1f}")
        print(f"   - Average reconstruction error: {cluster_data['avg_reconstruction_error'].mean():.4f}")
        print(f"   - Average change rate: {(cluster_data['codebook_changes'] / cluster_data['total_frames']).mean():.4f}")
        
        # Show sample motion descriptions
        sample_descriptions = cluster_data['motion_description'].unique()[:3]
        print(f"   - Sample motions: {', '.join(sample_descriptions)}")
        
        # Show motion IDs
        motion_ids = sorted(cluster_data['motion_id'].tolist())
        print(f"   - Motion IDs: {motion_ids[:10]}{'...' if len(motion_ids) > 10 else ''}")

def main():
    parser = argparse.ArgumentParser(description='Motion clustering visualization based on codebook patterns')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/dhbaek/dh_workspace/vqvae_motion_g1/outputs/codebook_sequences',
                       help='Base directory containing codebook sequence CSV files')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/dhbaek/dh_workspace/vqvae_motion_g1/outputs/motion_clustering',
                       help='Output directory for clustering results')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (auto-detect if not specified)')
    parser.add_argument('--max_motions', type=int, default=None,
                       help='Maximum number of motions to process (for testing)')
    parser.add_argument('--max_clusters', type=int, default=10,
                       help='Maximum number of clusters to test for auto-detection')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Motion Clustering Visualization")
    print(f"===============================================================================")
    print(f"Input directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max motions: {args.max_motions or 'All'}")
    print()
    
    # Load motion data
    motion_df = load_codebook_sequences(args.base_dir, args.max_motions)
    
    if len(motion_df) == 0:
        print("❌ No motion data loaded!")
        return
    
    # Extract encoder features
    features, feature_names = extract_encoder_features(motion_df, args.base_dir)
    
    # Determine number of clusters
    if args.n_clusters is None:
        n_clusters = find_optimal_clusters(features, args.max_clusters)
    else:
        n_clusters = args.n_clusters
    
    # Perform clustering
    motion_df, kmeans, scaler = perform_clustering(motion_df, features, n_clusters)
    
    # Create t-SNE visualization
    create_tsne_visualization(motion_df, features, kmeans, scaler, output_path)
    
    # Print summary
    print_cluster_summary(motion_df)
    
    # Save results
    motion_df.to_csv(output_path / 'clustered_motions.csv', index=False)
    
    print(f"\n✅ Motion clustering complete!")
    print(f"📁 Results saved to: {output_path}")
    print(f"📊 Created {n_clusters} clusters from {len(motion_df)} motions")
    print(f"📈 Visualization saved:")
    print(f"   - motion_clustering_tsne.png")
    print(f"📋 Cluster assignments saved to: clustered_motions.csv")

if __name__ == "__main__":
    main()

'''
Example usage:

# Basic clustering with auto-detection using encoder features
python scripts/motion_clustering_visualization.py

python scripts/motion_clustering_visualization.py --n_clusters 5
'''
