import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

# Create output directory
output_dir = "./protein_clusters"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load your similarity matrix
similarity_matrix = pd.read_csv("similarity_matrix.csv", index_col=0)
print(f"Loaded similarity matrix with shape: {similarity_matrix.shape}")

# Convert to distance matrix
if similarity_matrix.max().max() > 1:
    distance_matrix = 100 - similarity_matrix
else:
    distance_matrix = 1 - similarity_matrix

# Perform hierarchical clustering
condensed_dist = []
n = distance_matrix.shape[0]
for i in range(n):
    for j in range(i+1, n):
        condensed_dist.append(distance_matrix.iloc[i, j])

print(f"Created condensed distance matrix with {len(condensed_dist)} elements")

# Check if condensed distance matrix is valid
if len(condensed_dist) == 0:
    print("Error: Empty distance matrix. Check your similarity matrix data.")
else:
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # Visualize the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(
        linkage_matrix,
        labels=distance_matrix.index,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title('Hierarchical Clustering of Proteins')
    plt.xlabel('Protein Sequences')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hierarchical_clustering_dendrogram.png")
    
    print("Clustering completed successfully!")