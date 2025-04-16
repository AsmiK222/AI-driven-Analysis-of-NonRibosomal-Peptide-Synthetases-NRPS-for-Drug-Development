import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_data(similarity_matrix_path, alignment_path):
    """
    Load the similarity matrix and alignment information from CSV files.
    
    Args:
        similarity_matrix_path (str): Path to the CSV file containing the similarity matrix
        alignment_path (str): Path to the CSV file containing the alignment information
        
    Returns:
        tuple: (similarity_matrix, alignment_data)
    """
    similarity_matrix = pd.read_csv(similarity_matrix_path, index_col=0)
    alignment_data = pd.read_csv(alignment_path)
    
    return similarity_matrix, alignment_data

def identify_regions_of_interest(similarity_matrix, threshold=0.7, min_region_size=10):
    """
    Identify regions of interest based on similarity scores.
    
    Args:
        similarity_matrix (pd.DataFrame): The similarity matrix
        threshold (float): Similarity threshold to consider a region of interest
        min_region_size (int): Minimum size of a region to be considered significant
        
    Returns:
        list: List of tuples (start_idx, end_idx) for each region of interest
    """
    # Calculate the average similarity for each position
    avg_similarity = similarity_matrix.mean(axis=1)
    
    # Find regions where average similarity is above the threshold
    above_threshold = avg_similarity > threshold
    
    # Identify continuous regions
    regions = []
    in_region = False
    start_idx = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_region:
            # Start of a new region
            in_region = True
            start_idx = i
        elif not is_above and in_region:
            # End of a region
            in_region = False
            if i - start_idx >= min_region_size:
                regions.append((start_idx, i))
    
    # Check if we ended while still in a region
    if in_region and len(above_threshold) - start_idx >= min_region_size:
        regions.append((start_idx, len(above_threshold)))
    
    return regions

def find_boundaries_using_signal_processing(similarity_matrix, window_size=5):
    """
    Use signal processing to identify natural boundaries in the sequences.
    
    Args:
        similarity_matrix (pd.DataFrame): The similarity matrix
        window_size (int): Window size for smoothing
        
    Returns:
        list: Positions of natural boundaries
    """
    # Calculate row-wise variance as a measure of conservation variability
    row_variance = similarity_matrix.var(axis=1)
    
    # Apply smoothing using rolling window
    smoothed_variance = row_variance.rolling(window=window_size, center=True).mean()
    smoothed_variance = smoothed_variance.fillna(smoothed_variance.mean())
    
    # Find peaks in the variance (potential boundaries between domains or regions)
    peaks, _ = find_peaks(smoothed_variance, height=smoothed_variance.mean(), distance=window_size*2)
    
    return sorted(peaks)

def divide_sequences(alignment_data, boundaries):
    """
    Divide sequences into subsequences based on identified boundaries.
    
    Args:
        alignment_data (pd.DataFrame): The alignment information
        boundaries (list): List of boundary positions
        
    Returns:
        dict: Dictionary containing subsequences for each region
    """
    # Convert boundaries to a list including start and end positions
    all_boundaries = [0] + list(boundaries) + [len(alignment_data)]
    
    # Create regions
    subsequence_regions = {}
    
    for i in range(len(all_boundaries) - 1):
        start = all_boundaries[i]
        end = all_boundaries[i+1]
        
        region_name = f"region_{i+1}"
        subsequence_regions[region_name] = alignment_data.iloc[start:end].copy()
    
    return subsequence_regions

def analyze_region(region_data, similarity_submatrix=None):
    """
    Analyze a specific region of the alignment.
    
    Args:
        region_data (pd.DataFrame): Alignment data for the region
        similarity_submatrix (pd.DataFrame): Similarity matrix for the region
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Example analysis metrics
    results = {
        "size": len(region_data),
        "conservation_score": None,
        "gap_percentage": None
    }
    
    # Calculate conservation if similarity matrix is available
    if similarity_submatrix is not None:
        results["conservation_score"] = similarity_submatrix.mean().mean()
    
    # Calculate gap percentage if sequence data is available
    if "sequence" in region_data.columns:
        gap_count = region_data["sequence"].str.count('-').sum()
        total_chars = region_data["sequence"].str.len().sum()
        results["gap_percentage"] = gap_count / total_chars if total_chars > 0 else 0
    
    return results

def visualize_regions(similarity_matrix, regions, boundaries, output_path="region_analysis.png"):
    """
    Visualize the identified regions and boundaries.
    
    Args:
        similarity_matrix (pd.DataFrame): The similarity matrix
        regions (list): List of region tuples (start, end)
        boundaries (list): List of boundaries
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Plot average similarity
    avg_similarity = similarity_matrix.mean(axis=1)
    plt.plot(avg_similarity, label="Average Similarity", color="blue")
    
    # Highlight regions of interest
    for start, end in regions:
        plt.axvspan(start, end, alpha=0.2, color="green")
    
    # Mark boundaries
    for boundary in boundaries:
        plt.axvline(x=boundary, color="red", linestyle="--", alpha=0.7)
    
    plt.xlabel("Position")
    plt.ylabel("Average Similarity")
    plt.title("Sequence Analysis: Regions of Interest and Boundaries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def main(similarity_matrix_path, alignment_path, output_dir="./output"):
    """
    Main function to run the divide and conquer analysis.
    
    Args:
        similarity_matrix_path (str): Path to the similarity matrix CSV
        alignment_path (str): Path to the alignment data CSV
        output_dir (str): Directory to save output files
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    similarity_matrix, alignment_data = load_data(similarity_matrix_path, alignment_path)
    
    # Step 1: Identify regions of interest based on similarity
    regions = identify_regions_of_interest(similarity_matrix)
    print(f"Identified {len(regions)} regions of interest")
    
    # Step 2: Find natural sequence boundaries using signal processing
    boundaries = find_boundaries_using_signal_processing(similarity_matrix)
    print(f"Identified {len(boundaries)} natural boundaries")
    
    # Step 3: Divide sequences into subsequences
    subsequence_regions = divide_sequences(alignment_data, boundaries)
    print(f"Divided sequences into {len(subsequence_regions)} regions")
    
    # Step 4: Analyze each region separately
    region_results = {}
    for region_name, region_data in subsequence_regions.items():
        # Extract corresponding part of similarity matrix
        region_start = region_data.index[0] if not region_data.empty else 0
        region_end = region_data.index[-1] + 1 if not region_data.empty else 0
        
        similarity_submatrix = similarity_matrix.iloc[region_start:region_end, region_start:region_end]
        
        # Analyze the region
        region_results[region_name] = analyze_region(region_data, similarity_submatrix)
        
        # Save region data
        region_data.to_csv(f"{output_dir}/{region_name}_alignment.csv")
    
    # Save analysis results
    results_df = pd.DataFrame.from_dict(region_results, orient="index")
    results_df.to_csv(f"{output_dir}/region_analysis_summary.csv")
    
    # Visualize the regions
    visualize_regions(similarity_matrix, regions, boundaries, f"{output_dir}/region_visualization.png")
    
    return subsequence_regions, region_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Divide and conquer analysis for sequence alignments")
    parser.add_argument("--similarity", required=True, help="Path to similarity matrix CSV file")
    parser.add_argument("--alignment", required=True, help="Path to alignment data CSV file")
    parser.add_argument("--output", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    main(args.similarity, args.alignment, args.output)