import pandas as pd
from Bio import SeqIO
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import time
import sys

# Import all functions from your original script
# Assuming your original script is named nrps_comparison.py
# If it has a different name, change it here
from d14 import extract_sequences_from_csv, read_mibig_sequences, smith_waterman, process_sequence_pair, run_comparisons, create_summary_report, visualize_results

def resume_processing():
    """
    Function to resume NRPS vs MIBiG comparison workflow from where it left off.
    """
    # Define input files and parameters - using same paths as original script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nrps_csv = os.path.join(script_dir, "nrps_data", "nrps_processed.csv")
    mibig_fasta = os.path.join(script_dir, "mibig_prot_seqs_4.0.fasta")
    output_dir = os.path.join(script_dir, "nrps_mibig_comparison")
    
    # Parameters for the comparison - using same params as original script
    max_mibig = 500  
    top_n = 5  
    
    # Adjust number of workers based on your CPU
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1) 
    print(f"Using {num_workers} worker processes")

    # Extract all NRPS sequences from CSV
    all_nrps_sequences = extract_sequences_from_csv(nrps_csv)
    print(f"Total NRPS sequences: {len(all_nrps_sequences)}")
    
    # Check which sequences have already been processed
    processed_ids = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith("_alignments.txt"):
                # Extract the sequence ID from the filename
                seq_id = filename.replace("_alignments.txt", "")
                processed_ids.add(seq_id)
    
    # Filter out already processed sequences
    remaining_sequences = []
    for nrps_seq in all_nrps_sequences:
        seq_id, sequence = nrps_seq
        # Convert to safe filename format for comparison
        safe_id = seq_id.replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
        if safe_id not in processed_ids:
            remaining_sequences.append((seq_id, sequence))
    
    print(f"Already processed: {len(processed_ids)} sequences")
    print(f"Remaining to process: {len(remaining_sequences)} sequences")
    
    if not remaining_sequences:
        print("All sequences have been processed. Nothing to do.")
        return
    
    # Read MIBiG sequences
    mibig_sequences = read_mibig_sequences(mibig_fasta)

    # Run comparisons only on remaining sequences
    results = run_comparisons(
        remaining_sequences,
        mibig_sequences,
        output_dir=output_dir,
        max_mibig=max_mibig,
        top_n=top_n,
        num_workers=num_workers
    )

    print(f"Analysis complete. Results for remaining sequences saved to '{output_dir}' directory.")
    print(f"Note: Summary report and visualizations only include the newly processed sequences.")

if __name__ == "__main__":
    resume_processing()