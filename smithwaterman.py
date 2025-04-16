# import os
# import numpy as np
# import pandas as pd
# from Bio import Entrez, SeqIO

# # Set NCBI Entrez Email
# Entrez.email = "vibhusanchana2005@gmail.com"

# # Define Folder for Saving Sequences
# save_folder = os.path.join(os.getcwd(), "nrps_data")
# os.makedirs(save_folder, exist_ok=True)

# # Function to Fetch NRPS Sequences from NCBI
# def fetch_nrps_sequences():
#     search_query = "Nonribosomal peptide synthetase[Title] AND bacteria[Organism]"

#     # Search for NRPS sequences
#     handle = Entrez.esearch(db="protein", term=search_query, retmax=100)
#     record = Entrez.read(handle)
#     handle.close()

#     # Get sequence IDs
#     sequence_ids = record["IdList"]
#     print(f"Found {len(sequence_ids)} NRPS sequences.")

#     # Fetch sequence data
#     handle = Entrez.efetch(db="protein", id=sequence_ids, rettype="fasta", retmode="text")
#     sequences = handle.read()
#     handle.close()

#     # Save to a file
#     file_path = os.path.join(save_folder, "nrps_new.fasta")
#     with open(file_path, "w") as file:
#         file.write(sequences)

#     print(f"✅ NRPS sequences saved in: {file_path}")
#     return file_path

# # Function to Preprocess Sequences without TensorFlow/Keras
# def preprocess_sequences(fasta_file):
#     sequences = []
#     sequence_labels = []

#     # Read FASTA file
#     with open(fasta_file, "r") as file:
#         for record in SeqIO.parse(file, "fasta"):
#             sequences.append(str(record.seq))
#             sequence_labels.append(record.id)

#     # Remove duplicate sequences
#     df = pd.DataFrame({"ID": sequence_labels, "Sequence": sequences})
#     df.drop_duplicates(subset="Sequence", keep="first", inplace=True)

#     # Convert sequences to numerical features
#     unique_chars = sorted(set("".join(df["Sequence"])))  # Get unique amino acids
#     char_to_int = {char: i for i, char in enumerate(unique_chars)}

#     # Encode sequences into numerical format
#     encoded_sequences = []
#     for seq in df["Sequence"]:
#         encoded_seq = [char_to_int[char] for char in seq]
#         encoded_sequences.append(encoded_seq)
    
#     df["Encoded_Sequence"] = encoded_sequences
    
#     # Save processed data
#     processed_file = os.path.join(save_folder, "nrps_processed.csv")
#     df.to_csv(processed_file, index=False)
    
#     # Create a simple vocabulary and tokenized sequences
#     vocab = {char: idx+1 for idx, char in enumerate(unique_chars)}
    
#     # Instead of Keras tokenization, do a simple version
#     tokenized_sequences = []
#     max_length = max(len(seq) for seq in sequences)
    
#     for seq in sequences:
#         # Convert to tokens
#         tokens = [vocab[char] for char in seq]
#         # Pad sequence to max_length
#         padded = tokens + [0] * (max_length - len(tokens))
#         tokenized_sequences.append(padded)
    
#     # Save tokenized data
#     tokenized_array = np.array(tokenized_sequences)
#     np.save(os.path.join(save_folder, "tokenized_sequences.npy"), tokenized_array)
    
#     # Save vocabulary
#     with open(os.path.join(save_folder, "vocabulary.txt"), "w") as f:
#         for char, idx in vocab.items():
#             f.write(f"{char}\t{idx}\n")
    
#     print(f"✅ Preprocessed sequences saved in: {processed_file}")
#     print(f"✅ Tokenized sequences saved in: {os.path.join(save_folder, 'tokenized_sequences.npy')}")
#     print(f"✅ Vocabulary saved in: {os.path.join(save_folder, 'vocabulary.txt')}")
    
#     return df, tokenized_array, vocab

# # Run the Data Collection & Preprocessing Steps
# if __name__ == "__main__":
#     fasta_file = fetch_nrps_sequences()  # Download NRPS sequences
#     df, tokenized_data, vocab = preprocess_sequences(fasta_file)  # Preprocess them
    
#     print("\nData processing complete! Summary:")
#     print(f"Number of unique sequences: {len(df)}")
#     print(f"Number of unique amino acids: {len(vocab)}")
#     print(f"Maximum sequence length: {tokenized_data.shape[1]}")
import pandas as pd
from Bio import SeqIO
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import time

def extract_sequences_from_csv(csv_file):
    """
    Extract sequences from a CSV file containing NRPS data.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing NRPS sequences

    Returns:
    --------
    list of tuples
        (sequence_id, sequence) pairs
    """
    print(f"Reading sequences from {csv_file}...")

    # Read the CSV file
    df = pd.read_csv(os.path.abspath(csv_file))

    # Find the sequence column
    seq_col = None
    id_col = None

    # Common column names for sequences
    seq_columns = ["sequence", "seq", "protein_sequence", "dna_sequence", "aa_sequence", "nrps_sequence"]
    id_columns = ["id", "sequence_id", "protein_id", "accession", "name"]

    # Try to identify the sequence column
    for col in df.columns:
        if col.lower() in seq_columns:
            seq_col = col
            break

    # Try to identify the ID column
    for col in df.columns:
        if col.lower() in id_columns:
            id_col = col
            break

    # If no sequence column found, try to find a column with long string content
    if seq_col is None:
        # Look for columns with string values longer than 20 characters
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check a sample of rows
                sample = df[col].dropna().head(5)
                if any(isinstance(val, str) and len(val) > 20 for val in sample):
                    seq_col = col
                    print(f"Using column '{col}' as the sequence column based on content length")
                    break

    # If still no sequence column found, use the last column as a fallback
    if seq_col is None:
        seq_col = df.columns[-1]
        print(f"No obvious sequence column found. Using last column '{seq_col}' as fallback.")

    # If no ID column found, create numeric IDs
    sequences = []
    for idx, row in df.iterrows():
        if id_col:
            seq_id = str(row[id_col])
        else:
            seq_id = f"NRPS_seq_{idx+1}"

        # Get the sequence and clean it (remove spaces, ensure uppercase for amino acids)
        sequence = str(row[seq_col]).strip().upper()
        sequence = ''.join(sequence.split())  # Remove all whitespace

        if len(sequence) > 10:  # Only include sequences of reasonable length
            sequences.append((seq_id, sequence))

    print(f"Extracted {len(sequences)} sequences from {csv_file}")
    return sequences

def read_mibig_sequences(mibig_file):
    """
    Read sequences from the MIBiG database FASTA file.

    Parameters:
    -----------
    mibig_file : str
        Path to the MIBiG FASTA file

    Returns:
    --------
    list of tuples
        (sequence_id, sequence) pairs
    """
    print(f"Reading MIBiG sequences from {mibig_file}...")

    mibig_sequences = []
    for record in SeqIO.parse(mibig_file, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)
        mibig_sequences.append((seq_id, sequence))

    print(f"Read {len(mibig_sequences)} sequences from MIBiG database")
    return mibig_sequences

def smith_waterman(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2):
    """
    Implements the Smith-Waterman algorithm for local sequence alignment.

    Parameters:
    -----------
    seq1, seq2 : str
        The two sequences to align
    match_score : int
        Score for matching characters (default: 2)
    mismatch_penalty : int
        Penalty for mismatched characters (default: -1)
    gap_penalty : int
        Penalty for introducing a gap (default: -2)

    Returns:
    --------
    max_score : float
        The maximum alignment score
    aligned_seq1, aligned_seq2 : str
        The aligned subsequences
    identity : float
        Percentage identity between aligned sequences
    """
    # Initialize the scoring matrix
    rows, cols = len(seq1) + 1, len(seq2) + 1
    score_matrix = np.zeros((rows, cols), dtype=float)

    # Initialize the traceback matrix
    # 0 = end, 1 = diagonal, 2 = up, 3 = left
    traceback_matrix = np.zeros((rows, cols), dtype=int)

    # Fill the scoring matrix
    max_score = 0
    max_i, max_j = 0, 0

    for i in range(1, rows):
        for j in range(1, cols):
            # Calculate match score
            if seq1[i-1] == seq2[j-1]:
                match = score_matrix[i-1][j-1] + match_score
            else:
                match = score_matrix[i-1][j-1] + mismatch_penalty

            # Calculate gap scores
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty

            # Take the maximum score (or 0 if all scores are negative)
            score_matrix[i][j] = max(0, match, delete, insert)

            # Update traceback matrix
            if score_matrix[i][j] == 0:
                traceback_matrix[i][j] = 0
            elif score_matrix[i][j] == match:
                traceback_matrix[i][j] = 1
            elif score_matrix[i][j] == delete:
                traceback_matrix[i][j] = 2
            else:
                traceback_matrix[i][j] = 3

            # Update maximum score
            if score_matrix[i][j] > max_score:
                max_score = score_matrix[i][j]
                max_i, max_j = i, j

    # Traceback to find the aligned sequences
    aligned_seq1 = ""
    aligned_seq2 = ""
    i, j = max_i, max_j

    while traceback_matrix[i][j] != 0:
        if traceback_matrix[i][j] == 1:  # Diagonal
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 2:  # Up
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = '-' + aligned_seq2
            i -= 1
        else:  # Left
            aligned_seq1 = '-' + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1

    # Calculate sequence identity (percentage of exact matches)
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
    identity = (matches / len(aligned_seq1)) * 100 if aligned_seq1 else 0

    return max_score, aligned_seq1, aligned_seq2, identity

def process_sequence_pair(nrps_seq, mibig_seq):
    """
    Process a pair of sequences using Smith-Waterman algorithm.

    Parameters:
    -----------
    nrps_seq : tuple
        (id, sequence) for NRPS sequence
    mibig_seq : tuple
        (id, sequence) for MIBiG sequence

    Returns:
    --------
    dict
        Alignment results
    """
    nrps_id, nrps_sequence = nrps_seq
    mibig_id, mibig_sequence = mibig_seq

    # Run Smith-Waterman alignment
    score, aligned_nrps, aligned_mibig, identity = smith_waterman(nrps_sequence, mibig_sequence)

    return {
        'nrps_id': nrps_id,
        'mibig_id': mibig_id,
        'score': score,
        'identity': identity,
        'aligned_nrps': aligned_nrps,
        'aligned_mibig': aligned_mibig,
        'nrps_length': len(nrps_sequence),
        'mibig_length': len(mibig_sequence),
        'alignment_length': len(aligned_nrps)
    }

def run_comparisons(nrps_sequences, mibig_sequences, output_dir="results", max_mibig=100, top_n=5, num_workers=4):
    """
    Run Smith-Waterman comparisons between NRPS sequences and MIBiG sequences.

    Parameters:
    -----------
    nrps_sequences : list
        List of (id, sequence) tuples for NRPS sequences
    mibig_sequences : list
        List of (id, sequence) tuples for MIBiG sequences
    output_dir : str
        Directory to save results
    max_mibig : int
        Maximum number of MIBiG sequences to compare with each NRPS sequence
    top_n : int
        Number of top matches to keep for each NRPS sequence
    num_workers : int
        Number of parallel workers

    Returns:
    --------
    dict
        Dictionary of results for each NRPS sequence
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Limit the number of MIBiG sequences if needed
    if max_mibig and max_mibig < len(mibig_sequences):
        print(f"Limiting comparison to first {max_mibig} MIBiG sequences")
        mibig_sequences = mibig_sequences[:max_mibig]

    all_results = {}

    # Set up progress tracking
    total_comparisons = len(nrps_sequences) * len(mibig_sequences)
    print(f"Running {total_comparisons} comparisons between {len(nrps_sequences)} NRPS sequences and {len(mibig_sequences)} MIBiG sequences")

    start_time = time.time()

    # Use ProcessPoolExecutor for parallelization
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process each NRPS sequence
        for nrps_idx, nrps_seq in enumerate(nrps_sequences):
            nrps_id = nrps_seq[0]
            print(f"Processing NRPS sequence {nrps_idx+1}/{len(nrps_sequences)}: {nrps_id}")

            # Create a list of futures
            futures = []
            for mibig_seq in mibig_sequences:
                futures.append(executor.submit(process_sequence_pair, nrps_seq, mibig_seq))

            # Collect results as they complete
            sequence_results = []
            for future in futures:
                sequence_results.append(future.result())

            # Sort by score and keep top matches
            sequence_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = sequence_results[:top_n]

            # Save sequence-specific results
            all_results[nrps_id] = top_results

            # Write detailed results to file
            # Replace invalid characters in the filename
            safe_nrps_id = nrps_id.replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
            with open(os.path.join(output_dir, f"{safe_nrps_id}_alignments.txt"), "w") as f:

                f.write(f"Top {top_n} alignments for NRPS sequence {nrps_id}:\n\n")

                for i, result in enumerate(top_results):
                    f.write(f"Rank {i+1}: MIBiG {result['mibig_id']}\n")
                    f.write(f"Score: {result['score']}, Identity: {result['identity']:.2f}%\n")
                    f.write(f"Alignment length: {result['alignment_length']} positions\n")

                    # Format the aligned sequences with a line showing matches
                    match_line = ""
                    for a, b in zip(result['aligned_nrps'], result['aligned_mibig']):
                        if a == b:
                            match_line += "|"
                        elif a == "-" or b == "-":
                            match_line += " "
                        else:
                            match_line += "."

                    f.write("NRPS: " + result['aligned_nrps'] + "\n")
                    f.write("      " + match_line + "\n")
                    f.write("MIBiG: " + result['aligned_mibig'] + "\n\n")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Completed all comparisons in {elapsed_time:.2f} seconds")

    # Generate a summary report
    create_summary_report(all_results, nrps_sequences, output_dir)

    return all_results

def create_summary_report(all_results, nrps_sequences, output_dir):
    """
    Create a summary report of all alignments.

    Parameters:
    -----------
    all_results : dict
        Dictionary of results for each NRPS sequence
    nrps_sequences : list
        List of NRPS sequences
    output_dir : str
        Directory to save results
    """
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("NRPS vs MIBiG Database Alignment Summary\n")
        f.write("========================================\n\n")

        f.write(f"Analyzed {len(nrps_sequences)} NRPS sequences against the MIBiG database\n\n")

        f.write("Best matches for each NRPS sequence:\n\n")

        for nrps_id, results in all_results.items():
            if results:
                best_match = results[0]
                f.write(f"NRPS: {nrps_id} (length: {best_match['nrps_length']})\n")
                f.write(f"  Best match: {best_match['mibig_id']} (length: {best_match['mibig_length']})\n")
                f.write(f"  Score: {best_match['score']}, Identity: {best_match['identity']:.2f}%\n")
                f.write(f"  Alignment length: {best_match['alignment_length']} positions\n\n")
            else:
                f.write(f"NRPS: {nrps_id} - No significant matches found\n\n")

    # Create a visualization of the results
    visualize_results(all_results, output_dir)

def visualize_results(all_results, output_dir):
    """
    Create visualizations of the alignment results.

    Parameters:
    -----------
    all_results : dict
        Dictionary of results for each NRPS sequence
    output_dir : str
        Directory to save results
    """
    # Collect scores and identities
    scores = []
    identities = []
    nrps_ids = []

    for nrps_id, results in all_results.items():
        if results:
            best_match = results[0]
            scores.append(best_match['score'])
            identities.append(best_match['identity'])
            nrps_ids.append(nrps_id)

    # Create score vs identity plot
    plt.figure(figsize=(10, 6))
    plt.scatter(scores, identities, alpha=0.7)

    # Add labels and trend line
    plt.xlabel('Alignment Score')
    plt.ylabel('Sequence Identity (%)')
    plt.title('Alignment Score vs Sequence Identity')

    # Add trend line
    if scores and identities:
        z = np.polyfit(scores, identities, 1)
        p = np.poly1d(z)
        plt.plot(sorted(scores), p(sorted(scores)), "r--", alpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "score_vs_identity.png"), dpi=300)

    # Create a histogram of best match scores
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Best Match Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best Match Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=300)

def main():
    """
    Main function to run the NRPS vs MIBiG comparison workflow.
    """
    # Define input files and parameters - adjusted for VS Code local paths
    # Change these paths to match your file locations on your computer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nrps_csv = os.path.join(script_dir, "nrps_data", "nrps_processed.csv")
    mibig_fasta = os.path.join(script_dir, "mibig_prot_seqs_4.0.fasta")
    output_dir = os.path.join(script_dir, "nrps_mibig_comparison")

    # Parameters for the comparison
    max_mibig = 500  # Maximum number of MIBiG sequences to compare with each NRPS sequence
    top_n = 5  # Number of top matches to keep for each NRPS sequence
    
    # Adjust number of workers based on your CPU
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core
    print(f"Using {num_workers} worker processes")

    # Extract NRPS sequences from CSV
    nrps_sequences = extract_sequences_from_csv(nrps_csv)

    # Read MIBiG sequences
    mibig_sequences = read_mibig_sequences(mibig_fasta)

    # Run comparisons
    results = run_comparisons(
        nrps_sequences,
        mibig_sequences,
        output_dir=output_dir,
        max_mibig=max_mibig,
        top_n=top_n,
        num_workers=num_workers
    )

    print(f"Analysis complete. Results saved to '{output_dir}' directory.")
    print(f"Check '{output_dir}/summary_report.txt' for a summary of the results.")
    # Extract NRPS sequences from CSV
    all_nrps_sequences = extract_sequences_from_csv(nrps_csv)
    
    # ADD THIS BLOCK - Check which sequences have already been processed
    processed_ids = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith("_alignments.txt"):
                # Extract the sequence ID from the filename
                seq_id = filename.replace("_alignments.txt", "")
                processed_ids.add(seq_id)
    
    # Filter out already processed sequences
    nrps_sequences = []
    for nrps_seq in all_nrps_sequences:
        seq_id, sequence = nrps_seq
        # Convert to safe filename format for comparison
        safe_id = seq_id.replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
        if safe_id not in processed_ids:
            nrps_sequences.append((seq_id, sequence))
    
    print(f"Already processed: {len(processed_ids)} sequences")
    print(f"Remaining to process: {len(nrps_sequences)} sequences")
    
    if not nrps_sequences:
        print("All sequences have been processed. Nothing to do.")
        return
    # END OF ADDED BLOCK
    
    # Read MIBiG sequences
    mibig_sequences = read_mibig_sequences(mibig_fasta)
    
    # ... rest of your existing code using 'nrps_sequences' ...
if __name__ == "__main__":
    main()
