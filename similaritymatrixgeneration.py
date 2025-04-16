import os
import re
import pandas as pd

# Directory containing your Smith-Waterman output files
directory = "nrps_mibig_comparison"  # Update this path to point to your folder

# Dictionary to store parsed data
results = {}

# Process each file
file_count = 0
print("Looking for alignment files in:", os.path.abspath(directory))

for filename in os.listdir(directory):
    if filename.endswith("_alignments.txt"):  # Match your specific file pattern
        file_path = os.path.join(directory, filename)
        file_count += 1
        
        # Extract query sequence ID from filename (e.g., "ABR83084.1" from "ABR83084.1_alignments.txt")
        query_id = filename.split("_alignments.txt")[0]
        
        print(f"Processing file {file_count}: {filename}")
        
        # Read the file content
        with open(file_path, 'r') as file:
            content = file.read()
            
            # Extract the actual NRPS sequence ID from the header
            header_match = re.search(r'Top \d+ alignments for NRPS sequence ([^:]+):', content)
            if header_match:
                query_id = header_match.group(1).strip()
            
            # Find all alignment blocks using the Rank pattern
            alignment_blocks = re.findall(r'(Rank \d+:.+?)(?=Rank \d+:|$)', content, re.DOTALL)
            
            for block in alignment_blocks:
                if not block.strip():
                    continue
                
                # Extract the MIBiG target ID
                target_match = re.search(r'Rank \d+: MIBiG ([^\s|\]]+)', block)
                if target_match:
                    target_id = target_match.group(1).strip()
                else:
                    continue  # Skip if no target ID found
                
                # Extract gene cluster info if available
                gene_match = re.search(r'\|([^|]+)\|', block)
                gene_id = gene_match.group(1).strip() if gene_match else ""
                
                # Create a more informative pair ID
                pair_id = f"{query_id}_{target_id}"
                
                # Extract key metrics
                score_match = re.search(r'Score: (\d+\.?\d*)', block)
                identity_match = re.search(r'Identity: (\d+\.?\d*)%', block)
                align_len_match = re.search(r'Alignment length: (\d+)', block)
                
                # Store extracted data
                results[pair_id] = {
                    'query_id': query_id,
                    'target_id': target_id,
                    'gene_id': gene_id
                }
                
                if score_match:
                    results[pair_id]['score'] = float(score_match.group(1))
                if identity_match:
                    results[pair_id]['identity'] = float(identity_match.group(1))
                if align_len_match:
                    results[pair_id]['alignment_length'] = int(align_len_match.group(1))
                
                # Extract the alignment sequences
                nrps_match = re.search(r'NRPS: ([^\n]+)', block)
                mibig_match = re.search(r'MIBIG: ([^\n]+)', block)
                match_line = re.search(r'NRPS:.*\n([^\n]+)\nMIBIG:', block)
                
                if nrps_match and mibig_match:
                    results[pair_id]['nrps_sequence'] = nrps_match.group(1).strip()
                    results[pair_id]['mibig_sequence'] = mibig_match.group(1).strip()
                    if match_line:
                        results[pair_id]['match_pattern'] = match_line.group(1).strip()
                
                # Extract rank information
                rank_match = re.search(r'Rank (\d+):', block)
                if rank_match:
                    results[pair_id]['rank'] = int(rank_match.group(1))

# Convert to DataFrame for easier analysis
df = pd.DataFrame.from_dict(results, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index': 'sequence_pair'}, inplace=True)

# Save the parsed data
df.to_csv("parsed_alignments.csv", index=False)

# Create similarity matrix
unique_sequences = set()
for pair_id in results:
    query_id = results[pair_id]['query_id']
    target_id = results[pair_id]['target_id']
    unique_sequences.add(query_id)
    unique_sequences.add(target_id)

unique_sequences = list(unique_sequences)
similarity_matrix = pd.DataFrame(0, 
                               index=unique_sequences, 
                               columns=unique_sequences)

# Fill in similarity matrix with identity percentages
for pair_id, data in results.items():
    if 'identity' in data:
        query_id = data['query_id']
        target_id = data['target_id']
        if query_id in unique_sequences and target_id in unique_sequences:
            similarity_matrix.loc[query_id, target_id] = data['identity']
            similarity_matrix.loc[target_id, query_id] = data['identity']  # Mirror matrix

# Set diagonal to 100% (self-identity)
for seq in unique_sequences:
    similarity_matrix.loc[seq, seq] = 100.0

# Save similarity matrix
similarity_matrix.to_csv("similarity_matrix.csv")

print(f"Processed {file_count} alignment files")
print(f"Found {len(results)} valid alignments")
print(f"Created similarity matrix of size {len(unique_sequences)}x{len(unique_sequences)}")
print(f"Results saved to 'parsed_alignments.csv' and 'similarity_matrix.csv'")