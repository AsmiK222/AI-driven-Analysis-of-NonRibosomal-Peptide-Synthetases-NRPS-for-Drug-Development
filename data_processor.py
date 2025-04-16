import pandas as pd
import numpy as np
import re
import os
from io import StringIO

def parse_similarity_matrix(matrix_text):
    """
    Parse similarity matrix from the text format in Image 1
    Returns DataFrame with pairwise similarities
    """
    # Split the matrix text into lines
    lines = matrix_text.strip().split('\n')
    
    # Extract sequence IDs from the header row
    header = lines[0].strip().split()
    seq_ids = header
    
    # Initialize empty matrix
    n = len(seq_ids)
    similarity_matrix = np.zeros((n, n))
    
    # Parse each row of the matrix
    for i in range(1, len(lines)):
        if i >= n + 1:
            break  # Stop if we've processed all rows
            
        row_data = lines[i].strip().split()
        row_id = row_data[0]
        
        # Find index of this sequence ID
        try:
            row_idx = seq_ids.index(row_id)
        except ValueError:
            print(f"Warning: Row ID {row_id} not found in header")
            continue
            
        # Parse similarity values
        for j in range(1, len(row_data)):
            if j >= n + 1:
                break  # Stop if we've processed all columns
                
            try:
                similarity_matrix[row_idx, j-1] = float(row_data[j])
            except (ValueError, IndexError):
                print(f"Warning: Error parsing value at row {i}, column {j}")
    
    # Convert to DataFrame with sequence IDs as indices and columns
    df_matrix = pd.DataFrame(similarity_matrix, index=seq_ids, columns=seq_ids)
    
    # Convert matrix to pairwise format
    pairs = []
    for i in range(n):
        for j in range(n):
            # Only include non-zero similarities
            if similarity_matrix[i, j] > 0:
                pairs.append({
                    'sequence_id_1': seq_ids[i],
                    'sequence_id_2': seq_ids[j],
                    'similarity_score': similarity_matrix[i, j]
                })
    
    return pd.DataFrame(pairs)

def parse_protein_domains(domain_text):
    """
    Parse protein domains from text format in Image 2
    Returns DataFrame mapping sequence IDs to domains
    """
    # Split the domain text into lines
    lines = domain_text.strip().split('\n')
    
    # Skip header
    data = []
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        # Parse protein ID and domain
        parts = line.split()
        if len(parts) >= 2:
            sequence_id = parts[0]
            domain = ' '.join(parts[1:])
            
            data.append({
                'sequence_id': sequence_id,
                'domain_id': domain
            })
    
    return pd.DataFrame(data)

def parse_protein_functions(function_text):
    """
    Parse protein functions from text format in Image 3
    Returns DataFrame with domain-function mappings
    """
    # Split the function text into lines
    lines = function_text.strip().split('\n')
    
    # Skip header
    domain_functions = []
    seen_domains = set()
    
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        # Split at first space to get protein ID
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
            
        protein_id = parts[0]
        rest = parts[1]
        
        # Extract domains and functions
        # This is a simplified approach - in reality you'd need more sophisticated parsing
        domains_and_functions = rest.split(';')
        
        for item in domains_and_functions:
            item = item.strip()
            
            # Extract domain type and function
            domain_match = re.search(r'^(\w+)', item)
            if domain_match:
                domain_type = domain_match.group(1)
                function = item
                
                # Only add unique domain-function pairs
                domain_function_key = f"{domain_type}:{function}"
                if domain_function_key not in seen_domains:
                    seen_domains.add(domain_function_key)
                    domain_functions.append({
                        'domain_id': domain_type,
                        'function': function
                    })
    
    return pd.DataFrame(domain_functions)

def extract_complete_datasets(similarity_matrix_text, protein_domains_text, protein_functions_text):
    """
    Process all three data sources and create required CSV files for training
    """
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Process similarity matrix
    print("Processing similarity matrix...")
    similarity_df = parse_similarity_matrix(similarity_matrix_text)
    similarity_df.to_csv('data/similarity.csv', index=False)
    print(f"Saved similarity.csv with {len(similarity_df)} pairs")
    
    # Process protein domains
    print("Processing protein domains...")
    domains_df = parse_protein_domains(protein_domains_text)
    domains_df.to_csv('data/sequence_domain.csv', index=False)
    print(f"Saved sequence_domain.csv with {len(domains_df)} entries")
    
    # Process protein functions
    print("Processing protein functions...")
    functions_df = parse_protein_functions(protein_functions_text)
    functions_df.to_csv('data/domain_function.csv', index=False)
    print(f"Saved domain_function.csv with {len(functions_df)} entries")
    
    # Create reference sequences (dummy sequences for now)
    print("Creating reference sequences...")
    all_sequences = set(similarity_df['sequence_id_1']).union(set(similarity_df['sequence_id_2']))
    ref_sequences = []
    
    for seq_id in all_sequences:
        # Generate a dummy sequence - in reality you would use real sequences
        # Length between 200-500 residues with ATGC
        seq_length = np.random.randint(200, 500)
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 
                       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        sequence = ''.join(np.random.choice(amino_acids, size=seq_length))
        
        ref_sequences.append({
            'sequence_id': seq_id,
            'sequence': sequence
        })
    
    ref_df = pd.DataFrame(ref_sequences)
    ref_df.to_csv('data/reference_sequences.csv', index=False)
    print(f"Saved reference_sequences.csv with {len(ref_df)} sequences")
    
    return {
        'similarity': similarity_df,
        'domains': domains_df,
        'functions': functions_df,
        'references': ref_df
    }

def generate_fasta_file(reference_df, output_file='data/reference_sequences.fasta'):
    """
    Convert reference sequences DataFrame to FASTA format
    """
    with open(output_file, 'w') as f:
        for _, row in reference_df.iterrows():
            f.write(f">{row['sequence_id']}\n")
            
            # Write sequence with line breaks every 60 characters
            seq = row['sequence']
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + '\n')
    
    print(f"Saved FASTA file to {output_file}")

# Example usage (with placeholders for your actual data)
if __name__ == "__main__":
    # In practice, you would load these from files or parse from images
    # For now, using placeholders
    
    similarity_matrix_text = """
    |BGC00000 WP_37527 WP_40570 WP_39569 WP_39403 XHV06635 XJV75695 WP_40786 WP_39398 WP_41190 WP_39404 WP_39407 XMF84248 WP_39560 WP_41290 WP_13835 WP_40734
    BGC00000      100        0        0     63.16        0        0        0        0        0        0        0        0        0        0        0        0        0
    WP_37527        0      100        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
    WP_40570        0        0      100        0        0        0        0        0        0        0        0        0        0        0        0        0        0
    WP_39569     63.16        0        0      100        0        0        0        0        0        0        0        0        0        0        0        0        0
    """  # This is a truncated example - you would use your full data
    
    protein_domains_text = """
    Query_Protein Domain
    ABR83084 proline_adenylation_protein
    ABR83084 BafX
    ABR83084 polyketide_synthase
    ABR83084 BafY
    ABR83084 PKS_I
    ARO18898 putative_dTDP-glucose_4,6-dehydratase
    ARO18898 PKS_I
    """  # This is a truncated example - you would use your full data
    
    protein_functions_text = """
    Protein_ID Domains Functions
    ABR83084 proline_ad Activation of proline for peptide synthesis; Bafilomycin biosynthesis component X; Catalyzes polyketide synthesis; Bafilomycin biosynthesis component Y; Type I polyketide synthase activity
    ARO18898 putative_d Dehydration of dTDP-glucose; Bafilomycin biosynthesis component AII; Type I polyketide synthase activity
    """  # This is a truncated example - you would use your full data
    
    # Process all data
    results = extract_complete_datasets(similarity_matrix_text, protein_domains_text, protein_functions_text)
    
    # Generate FASTA file
    generate_fasta_file(results['references'])
    
    print("\nAll files created successfully. Use these files to train your NRPS model:")
    print("- data/similarity.csv")
    print("- data/sequence_domain.csv")
    print("- data/domain_function.csv")
    print("- data/reference_sequences.csv")
    print("- data/reference_sequences.fasta")