import pandas as pd

# 1. Fix Domain-Function Mapping File
print("Processing domain-function mapping file...")
try:
    df = pd.read_csv('protein_functions.csv')
    
    domain_function_pairs = []
    for index, row in df.iterrows():
        domains = row['Domains'].split('; ')
        functions = row['Functions'].split('; ')
        
        for domain in domains:
            domain = domain.strip()
            for function in functions:
                function = function.strip()
                domain_function_pairs.append([domain, function])
    
    domain_function_df = pd.DataFrame(domain_function_pairs, columns=['domain_id', 'fcol'])
    domain_function_df = domain_function_df.drop_duplicates()
    domain_function_df.to_csv('domain_function_mapping_fixed.csv', index=False)
    print("Domain-function mapping file created successfully!")
except Exception as e:
    print(f"Error processing domain-function file: {e}")

# 2. Fix Sequence-Domain Mapping File
print("\nProcessing sequence-domain mapping file...")
try:
    df = pd.read_csv('protein_domains.csv')
    df = df.rename(columns={
        'Query_Protein': 'sequence_id',
        'Domain': 'domain_id'
    })
    df.to_csv('sequence_domain_mapping_fixed.csv', index=False)
    print("Sequence-domain mapping file created successfully!")
except Exception as e:
    print(f"Error processing sequence-domain file: {e}")

# 3. Fix Sequence Similarity Matrix File
print("\nProcessing sequence similarity matrix file...")
try:
    df = pd.read_csv('similarity_matrix.csv')
    
    # Fix for the similarity matrix - rename first column to 'sequence_id_1'
    # This assumes the first column contains protein IDs
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'sequence_id_1'})
    else:
        # If the first column already has a name, we'll create a copy with the right name
        first_col_name = df.columns[0]
        df = df.rename(columns={first_col_name: 'sequence_id_1'})
    
    df.to_csv('sequence_similarity_fixed.csv', index=False)
    print("Sequence similarity file created successfully!")
except Exception as e:
    print(f"Error processing similarity matrix file: {e}")

print("\nAll files have been processed. Please upload the *_fixed.csv files to your model.")