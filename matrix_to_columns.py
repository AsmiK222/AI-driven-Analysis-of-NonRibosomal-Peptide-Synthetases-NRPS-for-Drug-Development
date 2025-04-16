import pandas as pd

# Load your similarity matrix file
# Replace with your actual file name
matrix_file = "similarity_matrix.csv"

# Read the matrix, assuming the first column is the row labels (sequence IDs)
df = pd.read_csv(matrix_file, index_col=0)

# Optional: Remove self-similarity rows if not needed (diagonal values)
# df.values[[range(len(df))]*2] = None  # sets diagonal to NaN

# Melt the DataFrame to get it in long/pairwise format
pairwise_df = df.reset_index().melt(id_vars=df.index.name, 
                                    var_name='sequence_id_2', 
                                    value_name='similarity')

# Rename the index column to 'sequence_id_1' for clarity
pairwise_df.rename(columns={df.index.name: 'sequence_id_1'}, inplace=True)

# Optional: Remove rows with missing similarity values (e.g., NaN)
pairwise_df.dropna(inplace=True)

# Save to a new CSV
pairwise_df.to_csv("pairwise_similarity.csv", index=False)

print("Conversion complete! Output saved as 'pairwise_similarity.csv'")
