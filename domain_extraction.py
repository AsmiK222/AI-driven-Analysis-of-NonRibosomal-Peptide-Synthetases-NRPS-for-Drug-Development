import os
import pandas as pd
import re

# Folder where all 95 files are stored
folder_path = "nrps_mibig_comparison"  # Change this to your actual folder path

# Initialize an empty list to store extracted data
protein_domain_data = []

# Regex pattern to capture Rank 1, 2, and 3 domain names
domain_pattern = re.compile(r"\|([^|]+)_protein\|")  

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # Assuming the files are .txt, change if needed
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            content = file.readlines()

            # Extract the query protein ID from the first line
            query_protein = content[0].split()[-1].strip(":")  

            # Find domains from Rank 1, 2, and 3
            domains = []
            rank_count = 0

            for line in content:
                if line.startswith(f"Rank {rank_count + 1}:"):  
                    match = domain_pattern.search(line)
                    if match:
                        domains.append(match.group(1))  # Extract domain name
                        rank_count += 1  
                    if rank_count == 3:  # Stop after extracting Rank 1, 2, and 3
                        break  

            # Store result
            protein_domain_data.append((query_protein, ", ".join(domains)))

# Convert to DataFrame
df = pd.DataFrame(protein_domain_data, columns=["Protein", "Domains"])

# Save as CSV
df.to_csv("protein_functions.csv", index=False)

print("✅ Multi-rank domain extraction completed! File saved as protein_functions.csv.")
