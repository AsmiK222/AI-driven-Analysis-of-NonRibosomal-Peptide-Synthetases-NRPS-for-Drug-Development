import pandas as pd

# Load domain information
domains_df = pd.read_csv("protein_domains.csv")

# Extract all unique domains from the CSV for comprehensive mapping
all_domains = domains_df['Domain'].unique().tolist()

# Define a dictionary of known domain-to-function mappings
domain_to_function = {
    # Transport-related domains
    "ABC_transporter": "Transport of molecules across membrane",
    "ABC_transporter_protein": "Transport of molecules across membrane",
    "ABC_transporter_oligopeptide_permease": "Oligopeptide transport across membrane",
    "binding-protein_dependent_ABC_transporter": "Substrate-binding transport across membrane",
    "ABC_transporter_oligopeptide_binding_protein": "Selective peptide transport",
    "ATP-binding_ABC_peptide_transporter": "ATP-dependent peptide transport",
    "EmrB/QacA_drug_resistance_transporter": "Drug efflux and multidrug resistance",
    "putative_transporter": "Molecular transport (predicted)",
    "putative_HC-toxin_efflux_carrier_TOXA": "Toxin efflux",
    "Na+/H+_antiporter": "Ion exchange across membrane",
    
    # Enzymatic activity domains
    "polyketide_synthase": "Catalyzes polyketide synthesis",
    "polyketide_synthase_type_I": "Type I polyketide synthesis (modular)",
    "type_1_polyketide_synthase": "Type I polyketide synthesis (modular)",
    "PKS_I": "Type I polyketide synthase activity",
    "polyketide_synthase_AufC": "Specific polyketide synthesis (AufC variant)",
    "polyketide_synthase_AufD": "Specific polyketide synthesis (AufD variant)",
    "polyketide_synthase_AufF": "Specific polyketide synthesis (AufF variant)",
    "polyketide_synthase_AufG": "Specific polyketide synthesis (AufG variant)",
    "PKSN_polyketide_synthase_for_alternapyrone_biosynthesis": "Alternapyrone synthesis",
    "PksA": "Polyketide synthase involved in aflatoxin biosynthesis",
    "type_II_thioesterase": "Removal of aberrant intermediates in polyketide synthesis",
    
    # Catalytic domains
    "Serine_protease": "Protein degradation and digestion",
    "PKinase": "Signal transduction and cell regulation",
    "kinase": "Phosphorylation of substrates",
    "Methyltransferase": "Catalyzes methylation reactions",
    "methyltransferase": "Catalyzes methylation reactions",
    "NDP-hexose-C3-methyltransferase": "Methylation of NDP-hexose at C3 position",
    "N-methyltransferase": "Methylation of nitrogen atoms",
    "protein_methyltransferase": "Protein-specific methylation",
    "caffeoyl-CoA_o-methyltransferase": "Methylation in phenylpropanoid pathway",
    
    # Oxidation-related domains
    "Cytochrome_P450": "Drug metabolism and detoxification",
    "cytochrome_P450_monooxygenase": "Single oxygen insertion reactions",
    "monooxygenase_AufJ": "Oxygen insertion (AufJ variant)",
    "Oxidoreductase": "Electron transfer in redox reactions",
    "oxidoreductase_A": "Catalyzes oxidation-reduction reactions",
    "reductase/dehydrogenase": "Reduction and dehydrogenation reactions",
    "reductase": "Catalyzes reduction reactions",
    "aflatoxin_B1_aldehyde_reductase_member_3": "Reduction of aflatoxin aldehydes",
    "Zn-dependent_alcohol_dehydrogenase": "Zinc-dependent alcohol oxidation",
    "glucose_1-dehydrogenase": "Glucose oxidation",
    "acyl_CoA_dehydrogenase": "Fatty acid oxidation",
    "desaturase": "Introduction of double bonds",
    
    # Regulatory domains
    "transcriptional_regulator": "Controls gene expression",
    "LuxR_family_transcriptional_regulator": "Quorum sensing and gene regulation",
    "LuxR-family_transcription_regulator": "Quorum sensing and gene regulation",
    "SARP_transcriptional_regulator": "Regulation of antibiotic production",
    "pathway-specific_SARP_activator": "Pathway-specific gene activation",
    "TetR_regulatory_protein": "Transcriptional repression",
    "TetR_transcriptional_regulator": "Transcriptional repression",
    "transcriptional_repressor": "Negative regulation of gene expression",
    "similar_to_AmphRI_regulatory_protein": "Regulation of amphoteronolide biosynthesis",
    "putative_citrinin_biosynthesis_transcriptional_activator_CtnR": "Regulation of citrinin biosynthesis",
    
    # Sugar/carbohydrate processing domains
    "glycosyltransferase": "Transfers glycosyl groups to acceptor molecules",
    "NDP-hexose_3,4-isomerase": "Carbohydrate isomerization",
    "putative_dTDP-glucose_4,6-dehydratase": "Dehydration of dTDP-glucose",
    "putative_NDP-glucose_4-epimerase": "Epimerization of NDP-glucose",
    "putative_dNDP-hexose_3-ketoreductase": "Reduction of ketone group in dNDP-hexose",
    "transaldolase": "Carbohydrate metabolism",
    "NDP-4-keto-6-deoxyhexose_reductase": "Reduction in deoxyhexose biosynthesis",
    
    # Biosynthetic domains
    "proline_adenylation_protein": "Activation of proline for peptide synthesis",
    "amide_synthase": "Formation of amide bonds",
    "Ligase": "Catalyzes bond formation",
    "acyl-coA_ligase": "Activation of fatty acids",
    "putative_acyl-coA_ligase": "Activation of fatty acids (predicted)",
    "3-O-acyltransferase": "Transfer of acyl groups",
    "3-amino-5-hydroxybenzoate_synthase": "AHBA biosynthesis for rifamycin",
    
    # Specialized domains
    "Hydrolase": "Catalyzes hydrolysis reactions",
    "versicolorin_B_synthase": "Synthesis of versicolorin B in aflatoxin pathway",
    "VBS": "Versicolorin B synthesis",
    "VerB": "Versicolorin B synthesis enzyme",
    
    # Named domains (likely product-specific)
    "BafX": "Bafilomycin biosynthesis component X",
    "BafY": "Bafilomycin biosynthesis component Y",
    "BafAI": "Bafilomycin biosynthesis component AI",
    "BafAII": "Bafilomycin biosynthesis component AII",
    "BafAIII": "Bafilomycin biosynthesis component AIII",
    "BafAIV": "Bafilomycin biosynthesis component AIV",
    "BafAV": "Bafilomycin biosynthesis component AV",
    "AmbA": "Ambobactin biosynthesis component A",
    "AmbB": "Ambobactin biosynthesis component B",
    "AmbC": "Ambobactin biosynthesis component C",
    "AmbE": "Ambobactin biosynthesis component E",
    "AmbF": "Ambobactin biosynthesis component F",
    "AmbG": "Ambobactin biosynthesis component G",
    "AmbQ": "Ambobactin biosynthesis component Q",
    "Amb4": "Ambobactin biosynthesis component 4",
    "AngAI": "Angiogenin-related component AI",
    "AngAII": "Angiogenin-related component AII",
    "AngAIII": "Angiogenin-related component AIII",
    "AngAIV": "Angiogenin-related component AIV",
    "AngAV": "Angiogenin-related component AV",
    "AviG3": "Avilamycin biosynthesis component G3",
    "AvnA": "Polyene antibiotic biosynthesis component A",
    "Aft9-1": "Antibiotic biosynthesis component 9-1",
    "OrdA": "Aflatoxin biosynthesis oxidoreductase A",
    "CypX": "Cytochrome P450 in aflatoxin biosynthesis",
    "StcC": "Sterigmatocystin biosynthesis component C",
    "HypA": "Hydrogenase maturation protein A"
}

# Check for domains in the CSV that don't have mappings and assign a generic function
for domain in all_domains:
    if domain not in domain_to_function and not domain.startswith("hypothetical"):
        domain_to_function[domain] = f"Function associated with {domain}"

# For hypothetical proteins
domain_to_function["hypothetical_protein"] = "Protein of unknown function"
domain_to_function["unknown"] = "Function not characterized"

# Group by protein to collect all domains
protein_domains = domains_df.groupby('Query_Protein')['Domain'].apply(list).reset_index()

# Create a new dataframe for protein functions
protein_functions = []

for _, row in protein_domains.iterrows():
    protein_id = row['Query_Protein']
    domains = row['Domain']
    
    # Get functions for each domain
    functions = []
    for domain in domains:
        if domain in domain_to_function:
            function = domain_to_function[domain]
            if function not in functions:  # Avoid duplicates
                functions.append(function)
    
    # If no known functions, mark as unknown
    if not functions:
        functions = ["Unknown function"]
    
    # Add to results
    protein_functions.append({
        'Protein_ID': protein_id,
        'Domains': '; '.join(domains),
        'Functions': '; '.join(functions)
    })

# Create dataframe and save to CSV
results_df = pd.DataFrame(protein_functions)
results_df.to_csv("protein_functions.csv", index=False)

# Calculate some statistics
total_proteins = len(results_df)
domains_with_known_functions = sum(1 for domain in all_domains if domain in domain_to_function and domain_to_function[domain] != f"Function associated with {domain}")
percent_known = (domains_with_known_functions / len(all_domains)) * 100 if all_domains else 0

print(f"✅ Protein function mapping completed!")
print(f"✅ Processed {total_proteins} unique proteins with {len(all_domains)} distinct domains")
print(f"✅ Domain annotation coverage: {domains_with_known_functions}/{len(all_domains)} ({percent_known:.1f}%)")
print(f"✅ Results saved as protein_functions.csv")