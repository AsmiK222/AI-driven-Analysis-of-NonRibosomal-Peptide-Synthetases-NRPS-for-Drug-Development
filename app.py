import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from io import StringIO
import os
import pickle
import numpy as np
import traceback
def get_colored_alignment(seq1, seq2):
        html = "<pre style='font-family: monospace; font-size: 14px;'>"
        for a, b in zip(seq1, seq2):
            if a == b and a != "-":
                html += f"<span style='color: lime;'>{a}</span>"
            elif a == "-" or b == "-":
                html += f"<span style='color: gray;'>{a}</span>"
            else:
                html += f"<span style='color: red;'>{a}</span>"
        html += "<br>"
        for a, b in zip(seq1, seq2):
            if a == b and a != "-":
                html += f"<span style='color: lime;'>{b}</span>"
            elif a == "-" or b == "-":
                html += f"<span style='color: gray;'>{b}</span>"
            else:
                html += f"<span style='color: red;'>{b}</span>"
        html += "</pre>"
        return html
# Implementing the NRPSFunctionPredictor class with improved error handling
class NRPSFunctionPredictor:
    def __init__(self):
        self.domain_function_map = {}
        self.sequence_domain_map = {}
    
    def load_data(self, domain_function_file, sequence_domain_file):
        try:
            # Load domain-function mapping with more robust error handling
            df_domain = pd.read_csv(domain_function_file)
            
            # Check if required columns exist
            required_columns = ['domain_id', 'function']
            for col in required_columns:
                if col not in df_domain.columns:
                    raise ValueError(f"Required column '{col}' not found in domain-function file. Available columns: {list(df_domain.columns)}")
            
            # Load the mapping
            for _, row in df_domain.iterrows():
                self.domain_function_map[row['domain_id']] = row['function']
            
            # Load sequence-domain mapping
            df_seq = pd.read_csv(sequence_domain_file)
            
            # Check if required columns exist
            required_columns = ['sequence_id', 'domain_id']
            for col in required_columns:
                if col not in df_seq.columns:
                    raise ValueError(f"Required column '{col}' not found in sequence-domain file. Available columns: {list(df_seq.columns)}")
            
            # Load the mapping
            for _, row in df_seq.iterrows():
                if row['sequence_id'] not in self.sequence_domain_map:
                    self.sequence_domain_map[row['sequence_id']] = []
                self.sequence_domain_map[row['sequence_id']].append(row['domain_id'])
                
            return True
        except Exception as e:
            st.error(f"Error in load_data: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    def predict_function(self, unknown_seq, ref_sequences, threshold=60):
        results = {
            'predicted_domains': [],
            'predicted_functions': [],
            'best_match': None,
            'alignment_score': 0,
            'identity': 0
        }

        best_score = 0
        best_match = None
        best_identity = 0
        best_alignment = None

        for ref_id, ref_seq in ref_sequences.items():
            alignments = pairwise2.align.globalxx(unknown_seq, ref_seq, one_alignment_only=True)
            if alignments:
                alignment = alignments[0]
                score = alignment.score
                identity = (score / max(len(unknown_seq), len(ref_seq))) * 100

                if score > best_score:
                    best_score = score
                    best_identity = identity
                    best_match = ref_id
                    best_alignment = alignment

        if best_match:
            results['best_match'] = {
                'ref_id': best_match,
                'score': best_score,
                'identity': best_identity,
                'seq1_aligned': best_alignment.seqA,
                'seq2_aligned': best_alignment.seqB
            }
            results['alignment_score'] = best_score
            results['identity'] = best_identity

            if best_match in self.sequence_domain_map:
                results['predicted_domains'] = self.sequence_domain_map[best_match]
                for domain in self.sequence_domain_map[best_match]:
                    if domain in self.domain_function_map:
                        results['predicted_functions'].append(self.domain_function_map[domain])

        return results

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    def get_colored_alignment(seq1, seq2):
        html = "<pre style='font-family: monospace; font-size: 14px;'>"
        for a, b in zip(seq1, seq2):
            if a == b and a != "-":
                html += f"<span style='color: lime;'>{a}</span>"
            elif a == "-" or b == "-":
                html += f"<span style='color: gray;'>{a}</span>"
            else:
                html += f"<span style='color: red;'>{a}</span>"
        html += "<br>"
        for a, b in zip(seq1, seq2):
            if a == b and a != "-":
                html += f"<span style='color: lime;'>{b}</span>"
            elif a == "-" or b == "-":
                html += f"<span style='color: gray;'>{b}</span>"
            else:
                html += f"<span style='color: red;'>{b}</span>"
        html += "</pre>"
        return html

def main():
    st.title("AI-Driven Analysis for NRPS in Drug Development")
    st.write("Upload or paste an unknown NRPS sequence to predict its function")

    # Debug section in sidebar to help troubleshoot CSV issues
    with st.sidebar.expander("Debug Options"):
        show_csv_preview = st.checkbox("Show CSV File Previews")

    # Load model or train if not available
    model_file = "nrps_predictor_model.pkl"
    predictor = None
    
    try:
        if os.path.exists(model_file):
            predictor = NRPSFunctionPredictor.load_model(model_file)
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.warning("Model needs to be trained first!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    alignment_threshold = st.sidebar.slider("Alignment Threshold (%)", 40, 100, 60)
    
    # Option to train a new model
    with st.sidebar.expander("Train New Model"):
        domain_function_file = st.file_uploader("Upload Domain-Function Mapping (CSV)", type=['csv'], key="domain_func")
        sequence_domain_file = st.file_uploader("Upload Sequence-Domain Mapping (CSV)", type=['csv'], key="seq_domain")
        
        # Show previews of uploaded files if debug option is checked
        if show_csv_preview:
            if domain_function_file:
                st.sidebar.write("Domain-Function File Preview:")
                df = pd.read_csv(domain_function_file)
                st.sidebar.dataframe(df.head(3))
                st.sidebar.write(f"Columns: {list(df.columns)}")
                domain_function_file.seek(0)  # Reset file pointer
                
            if sequence_domain_file:
                st.sidebar.write("Sequence-Domain File Preview:")
                df = pd.read_csv(sequence_domain_file)
                st.sidebar.dataframe(df.head(3))
                st.sidebar.write(f"Columns: {list(df.columns)}")
                sequence_domain_file.seek(0)  # Reset file pointer
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train Model") and domain_function_file and sequence_domain_file:
                try:
                    with st.spinner("Training model..."):
                        predictor = NRPSFunctionPredictor()
                        
                        # Step 1: Load domain-function data
                        st.write("Step 1: Loading domain-function mapping...")
                        if not predictor.load_data(domain_function_file, sequence_domain_file):
                            st.error("Failed to load data files.")
                            st.stop()
                        else:
                            st.success("Data loaded successfully!")
                        
                        # Step 3: Save model
                        st.write("Step 2: Saving model...")
                        predictor.save_model(model_file)
                        st.success("Model trained and saved successfully!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.error(traceback.format_exc())
        
        with col2:
            if st.button("Upload Sample Files"):
                st.info("This would download sample CSV files with the correct format.")
                st.code("""
# Sample domain_function.csv format:
domain_id,function
domain1,antibiotic
domain2,enzyme
domain3,inhibitor

# Sample sequence_domain.csv format:
sequence_id,domain_id
seq1,domain1
seq1,domain2
seq2,domain3
                """)

    # Input options
    input_method = st.radio("Select input method:", ["Paste Sequence", "Upload FASTA"])
    
    unknown_seq = ""
    if input_method == "Paste Sequence":
        unknown_seq = st.text_area("Enter your unknown NRPS sequence:", height=150)
    else:
        uploaded_file = st.file_uploader("Upload FASTA file", type=['fasta', 'fa'])
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                for record in SeqIO.parse(StringIO(content), "fasta"):
                    unknown_seq = str(record.seq)
                    st.success(f"Loaded sequence: {record.id} ({len(unknown_seq)} bases)")
                    break  # Only take first sequence
            except Exception as e:
                st.error(f"Error reading FASTA file: {str(e)}")
    
    # Reference sequences
    ref_seq_file = st.file_uploader("Upload Reference Sequences (FASTA or CSV)", type=['fasta', 'fa', 'csv'])
    ref_sequences = {}
    
    if ref_seq_file:
        try:
            if ref_seq_file.name.endswith(('.fasta', '.fa')):
                content = ref_seq_file.read().decode('utf-8')
                for record in SeqIO.parse(StringIO(content), "fasta"):
                    ref_sequences[record.id] = str(record.seq)
            else:  # CSV
                df = pd.read_csv(ref_seq_file)
                
                # Check if required columns exist
                if 'sequence_id' not in df.columns or 'sequence' not in df.columns:
                    available_cols = list(df.columns)
                    st.warning(f"Required columns 'sequence_id' and 'sequence' not found. Available columns: {available_cols}")
                    
                    # Try to guess columns
                    id_col = next((col for col in df.columns if 'id' in col.lower()), df.columns[0])
                    seq_col = next((col for col in df.columns if 'seq' in col.lower()), df.columns[1] if len(df.columns) > 1 else None)
                    
                    if seq_col:
                        st.info(f"Using '{id_col}' as sequence ID and '{seq_col}' as sequence")
                        for _, row in df.iterrows():
                            ref_sequences[str(row[id_col])] = str(row[seq_col])
                else:
                    for _, row in df.iterrows():
                        ref_sequences[row['sequence_id']] = row['sequence']
                        
            st.success(f"Loaded {len(ref_sequences)} reference sequences")
        except Exception as e:
            st.error(f"Error reading reference sequences: {str(e)}")
            st.error(traceback.format_exc())
    
    # Process button
    if st.button("Predict Function"):
        # Validate inputs before proceeding
        if not unknown_seq:
            st.error("Please input an unknown sequence first")
        elif not ref_sequences:
            st.error("Please upload reference sequences")
        elif not predictor:
            st.error("Model needs to be trained or loaded first")
        else:
            try:
                with st.spinner("Analyzing sequence..."):
                    # Predict function
                    result = predictor.predict_function(unknown_seq, ref_sequences, threshold=alignment_threshold)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Predicted Domains:**")
                        if result['predicted_domains']:
                            for domain in result['predicted_domains']:
                                st.write(f"- {domain}")
                        else:
                            st.write("No domains predicted")
                        if result['best_match']:
                            st.markdown("### Most Similar Known Sequence")
                            st.write("Reference ID:", result['best_match']['ref_id'])
                            st.write(f"Similarity Score: {result['alignment_score']:.2f}")
                            st.write(f"Identity: {result['identity']:.2f}%")
                            st.markdown("#### Sequence Alignment (snippet)")
                            st.text("Query : " + result['best_match']['seq1_aligned'][:100])
                            st.text("Match : " + result['best_match']['seq2_aligned'][:100])
                            seq1 = result['best_match']['seq1_aligned'][:100]  # or remove [:100] for full alignment
                            seq2 = result['best_match']['seq2_aligned'][:100]
                            colored_html = get_colored_alignment(seq1, seq2)

                            st.markdown("#### Sequence Alignment (colored snippet)")
                            st.markdown(colored_html, unsafe_allow_html=True)


                    with col2:
                        st.write("**Predicted Functions:**")
                        if result['predicted_functions']:
                            for function in result['predicted_functions']:
                                st.write(f"- {function}")
                        else:
                            st.write("No functions predicted")
                    
                    # Show best match details
                    if result['best_match']:
                        st.subheader("Best Match Details")
                        best = result['best_match']
                        st.write(f"Reference ID: {best['ref_id']}")
                        st.write(f"Alignment Score: {best['score']}")
                        st.write(f"Sequence Identity: {best['identity']:.2f}%")
                        
                        # Display alignment (truncate if too long)
                        st.write("**Sequence Alignment:**")
                        max_display_len = 100  # Maximum characters to display
                        
                        if len(best['seq1_aligned']) > max_display_len:
                            st.text(f"Query:     {best['seq1_aligned'][:max_display_len]}...")
                            st.text(f"Reference: {best['seq2_aligned'][:max_display_len]}...")
                            st.info(f"Alignment truncated (showing first {max_display_len} of {len(best['seq1_aligned'])} characters)")
                        else:
                            st.text(f"Query:     {best['seq1_aligned']}")
                            st.text(f"Reference: {best['seq2_aligned']}")
                    else:
                        st.warning("No significant matches found above the threshold")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()