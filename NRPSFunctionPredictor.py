import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from collections import Counter, defaultdict

class NRPSFunctionPredictor:
    def __init__(self):
        # Dictionary to store domain-function mappings
        self.domain_function_map = defaultdict(list)
        # Dictionary to store sequence-domain mappings
        self.seq_domain_map = {}
        # Graph for network analysis
        self.sequence_graph = nx.Graph()
        
    def load_data(self, domain_function_file, sequence_domain_file):
        """
        Load domain-function mappings and sequence-domain mappings from files
        """
        # Debug printing to see actual columns
        domain_functions = pd.read_csv(domain_function_file)
        st.write("Domain function columns:", domain_functions.columns.tolist())
        
        # Load domain-function mappings - updated to use your column names
        for _, row in domain_functions.iterrows():
            self.domain_function_map[row['Domains']].append(row['Functions'])
        
        # Debug printing
        seq_domains = pd.read_csv(sequence_domain_file)
        st.write("Sequence domain columns:", seq_domains.columns.tolist())
        
        # Load sequence-domain mappings - updated to use your column names
        for _, row in seq_domains.iterrows():
            # Check if 'Domain' exists or use 'Query_Protein' and 'Domain'
            if 'Domain' in seq_domains.columns:
                sequence_id_col = 'sequence_id' if 'sequence_id' in seq_domains.columns else 'Query_Protein'
                self.seq_domain_map[row[sequence_id_col]] = [row['Domain']]
            else:
                # Handle the case where domains might be comma-separated in one column
                self.seq_domain_map[row['Query_Protein']] = [row['Domain']]
    
    def build_sequence_graph(self, similarity_file):
        """
        Build a graph where nodes are sequences and edges represent similarity scores
        """
        similarity_data = pd.read_csv(similarity_file)
        
        # Debug printing
        st.write("Similarity data columns:", similarity_data.columns.tolist())
        
        # Check column names and adapt
        if 'sequence1' in similarity_data.columns and 'sequence2' in similarity_data.columns:
            seq1_col = 'sequence1'
            seq2_col = 'sequence2'
        else:
            # Get the first two columns if names don't match
            columns = similarity_data.columns.tolist()
            seq1_col = columns[0]
            seq2_col = columns[1]
            
        if 'similarity_score' in similarity_data.columns:
            score_col = 'similarity_score'
        else:
            # Assume third column is the score
            score_col = similarity_data.columns[2]
        
        for _, row in similarity_data.iterrows():
            seq1 = str(row[seq1_col])
            seq2 = str(row[seq2_col])
            similarity_score = float(row[score_col])
            
            # Add nodes if they don't exist
            if seq1 not in self.sequence_graph:
                self.sequence_graph.add_node(seq1)
            if seq2 not in self.sequence_graph:
                self.sequence_graph.add_node(seq2)
            
            # Add edge with similarity score as weight
            self.sequence_graph.add_edge(seq1, seq2, weight=similarity_score)
    
    def smith_waterman(self, seq1, seq2, match=2, mismatch=-1, gap_open=-2, gap_extend=-1):
        """
        Perform Smith-Waterman local alignment between two sequences
        Returns alignment score and aligned sequences
        """
        alignments = PairwiseAligner.align.localms(seq1, seq2, match, mismatch, gap_open, gap_extend)
        if alignments:
            best_alignment = alignments[0]
            return {
                'score': best_alignment.score,
                'seq1_aligned': best_alignment.seqA,
                'seq2_aligned': best_alignment.seqB,
                'identity': self.calculate_identity(best_alignment.seqA, best_alignment.seqB)
            }
        return {'score': 0, 'seq1_aligned': '', 'seq2_aligned': '', 'identity': 0}
    
    def calculate_identity(self, seq1, seq2):
        """
        Calculate the percentage identity between two aligned sequences
        """
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and b != '-')
        total = sum(1 for a, b in zip(seq1, seq2) if a != '-' or b != '-')
        return (matches / total * 100) if total > 0 else 0
    
    def predict_domains_greedy(self, unknown_seq, reference_sequences, threshold=60):
        """
        Greedy algorithm to predict domains for an unknown sequence
        1. Align unknown seq with all reference sequences
        2. Select top matches based on similarity score
        3. Assign domains from the most similar sequences
        """
        alignment_results = []
        
        # Step 1: Align with all reference sequences
        for ref_id, ref_seq in reference_sequences.items():
            alignment = self.smith_waterman(unknown_seq, ref_seq)
            alignment['ref_id'] = ref_id
            alignment_results.append(alignment)
        
        # Step 2: Sort by score and filter by threshold
        alignment_results.sort(key=lambda x: x['score'], reverse=True)
        top_matches = [a for a in alignment_results if a['identity'] >= threshold]
        
        if not top_matches:
            return None, alignment_results[0] if alignment_results else None
        
        # Step 3: Greedy domain assignment
        predicted_domains = []
        covered_regions = []
        
        # Start with the highest scoring matches
        for match in top_matches:
            ref_id = match['ref_id']
            
            # Get domains from the reference sequence
            if ref_id in self.seq_domain_map:
                for domain in self.seq_domain_map[ref_id]:
                    if domain not in predicted_domains:
                        predicted_domains.append(domain)
        
        return predicted_domains, top_matches[0]
    
    def predict_function_from_domains(self, domains):
        """
        Predict functions based on the identified domains using a greedy approach
        """
        if not domains:
            return []
        
        # Collect all possible functions for the identified domains
        function_votes = Counter()
        
        for domain in domains:
            if domain in self.domain_function_map:
                for function in self.domain_function_map[domain]:
                    function_votes[function] += 1
        
        # Sort by vote count
        predicted_functions = [func for func, count in function_votes.most_common()]
        
        return predicted_functions
    
    def predict_function(self, unknown_seq, reference_sequences, threshold=60):
        """
        Predict function for an unknown sequence
        """
        # Step 1: Predict domains using greedy approach
        predicted_domains, best_match = self.predict_domains_greedy(unknown_seq, reference_sequences, threshold)
        
        # Step 2: If domains found, predict functions
        if predicted_domains:
            predicted_functions = self.predict_function_from_domains(predicted_domains)
            return {
                'predicted_domains': predicted_domains,
                'predicted_functions': predicted_functions,
                'best_match': best_match
            }
        else:
            return {
                'predicted_domains': [],
                'predicted_functions': [],
                'best_match': best_match
            }
    
    def save_model(self, filename):
        """
        Save the model to a file
        """
        model_data = {
            'domain_function_map': dict(self.domain_function_map),
            'seq_domain_map': self.seq_domain_map,
            'sequence_graph': nx.to_dict_of_dicts(self.sequence_graph)
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filename):
        """
        Load the model from a file
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.domain_function_map = defaultdict(list, model_data['domain_function_map'])
        model.seq_domain_map = model_data['seq_domain_map']
        model.sequence_graph = nx.from_dict_of_dicts(model_data['sequence_graph'])
        
        return model
    
    def plot_network(self, highlighted_node=None):
        """
        Plot the sequence network graph
        Optionally highlight a specific node
        """
        plt.figure(figsize=(12, 8))
        
        # Create positions for nodes
        pos = nx.spring_layout(self.sequence_graph, seed=42)
        
        # Get edge weights for thickness
        edge_weights = [self.sequence_graph[u][v]['weight'] * 3 for u, v in self.sequence_graph.edges()]
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(self.sequence_graph, pos, node_size=100, alpha=0.8)
        
        # If a node is highlighted, draw it differently
        if highlighted_node and highlighted_node in self.sequence_graph:
            nx.draw_networkx_nodes(self.sequence_graph, pos, nodelist=[highlighted_node],
                                 node_size=300, node_color='red', alpha=1.0)
        
        nx.draw_networkx_edges(self.sequence_graph, pos, width=edge_weights, alpha=0.5)
        
        # Draw labels with smaller font
        nx.draw_networkx_labels(self.sequence_graph, pos, font_size=8)
        
        plt.title("NRPS Sequence Similarity Network")
        plt.axis('off')
        
        return plt.gcf()


# Example usage to train the model:
def train_predictor(domain_function_file, sequence_domain_file, similarity_file, output_model_file):
    predictor = NRPSFunctionPredictor()
    predictor.load_data(domain_function_file, sequence_domain_file)
    predictor.build_sequence_graph(similarity_file)
    predictor.save_model(output_model_file)
    return predictor