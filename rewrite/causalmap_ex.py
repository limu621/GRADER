import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

class MedicalCausalSystem:
    def __init__(self):
        # Define medical systems and their subsystems
        self.medical_systems = {
            "Cardiovascular": ["Blood Pressure State", "Heart Rate State", "Perfusion State"],
            "Respiratory": ["Oxygenation State", "Ventilation State", "Respiratory Effort"],
            "Renal": ["Urine Output State", "Kidney Function State", "Electrolyte Balance"],
            "Neurological": ["Consciousness State", "GCS State", "Pupillary Response"],
            "Hepatic": ["Liver Function State", "Bilirubin Level", "Coagulation State"],
            "Metabolic": ["Acid-Base State", "Lactate Level", "Glucose State"]
        }
        
        # Define possible states for each subsystem
        self.subsystem_states = {
            "Blood Pressure State": ["Normal", "Mild Hypotension", "Moderate Hypotension", "Severe Hypotension"],
            "Heart Rate State": ["Normal", "Mild Tachycardia", "Moderate Tachycardia", "Severe Tachycardia"],
            "Perfusion State": ["Normal", "Mildly Reduced", "Moderately Reduced", "Severely Reduced"],
            "Oxygenation State": ["Normal", "Mild Hypoxemia", "Moderate Hypoxemia", "Severe Hypoxemia"],
            "Ventilation State": ["Normal", "Mild Abnormality", "Moderate Abnormality", "Severe Abnormality"],
            "Respiratory Effort": ["Normal", "Mild Increase", "Moderate Increase", "Severe Increase"],
            "Urine Output State": ["Normal", "Oliguria", "Anuria"],
            "Kidney Function State": ["Normal", "Mild Impairment", "Moderate Impairment", "Severe Impairment"],
            "Electrolyte Balance": ["Normal", "Mild Imbalance", "Moderate Imbalance", "Severe Imbalance"],
            "Consciousness State": ["Alert", "Drowsy", "Stuporous", "Comatose"],
            "GCS State": ["Normal", "Mild Decrease", "Moderate Decrease", "Severe Decrease"],
            "Pupillary Response": ["Normal", "Abnormal Unilateral", "Abnormal Bilateral"],
            "Liver Function State": ["Normal", "Mild Impairment", "Moderate Impairment", "Severe Impairment"],
            "Bilirubin Level": ["Normal", "Mildly Elevated", "Moderately Elevated", "Severely Elevated"],
            "Coagulation State": ["Normal", "Mild Abnormality", "Moderate Abnormality", "Severe Abnormality"],
            "Acid-Base State": ["Normal", "Mild Acidosis", "Moderate Acidosis", "Severe Acidosis", "Alkalosis"],
            "Lactate Level": ["Normal", "Mildly Elevated", "Moderately Elevated", "Severely Elevated"],
            "Glucose State": ["Hypoglycemia", "Normal", "Mild Hyperglycemia", "Moderate Hyperglycemia", "Severe Hyperglycemia"]
        }
        
        # Define indicators that map to subsystem states
        self.indicators = {
            "MAP": "Blood Pressure State",
            "SBP": "Blood Pressure State",
            "DBP": "Blood Pressure State",
            "HR": "Heart Rate State",
            "Capillary Refill": "Perfusion State",
            "Lactate": "Perfusion State",
            "SpO2": "Oxygenation State",
            "PaO2": "Oxygenation State",
            "P/F Ratio": "Oxygenation State",
            "PaCO2": "Ventilation State",
            "RR": "Respiratory Effort",
            "UO-4h": "Urine Output State",
            "UO-24h": "Urine Output State",
            "Creatinine": "Kidney Function State",
            "BUN": "Kidney Function State",
            "GCS": "Consciousness State",
            "Pupil-L": "Pupillary Response",
            "Pupil-R": "Pupillary Response",
            "ALT": "Liver Function State",
            "AST": "Liver Function State",
            "Bilirubin": "Bilirubin Level",
            "PT": "Coagulation State",
            "INR": "Coagulation State",
            "pH": "Acid-Base State",
            "BE": "Acid-Base State",
            "HCO3": "Acid-Base State",
            "Glucose": "Glucose State"
        }
        
        # Threshold definitions for mapping raw values to states
        self.thresholds = {
            "MAP": {
                "Normal": 65,
                "Mild Hypotension": 60,
                "Moderate Hypotension": 50,
                # Below 50 is Severe Hypotension
            },
            "UO-4h": {
                "Normal": 0.5,  # ml/kg/h
                "Oliguria": 0.1,
                # Below 0.1 is Anuria
            },
            # Other thresholds would be defined similarly
        }
        
        # Initialize cluster-specific causal graphs
        self.cluster_graphs = {}
        self.initialize_cluster_graphs()
    
    def initialize_cluster_graphs(self):
        """Initialize causal graphs for different patient clusters"""
        # For demonstration, we'll create a specific graph for cluster 142
        # In a real system, these would be learned from data
        
        G = nx.DiGraph()
        
        # Add nodes (subsystem states)
        subsystems_to_include = [
            "Blood Pressure State", "Heart Rate State", "Perfusion State",
            "Oxygenation State", "Ventilation State", "Respiratory Effort",
            "Urine Output State", "Kidney Function State", 
            "Consciousness State", "GCS State"
        ]
        
        for subsystem in subsystems_to_include:
            G.add_node(subsystem)
        
        # Add edges with weights for cluster 142
        # These represent the causal relationships and their strengths
        edges_with_weights = [
            ("Blood Pressure State", "Urine Output State", 0.8),
            ("Blood Pressure State", "Kidney Function State", 0.7),
            ("Blood Pressure State", "Perfusion State", 0.9),
            ("Blood Pressure State", "Consciousness State", 0.6),
            ("Oxygenation State", "Consciousness State", 0.7),
            ("Oxygenation State", "Respiratory Effort", 0.5),
            ("Ventilation State", "Oxygenation State", 0.6),
            ("Heart Rate State", "Perfusion State", 0.4),
            ("Perfusion State", "Kidney Function State", 0.5),
            ("Perfusion State", "Consciousness State", 0.4),
            ("Kidney Function State", "Acid-Base State", 0.3),
            ("Blood Pressure State", "GCS State", 0.5),
            ("Oxygenation State", "GCS State", 0.6)
        ]
        
        for source, target, weight in edges_with_weights:
            G.add_edge(source, target, weight=weight)
        
        # Store the graph
        self.cluster_graphs[142] = G
    
    def map_raw_value_to_state(self, indicator, value):
        """Map a raw indicator value to a subsystem state"""
        if indicator not in self.indicators:
            return "Unknown"
        
        subsystem = self.indicators[indicator]
        
        if indicator == "MAP":
            if value >= self.thresholds[indicator]["Normal"]:
                return "Normal"
            elif value >= self.thresholds[indicator]["Mild Hypotension"]:
                return "Mild Hypotension"
            elif value >= self.thresholds[indicator]["Moderate Hypotension"]:
                return "Moderate Hypotension"
            else:
                return "Severe Hypotension"
        
        elif indicator == "UO-4h":
            if value >= self.thresholds[indicator]["Normal"]:
                return "Normal"
            elif value >= self.thresholds[indicator]["Oliguria"]:
                return "Oliguria"
            else:
                return "Anuria"
        
        # Additional indicators would have similar logic
        
        return "Unknown"  # Default return if not specifically handled
    
    def process_patient_data(self, raw_measurements):
        """Process raw patient measurements into subsystem states"""
        subsystem_states = {}
        
        for indicator, value in raw_measurements.items():
            if indicator in self.indicators:
                subsystem = self.indicators[indicator]
                state = self.map_raw_value_to_state(indicator, value)
                
                # Only update if we got a valid state
                if state != "Unknown":
                    subsystem_states[subsystem] = state
        
        return subsystem_states
    
    def determine_cluster(self, subsystem_states):
        """Determine which cluster a patient belongs to based on their states"""
        # In a real system, this would use a trained clustering model
        # For demonstration, we'll use a simple rule-based approach
        
        # Check if the pattern matches cluster 142
        if (subsystem_states.get("Blood Pressure State") in ["Moderate Hypotension", "Severe Hypotension"] and
            subsystem_states.get("Kidney Function State") in ["Mild Impairment", "Moderate Impairment"]):
            return 142
        
        # Other cluster checks would go here
        
        # Default cluster if no specific match
        return 1
    
    def visualize_causal_graph(self, cluster_id):
        """Visualize the causal graph for a specific cluster"""
        if cluster_id not in self.cluster_graphs:
            print(f"No causal graph available for cluster {cluster_id}")
            return
        
        G = self.cluster_graphs[cluster_id]
        
        # Create a figure with multiple parts
        fig = plt.figure(figsize=(20, 10))
        
        # Part 1: Basic Graph Visualization
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
        
        # Create positions for the nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for line thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        
        # Color nodes by medical system
        node_colors = []
        for node in G.nodes():
            for system, subsystems in self.medical_systems.items():
                if node in subsystems:
                    if system == "Cardiovascular":
                        node_colors.append('red')
                    elif system == "Respiratory":
                        node_colors.append('blue')
                    elif system == "Renal":
                        node_colors.append('green')
                    elif system == "Neurological":
                        node_colors.append('purple')
                    elif system == "Hepatic":
                        node_colors.append('orange')
                    elif system == "Metabolic":
                        node_colors.append('brown')
                    break
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8, ax=ax1)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='gray', arrows=True, arrowsize=20, ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax1)
        
        # Create a legend
        legend_elements = [
            Patch(facecolor='red', label='Cardiovascular'),
            Patch(facecolor='blue', label='Respiratory'),
            Patch(facecolor='green', label='Renal'),
            Patch(facecolor='purple', label='Neurological'),
            Patch(facecolor='orange', label='Hepatic'),
            Patch(facecolor='brown', label='Metabolic')
        ]
        
        ax1.legend(handles=legend_elements, loc='upper right')
        ax1.set_title(f"Causal Graph for Cluster {cluster_id}")
        ax1.axis('off')
        
        # Part 2: Adjacency Matrix Heatmap
        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
        
        # Create adjacency matrix
        nodes = list(G.nodes())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if G.has_edge(source, target):
                    adj_matrix[i, j] = G[source][target]['weight']
        
        # Create a DataFrame for better visualization
        df_adj = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
        
        # Create heatmap
        sns.heatmap(df_adj, cmap='YlOrRd', annot=True, fmt=".1f", linewidths=.5, ax=ax2)
        ax2.set_title(f"Adjacency Matrix for Cluster {cluster_id}")
        
        # Part 3: Show example of mapping raw values to states
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)
        
        # Create example patient
        example_patient = {
            "MAP": 58,  # mmHg
            "HR": 115,  # bpm
            "UO-4h": 0.3,  # ml/kg/h
            "SpO2": 89,  # %
            "GCS": 13
        }
        
        # Process patient data
        states = self.process_patient_data(example_patient)
        
        # Create a description of the processing flow
        description = [
            f"Example Patient Raw Measurements:",
            f"  MAP = {example_patient['MAP']} mmHg",
            f"  HR = {example_patient['HR']} bpm",
            f"  UO-4h = {example_patient['UO-4h']} ml/kg/h",
            f"  SpO2 = {example_patient['SpO2']}%",
            f"  GCS = {example_patient['GCS']}",
            f"\nMapped to Subsystem States:",
            f"  Blood Pressure State = {self.map_raw_value_to_state('MAP', example_patient['MAP'])}",
            f"  Heart Rate State = {self.map_raw_value_to_state('HR', example_patient['HR'])}",
            f"  Urine Output State = {self.map_raw_value_to_state('UO-4h', example_patient['UO-4h'])}",
            f"  Oxygenation State = {self.map_raw_value_to_state('SpO2', example_patient['SpO2'])}",
            f"  GCS State = Normal",  # Would be mapped with proper threshold
            f"\nCluster Determination:",
            f"  Based on these subsystem states, patient belongs to Cluster {self.determine_cluster(states)}",
            f"\nCausal Inference (examples):",
            f"  1. Blood Pressure State → Urine Output State (Edge Weight: 0.8)",
            f"     Inference: Improving blood pressure will likely improve urine output",
            f"  2. Blood Pressure State → Consciousness State (Edge Weight: 0.6)",
            f"     Inference: Improving blood pressure may improve consciousness",
            f"\nTreatment Decision Support:",
            f"  Primary target: Blood Pressure State (central node with high out-degree)",
            f"  Recommended intervention: Vasopressors level 3",
            f"  Expected improvements: Blood Pressure, Urine Output, Consciousness"
        ]
        
        # Add the description to the plot
        ax3.text(0.1, 0.5, "\n".join(description), fontsize=12,
                 verticalalignment='center', horizontalalignment='left',
                 transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig("cluster_142_causal_graph.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    

# Create the medical causal system
medical_system = MedicalCausalSystem()

# Visualize the causal graph for cluster 142
medical_system.visualize_causal_graph(142)