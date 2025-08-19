import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2
from scipy.stats import chi2_contingency
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union
warnings.filterwarnings('ignore')

# Required libraries for Bayesian Networks
try:
    from pgmpy.models import BayesianNetwork 
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.estimators import HillClimbSearch, BicScore, PC
    PGMPY_AVAILABLE = True
    print("pgmpy imported successfully")
except ImportError:
    PGMPY_AVAILABLE = False
    print("Warning: pgmpy not available. Install with: pip install pgmpy")

class ENSINBayesianNetwork:
    """
    Enhanced Bayesian Network for ENSIN Health & Nutrition Data
    
    Features:
    - Layer-wise causal structure based on epidemiological theory
    - Multiple structure learning algorithms
    - Robust preprocessing and validation
    - Comprehensive inference capabilities
    - Model persistence and visualization
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.network = None
        self.fitted_network = None
        self.structure_model = None
        self.edge_scores = {}
        self.encoders = {}
        self.df_processed = None
        
        # Enhanced variable categories following causal epidemiological pathways
        self.variable_layers = {
            'layer_1_demographics': [
                'edad', 'sexo', 'meses', 'aedad', 'asexo'
            ],
            'layer_2_geography_ses': [
                'x_region', 'desc_ent', 'desc_mun', 'entidad', 'municipio', 'estrato'
            ],
            'layer_3_household': [
                'h0406', 'upm', 'est_sel'
            ],
            'layer_4_body_perception': [
                'a0104', 'a0107', 'a0108', 'a0109'
            ],
            'layer_5_mental_health': [
                'a0202', 'a0203', 'a0204', 'a0205', 'a0206', 'a0207',
                'a0211', 'a0212', 'a0213', 'a0214', 'a0215', 'a0216', 'a0217'
            ],
            'layer_6_disease_diagnosis': [
                'a0301', 'a0401', 'a0402a', 'a0405a', 'a0409',
                'a0601a', 'a0603', 'a0604', 'a0606'
            ],
            'layer_7_medical_management': [
                'a0303num', 'a0306e', 'a0406e', 'a0410a', 'a0410b', 'a0410c', 'a0605b'
            ],
            'layer_8_dietary_patterns': [
                'a0701p', 'a0702p', 'a0703p', 'a0701m', 'a0702m', 'a0802', 'a0811d'
            ],
            'layer_9_food_consumption': [
                'a1001a', 'a1001b', 'a1001c', 'a1001f', 'a1001f1'
            ],
            'layer_10_nutritional_status': [
                'a1503', 'a1306t', 'a1307', 'a1308'
            ],
            'layer_11_health_outcomes': [
                'a1210', 'a1301'
            ]
        }
        
        # Variable interpretations for better understanding
        self.variable_meanings = {
            # Demographics
            'edad': 'Age',
            'sexo': 'Sex/Gender',
            
            # Body perception and weight management
            'a0104': 'Body perception/satisfaction',
            'a0107': 'Intentional weight loss attempts',  
            'a0108': 'Intentional weight gain attempts',
            'a0109': 'Unintentional weight changes',
            
            # Mental health indicators
            'a0202': 'Depression symptoms',
            'a0203': 'Anxiety symptoms',
            'a0204': 'Sleep problems',
            'a0205': 'Stress levels',
            'a0206': 'Social support',
            'a0207': 'Life satisfaction',
            
            # Disease diagnosis
            'a0301': 'Diabetes diagnosis',
            'a0401': 'Hypertension diagnosis', 
            'a0604': 'High cholesterol diagnosis',
            
            # Dietary patterns
            'a0701p': 'Fruit consumption frequency',
            'a0702p': 'Vegetable consumption frequency',
            'a0703p': 'Whole grain consumption',
            
            # Nutritional status
            'a1503': 'BMI category/nutritional status',
            
            # Health outcomes
            'a1210': 'Self-reported health status',
            'a1301': 'Quality of life measure'
        }
    
    def get_layer_variables(self, layer_key: str) -> List[str]:
        """Get all variables from a specific layer"""
        return self.variable_layers.get(layer_key, [])
    
    def get_available_layer_variables(self, layer_key: str) -> List[str]:
        """Get variables from a layer that are actually present in the dataset"""
        layer_vars = self.get_layer_variables(layer_key)
        return [var for var in layer_vars if var in self.df.columns]
    
    def preprocess_data(self, discretization_method: str = 'quantile', n_bins: int = 3) -> pd.DataFrame:
        """
        Enhanced preprocessing with multiple discretization options
        """
        print("Preprocessing data...")
        df_processed = self.df.copy()
        
        # Handle missing values first
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    mode_val = df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 'unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
                else:
                    # Fill numeric with median
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
        
        # Encode and discretize variables
        for col in df_processed.columns:
            try:
                if df_processed[col].dtype == 'object':
                    # Categorical encoding
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Numeric discretization
                    if len(df_processed[col].unique()) > n_bins:
                        if discretization_method == 'quantile':
                            df_processed[col], bins = pd.qcut(
                                df_processed[col], 
                                q=n_bins, 
                                labels=False, 
                                retbins=True, 
                                duplicates='drop'
                            )
                        elif discretization_method == 'uniform':
                            df_processed[col], bins = pd.cut(
                                df_processed[col], 
                                bins=n_bins, 
                                labels=False, 
                                retbins=True
                            )
                        
                        # Store binning information
                        self.encoders[f"{col}_bins"] = bins
            
            except Exception as e:
                print(f"Warning: Could not process {col}: {e}")
                # Keep original values if processing fails
                pass
        
        self.df_processed = df_processed
        print(f"Data shape after preprocessing: {df_processed.shape}")
        print(f"Encoded variables: {len(self.encoders)}")
        
        return df_processed
    
    def create_expert_structure(self) -> List[Tuple[str, str]]:
        """
        Create network structure based on epidemiological expert knowledge
        """
        print("Creating expert-based causal structure...")
        
        edges = []
        layer_keys = list(self.variable_layers.keys())
        
        # Get available variables for each layer
        layer_variables = {}
        for layer_key in layer_keys:
            available_vars = self.get_available_layer_variables(layer_key)
            layer_variables[layer_key] = available_vars
            if available_vars:
                print(f"{layer_key}: {len(available_vars)} variables")
        
        # Create causal relationships between layers
        for i in range(len(layer_keys) - 1):
            current_layer = layer_keys[i]
            
            # Connect to immediate next layer and skip connections to important outcomes
            for j in range(i + 1, len(layer_keys)):
                next_layer = layer_keys[j]
                
                current_vars = layer_variables[current_layer]
                next_vars = layer_variables[next_layer]
                
                # Selective connections based on epidemiological theory
                if self._should_connect_layers(current_layer, next_layer):
                    # Connect key variables from current to next layer
                    for current_var in current_vars[:5]:  # Limit connections
                        for next_var in next_vars[:3]:
                            edges.append((current_var, next_var))
        
        # Add specific expert knowledge connections
        expert_connections = [
            ('edad', 'a0301'),     # Age -> Diabetes
            ('edad', 'a0401'),     # Age -> Hypertension  
            ('sexo', 'a1503'),     # Sex -> Nutritional status
            ('estrato', 'a0701p'), # SES -> Fruit consumption
            ('a0301', 'a1210'),    # Diabetes -> Health status
            ('a0401', 'a1210'),    # Hypertension -> Health status
            ('a0701p', 'a1503'),   # Diet -> Nutritional status
            ('a1503', 'a1210'),    # Nutritional status -> Health
        ]
        
        for parent, child in expert_connections:
            if parent in self.df.columns and child in self.df.columns:
                if (parent, child) not in edges:
                    edges.append((parent, child))
        
        print(f"Created expert structure with {len(edges)} edges")
        return edges
    
    def _should_connect_layers(self, layer1: str, layer2: str) -> bool:
        """
        Determine if two layers should be connected based on causal theory
        """
        # Define strong causal relationships
        strong_connections = {
            'layer_1_demographics': ['layer_4_body_perception', 'layer_6_disease_diagnosis', 'layer_11_health_outcomes'],
            'layer_2_geography_ses': ['layer_8_dietary_patterns', 'layer_9_food_consumption'],
            'layer_4_body_perception': ['layer_5_mental_health'],
            'layer_5_mental_health': ['layer_8_dietary_patterns'],
            'layer_6_disease_diagnosis': ['layer_7_medical_management', 'layer_11_health_outcomes'],
            'layer_8_dietary_patterns': ['layer_10_nutritional_status'],
            'layer_10_nutritional_status': ['layer_11_health_outcomes'],
        }
        
        return layer2 in strong_connections.get(layer1, [])
    
    def learn_structure_pc(self) -> List[Tuple[str, str]]:
        """
        Learn network structure using PC algorithm (constraint-based)
        """
        print("Learning structure using PC algorithm...")
        
        if not PGMPY_AVAILABLE:
            print("pgmpy not available, using expert structure")
            return self.create_expert_structure()
        
        try:
            if self.df_processed is None:
                self.preprocess_data()
            
            # Use subset of variables for computational efficiency
            key_vars = []
            for layer in self.variable_layers.values():
                available = [v for v in layer if v in self.df_processed.columns]
                key_vars.extend(available[:3])  # Top 3 from each layer
            
            subset_df = self.df_processed[key_vars].copy()
            
            # Use PC algorithm if available
            if PC is not None:
                pc = PC(subset_df)
                model = pc.estimate(significance_level=0.05)
                edges = list(model.edges())
            else:
                print("PC algorithm not available, using expert structure")
                edges = self.create_expert_structure()
            
            print(f"PC algorithm learned {len(edges)} edges")
            return edges
            
        except Exception as e:
            print(f"Error in PC structure learning: {e}")
            print("Falling back to expert structure")
            return self.create_expert_structure()
    
    def learn_structure_hill_climb(self) -> List[Tuple[str, str]]:
        """
        Learn network structure using Hill Climbing with BIC score
        """
        print("Learning structure using Hill Climbing...")
        
        if not PGMPY_AVAILABLE or HillClimbSearch is None:
            print("Hill Climbing not available, using expert structure")
            return self.create_expert_structure()
        
        try:
            if self.df_processed is None:
                self.preprocess_data()
            
            # Use subset for efficiency
            key_vars = []
            for layer in self.variable_layers.values():
                available = [v for v in layer if v in self.df_processed.columns]
                key_vars.extend(available[:2])  # Top 2 from each layer
            
            subset_df = self.df_processed[key_vars].copy()
            
            scoring_method = BicScore(subset_df)
            hc = HillClimbSearch(subset_df)
            
            # Start with expert structure as initial model
            expert_edges = self.create_expert_structure()
            expert_edges_filtered = [(p, c) for p, c in expert_edges if p in key_vars and c in key_vars]
            
            if expert_edges_filtered:
                start_model = BayesianNetwork(expert_edges_filtered[:10])  # Limit initial edges
                model = hc.estimate(scoring_method=scoring_method, start_dag=start_model)
            else:
                model = hc.estimate(scoring_method=scoring_method)
            
            edges = list(model.edges())
            print(f"Hill Climbing learned {len(edges)} edges")
            return edges
            
        except Exception as e:
            print(f"Error in Hill Climbing structure learning: {e}")
            return self.create_expert_structure()
    
    def create_network(self, method: str = 'expert') -> Union[BayesianNetwork, nx.DiGraph]:
        """
        Create Bayesian Network using specified structure learning method
        """
        print(f"Creating network using {method} method...")
        
        if method == 'expert':
            edges = self.create_expert_structure()
        elif method == 'pc':
            edges = self.learn_structure_pc()
        elif method == 'hill_climb':
            edges = self.learn_structure_hill_climb()
        else:
            print(f"Unknown method {method}, using expert")
            edges = self.create_expert_structure()
        
        # Create network from edges
        if PGMPY_AVAILABLE:
            try:
                self.network = BayesianNetwork(edges)
                
                # Check for cycles and remove if necessary
                if not nx.is_directed_acyclic_graph(self.network):
                    print("Removing cycles from network...")
                    dag_edges = self._remove_cycles(edges)
                    self.network = BayesianNetwork(dag_edges)
                
                print(f"Created pgmpy BayesianNetwork with {len(self.network.nodes())} nodes and {len(self.network.edges())} edges")
                
            except Exception as e:
                print(f"Error creating pgmpy network: {e}")
                # Fallback to NetworkX
                self.network = nx.DiGraph()
                self.network.add_edges_from(edges)
                if not nx.is_directed_acyclic_graph(self.network):
                    dag_edges = self._remove_cycles(edges)
                    self.network = nx.DiGraph()
                    self.network.add_edges_from(dag_edges)
        else:
            # Use NetworkX
            self.network = nx.DiGraph()
            self.network.add_edges_from(edges)
            if not nx.is_directed_acyclic_graph(self.network):
                dag_edges = self._remove_cycles(edges)
                self.network = nx.DiGraph()
                self.network.add_edges_from(dag_edges)
            
            print(f"Created NetworkX DiGraph with {len(self.network.nodes())} nodes and {len(self.network.edges())} edges")
        
        return self.network
    
    def _remove_cycles(self, edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Remove cycles from edge list to ensure DAG property
        """
        dag_edges = []
        temp_graph = nx.DiGraph()
        
        for edge in edges:
            temp_graph.add_edge(*edge)
            if nx.is_directed_acyclic_graph(temp_graph):
                dag_edges.append(edge)
            else:
                temp_graph.remove_edge(*edge)
        
        print(f"Removed {len(edges) - len(dag_edges)} edges to ensure DAG property")
        return dag_edges
    
    def fit_parameters(self, method: str = 'mle') -> Optional[BayesianNetwork]:
        """
        Fit network parameters using Maximum Likelihood or Bayesian estimation
        """
        print(f"Fitting network parameters using {method}...")
        
        if not PGMPY_AVAILABLE:
            print("pgmpy not available - cannot fit parameters")
            return None
        
        if not isinstance(self.network, BayesianNetwork):
            print("Network is not a pgmpy BayesianNetwork - cannot fit parameters")
            return None
        
        try:
            if self.df_processed is None:
                self.preprocess_data()
            
            network_vars = list(self.network.nodes())
            subset_df = self.df_processed[network_vars].copy()
            
            # Ensure all variables are properly categorized
            for var in network_vars:
                if subset_df[var].dtype not in ['category', 'int64']:
                    subset_df[var] = subset_df[var].astype('category')
            
            # Choose estimator
            if method == 'mle':
                estimator = MaximumLikelihoodEstimator(self.network, subset_df)
            else:
                estimator = BayesianEstimator(self.network, subset_df)
            
            self.fitted_network = self.network.copy()
            
            # Fit CPDs for each node
            successful_fits = 0
            for node in self.network.nodes():
                try:
                    cpd = estimator.estimate_cpd(node)
                    self.fitted_network.add_cpds(cpd)
                    successful_fits += 1
                except Exception as e:
                    print(f"Warning: Could not fit CPD for node {node}: {e}")
            
            print(f"Successfully fitted CPDs for {successful_fits}/{len(self.network.nodes())} nodes")
            
            # Validate model
            try:
                self.fitted_network.check_model()
                print("Network validation successful!")
            except Exception as e:
                print(f"Network validation warning: {e}")
            
            return self.fitted_network
            
        except Exception as e:
            print(f"Error fitting parameters: {e}")
            return None
    
    def perform_inference(self, evidence: Dict[str, Union[str, int]], 
                         query_variables: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Perform probabilistic inference on the fitted network
        """
        if not PGMPY_AVAILABLE or self.fitted_network is None:
            print("Cannot perform inference: network not fitted or pgmpy unavailable")
            return None
        
        try:
            inference_engine = VariableElimination(self.fitted_network)
            
            if query_variables is None:
                # Default query variables
                query_variables = ['a1503', 'a1210', 'a0301']  # Nutrition, health, diabetes
                query_variables = [var for var in query_variables if var in self.fitted_network.nodes()]
            
            if not query_variables:
                print("No valid query variables found")
                return None
            
            # Encode evidence if necessary
            encoded_evidence = {}
            for var, value in evidence.items():
                if var in self.encoders:
                    if isinstance(value, str):
                        try:
                            encoded_evidence[var] = self.encoders[var].transform([value])[0]
                        except:
                            print(f"Warning: Could not encode {var}={value}")
                            encoded_evidence[var] = value
                    else:
                        encoded_evidence[var] = value
                else:
                    encoded_evidence[var] = value
            
            # Perform inference
            result = inference_engine.query(variables=query_variables, evidence=encoded_evidence)
            
            print(f"\nInference Results for evidence {evidence}:")
            print("-" * 50)
            print(result)
            
            return {'result': result, 'evidence': encoded_evidence, 'query_vars': query_variables}
            
        except Exception as e:
            print(f"Error in inference: {e}")
            return None
    
    def visualize_network(self, figsize: Tuple[int, int] = (16, 12), 
                         layout: str = 'hierarchical') -> None:
        """
        Visualize the network with improved layout and styling
        """
        if self.network is None:
            print("No network to visualize")
            return
        
        plt.figure(figsize=figsize)
        
        # Create position layout
        if layout == 'hierarchical':
            pos = self._create_hierarchical_layout()
        elif layout == 'spring':
            pos = nx.spring_layout(self.network, k=3, iterations=50)
        else:
            pos = nx.circular_layout(self.network)
        
        # Node colors based on layers
        node_colors = self._get_node_colors()
        
        # Draw network
        nx.draw(self.network, pos,
                node_color=[node_colors.get(node, 'lightgray') for node in self.network.nodes()],
                node_size=1000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='darkgray',
                with_labels=True,
                alpha=0.8)
        
        plt.title("ENSIN Bayesian Network Structure", fontsize=16, fontweight='bold')
        
        # Add legend for layers
        self._add_layer_legend()
        
        plt.tight_layout()
        plt.savefig('ensin_network.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print network statistics
        self._print_network_stats()
    
    def _create_hierarchical_layout(self) -> Dict[str, np.ndarray]:
        """Create hierarchical layout based on variable layers"""
        pos = {}
        layer_keys = list(self.variable_layers.keys())
        
        for i, layer_key in enumerate(layer_keys):
            layer_nodes = [node for node in self.network.nodes() 
                          if self._get_node_layer(node) == layer_key]
            
            if layer_nodes:
                x_positions = np.linspace(-3, 3, len(layer_nodes))
                y_position = -i * 1.5
                
                for j, node in enumerate(layer_nodes):
                    pos[node] = np.array([x_positions[j], y_position])
        
        return pos
    
    def _get_node_layer(self, node: str) -> str:
        """Get the layer of a node"""
        for layer_key, layer_vars in self.variable_layers.items():
            if node in layer_vars:
                return layer_key
        return 'unknown'
    
    def _get_node_colors(self) -> Dict[str, str]:
        """Get color mapping for nodes based on layers"""
        colors = {
            'layer_1_demographics': '#FF6B6B',
            'layer_2_geography_ses': '#4ECDC4', 
            'layer_3_household': '#45B7D1',
            'layer_4_body_perception': '#96CEB4',
            'layer_5_mental_health': '#FFEAA7',
            'layer_6_disease_diagnosis': '#DDA0DD',
            'layer_7_medical_management': '#98D8C8',
            'layer_8_dietary_patterns': '#F7DC6F',
            'layer_9_food_consumption': '#BB8FCE',
            'layer_10_nutritional_status': '#F1948A',
            'layer_11_health_outcomes': '#85C1E9'
        }
        
        node_colors = {}
        for node in self.network.nodes():
            layer = self._get_node_layer(node)
            node_colors[node] = colors.get(layer, 'lightgray')
        
        return node_colors
    
    def _add_layer_legend(self) -> None:
        """Add legend for layer colors"""
        colors = self._get_node_colors()
        unique_layers = set()
        
        for node in self.network.nodes():
            layer = self._get_node_layer(node)
            unique_layers.add(layer)
        
        legend_elements = []
        for layer in sorted(unique_layers):
            if layer in colors:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=colors[layer], markersize=10, 
                              label=layer.replace('_', ' ').title())
                )
        
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def _print_network_stats(self) -> None:
        """Print comprehensive network statistics"""
        print(f"\n{'='*50}")
        print("NETWORK STATISTICS")
        print(f"{'='*50}")
        print(f"Nodes: {len(self.network.nodes())}")
        print(f"Edges: {len(self.network.edges())}")
        print(f"Density: {nx.density(self.network):.3f}")
        
        if isinstance(self.network, nx.DiGraph):
            print(f"Is DAG: {nx.is_directed_acyclic_graph(self.network)}")
            
            # Node statistics
            in_degrees = dict(self.network.in_degree())
            out_degrees = dict(self.network.out_degree())
            
            print(f"Average in-degree: {np.mean(list(in_degrees.values())):.2f}")
            print(f"Average out-degree: {np.mean(list(out_degrees.values())):.2f}")
            
            # Top connected nodes
            print(f"\nTop 5 nodes by total degree:")
            total_degrees = {node: in_degrees[node] + out_degrees[node] 
                            for node in self.network.nodes()}
            top_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for node, degree in top_nodes:
                meaning = self.variable_meanings.get(node, 'Unknown variable')
                print(f"  {node} ({meaning}): {degree} connections")
    
    def save_model(self, filepath: str) -> None:
        """Save the complete model to disk"""
        model_data = {
            'network_edges': list(self.network.edges()) if self.network else [],
            'encoders': self.encoders,
            'variable_layers': self.variable_layers,
            'variable_meanings': self.variable_meanings,
        }
        
        # Save fitted network separately if available
        if self.fitted_network and PGMPY_AVAILABLE:
            try:
                cpds_data = []
                for cpd in self.fitted_network.get_cpds():
                    cpds_data.append({
                        'variable': cpd.variable,
                        'variable_card': cpd.variable_card,
                        'values': cpd.get_values().tolist(),
                        'evidence': cpd.evidence,
                        'evidence_card': cpd.evidence_card
                    })
                model_data['cpds'] = cpds_data
            except Exception as e:
                print(f"Warning: Could not save CPDs: {e}")
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4, default=str)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a complete model from disk"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Restore basic components
        self.encoders = model_data.get('encoders', {})
        self.variable_layers = model_data.get('variable_layers', {})
        self.variable_meanings = model_data.get('variable_meanings', {})
        
        # Restore network structure
        edges = model_data.get('network_edges', [])
        if edges and PGMPY_AVAILABLE:
            try:
                self.network = BayesianNetwork(edges)
                
                # Restore CPDs if available
                if 'cpds' in model_data:
                    for cpd_data in model_data['cpds']:
                        cpd = TabularCPD(
                            variable=cpd_data['variable'],
                            variable_card=cpd_data['variable_card'],
                            values=cpd_data['values'],
                            evidence=cpd_data['evidence'],
                            evidence_card=cpd_data['evidence_card']
                        )
                        self.network.add_cpds(cpd)
                    
                    self.fitted_network = self.network
                    print("Loaded fitted network with CPDs")
                else:
                    print("Loaded network structure only")
                    
            except Exception as e: