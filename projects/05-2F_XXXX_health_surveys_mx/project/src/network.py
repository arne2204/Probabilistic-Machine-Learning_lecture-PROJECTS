import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
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
        self.data_encoded_once = False  # Flag to prevent double encoding
        
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
        FIXED: Enhanced preprocessing - prevents double encoding
        """
        print("Preprocessing data...")
        
        # Prevent double encoding - this was the main issue
        if self.data_encoded_once and self.df_processed is not None:
            print("Data already preprocessed, returning existing version")
            return self.df_processed
        
        df_processed = self.df.copy()
        
        # Handle missing values first
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['object', 'category']:
                    mode_val = df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 'unknown'
                    df_processed[col].fillna(mode_val, inplace=True)
                else:
                    median_val = df_processed[col].median()
                    df_processed[col].fillna(median_val, inplace=True)
        
        # FIXED: Better encoding logic
        for col in df_processed.columns:
            try:
                if df_processed[col].dtype == 'object':
                    # Categorical encoding
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Check if already discrete/categorical
                    unique_vals = df_processed[col].nunique()
                    
                    # Only discretize if many unique values AND they're not already integer codes
                    if unique_vals > n_bins and unique_vals > 10 and not (df_processed[col].dtype == 'int64' and df_processed[col].min() >= 0 and df_processed[col].max() < 10):
                        if discretization_method == 'quantile':
                            try:
                                df_processed[col], bins = pd.qcut(
                                    df_processed[col], q=n_bins, labels=False, retbins=True, duplicates='drop'
                                )
                                self.encoders[f"{col}_bins"] = bins
                            except ValueError:
                                df_processed[col], bins = pd.cut(
                                    df_processed[col], bins=n_bins, labels=False, retbins=True
                                )
                                self.encoders[f"{col}_bins"] = bins
                        elif discretization_method == 'uniform':
                            df_processed[col], bins = pd.cut(
                                df_processed[col], bins=n_bins, labels=False, retbins=True
                            )
                            self.encoders[f"{col}_bins"] = bins
                    
                    # Ensure integer type
                    df_processed[col] = df_processed[col].astype(int)
            
            except Exception as e:
                print(f"Warning: Could not process {col}: {e}")
                try:
                    if df_processed[col].dtype != 'int64':
                        df_processed[col] = pd.Categorical(df_processed[col]).codes
                    if df_processed[col].min() < 0:
                        df_processed[col] = df_processed[col] - df_processed[col].min()
                except:
                    pass
        
        self.df_processed = df_processed
        self.data_encoded_once = True
        print(f"Data shape after preprocessing: {df_processed.shape}")
        print(f"Variables in processed data: {list(df_processed.columns)}")
        
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
                    # Connect key variables from current to next layer (limit connections)
                    for current_var in current_vars[:3]:  # Max 3 from each layer
                        for next_var in next_vars[:2]:    # Max 2 to each layer
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
            
            # Remove duplicates and limit total variables
            key_vars = list(set(key_vars))[:20]  # Max 20 variables for PC
            subset_df = self.df_processed[key_vars].copy().dropna()
            
            if len(subset_df) < 50:
                print("Insufficient data for PC algorithm, using expert structure")
                return self.create_expert_structure()
            
            # Use PC algorithm if available
            try:
                pc = PC(subset_df)
                model = pc.estimate(significance_level=0.05)
                edges = list(model.edges())
            except Exception as pc_error:
                print(f"PC algorithm failed: {pc_error}, using expert structure")
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
            
            key_vars = list(set(key_vars))[:15]  # Max 15 variables for Hill Climbing
            subset_df = self.df_processed[key_vars].copy().dropna()
            
            if len(subset_df) < 50:
                print("Insufficient data for Hill Climbing, using expert structure")
                return self.create_expert_structure()
            
            scoring_method = BicScore(subset_df)
            hc = HillClimbSearch(subset_df)
            
            # Start with expert structure as initial model (limited)
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
        FIXED: Create network with proper node validation
        """
        print(f"Creating network using {method} method...")
        
        # Ensure data is preprocessed
        if self.df_processed is None:
            self.preprocess_data()
        
        if method == 'expert':
            edges = self.create_expert_structure()
        elif method == 'pc':
            edges = self.learn_structure_pc()
        elif method == 'hill_climb':
            edges = self.learn_structure_hill_climb()
        else:
            print(f"Unknown method {method}, using expert")
            edges = self.create_expert_structure()
        
        # FIXED: Critical validation - only include nodes that exist in processed data
        valid_edges = []
        available_vars = set(self.df_processed.columns)
        
        for parent, child in edges:
            if parent in available_vars and child in available_vars:
                valid_edges.append((parent, child))
            else:
                if parent not in available_vars:
                    print(f"Warning: Parent node '{parent}' not in processed data, skipping edge ({parent}, {child})")
                if child not in available_vars:
                    print(f"Warning: Child node '{child}' not in processed data, skipping edge ({parent}, {child})")
        
        if not valid_edges:
            print("Warning: No valid edges found, creating minimal network with available variables")
            # Create minimal network with first few available variables
            available_list = list(available_vars)[:5]
            valid_edges = [(available_list[i], available_list[i+1]) for i in range(len(available_list)-1)]
        
        print(f"Using {len(valid_edges)} valid edges out of {len(edges)} proposed edges")
        
        # Create network from validated edges
        if PGMPY_AVAILABLE:
            try:
                self.network = BayesianNetwork(valid_edges)
                
                # Check for cycles and remove if necessary
                if not nx.is_directed_acyclic_graph(self.network):
                    print("Removing cycles from network...")
                    dag_edges = self._remove_cycles(valid_edges)
                    self.network = BayesianNetwork(dag_edges)
                
                print(f"Created BayesianNetwork with {len(self.network.nodes())} nodes and {len(self.network.edges())} edges")
                print(f"Network nodes: {sorted(list(self.network.nodes()))}")
                
            except Exception as e:
                print(f"Error creating pgmpy network: {e}")
                self.network = nx.DiGraph()
                self.network.add_edges_from(valid_edges)
        else:
            self.network = nx.DiGraph()
            self.network.add_edges_from(valid_edges)
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
    
    def fit_parameters(self, method: str = 'mle', pseudo_counts: int = 1) -> Optional[BayesianNetwork]:
        """
        Fit network parameters using Maximum Likelihood or Bayesian estimation.
        FIXED: More robust parameter fitting with better error handling
        """
        print(f"Fitting network parameters using {method}...")
        
        if not PGMPY_AVAILABLE or not isinstance(self.network, BayesianNetwork):
            print("Cannot fit parameters: pgmpy unavailable or network invalid")
            return None
        
        if self.df_processed is None:
            self.preprocess_data()
        
        network_vars = list(self.network.nodes())
        if not network_vars:
            print("No nodes in network to fit")
            return None
        
        # Ensure all network variables are in the data
        missing_vars = [v for v in network_vars if v not in self.df_processed.columns]
        if missing_vars:
            print(f"Warning: Variables {missing_vars} not in data, removing from network")
            for var in missing_vars:
                if self.network.has_node(var):
                    self.network.remove_node(var)
            network_vars = list(self.network.nodes())
        
        if not network_vars:
            print("No valid nodes remaining after cleanup")
            return None
        
        subset_df = self.df_processed[network_vars].copy().dropna()
        
        if len(subset_df) == 0:
            print("No complete cases available for parameter fitting")
            return None
        
        # Ensure all variables are properly encoded as integers
        for var in network_vars:
            if subset_df[var].dtype not in ['int64', 'int32']:
                subset_df[var] = pd.Categorical(subset_df[var]).codes
            # Ensure non-negative integers
            if subset_df[var].min() < 0:
                subset_df[var] = subset_df[var] - subset_df[var].min()
        
        self.fitted_network = self.network.copy()
        successful_fits = 0
        failed_nodes = []
        
        for node in network_vars:
            try:
                if method == 'mle':
                    estimator = MaximumLikelihoodEstimator(self.network, subset_df)
                    cpd = estimator.estimate_cpd(node)
                elif method == 'bayesian':
                    estimator = BayesianEstimator(self.network, subset_df)
                    cpd = estimator.estimate_cpd(node, equivalent_sample_size=pseudo_counts)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                if cpd is not None:
                    # Ensure CPD is properly normalized
                    cpd.normalize()
                    self.fitted_network.add_cpds(cpd)
                    successful_fits += 1
                else:
                    failed_nodes.append(node)
                    
            except Exception as e:
                print(f"Warning: Could not fit CPD for node {node}: {e}")
                failed_nodes.append(node)
                # Try to add a uniform CPD as fallback
                try:
                    uniform_cpd = self._create_uniform_cpd(node, subset_df)
                    if uniform_cpd:
                        self.fitted_network.add_cpds(uniform_cpd)
                        successful_fits += 1
                        print(f"  Added uniform CPD for {node}")
                except Exception as uniform_error:
                    print(f"  Failed to create uniform CPD for {node}: {uniform_error}")
        
        print(f"Successfully fitted CPDs for {successful_fits}/{len(network_vars)} nodes")
        if failed_nodes:
            print(f"Failed nodes: {failed_nodes}")
        
        # Validate model
        try:
            self.fitted_network.check_model()
            print("Network validation successful!")
        except Exception as e:
            print(f"Network validation warning: {e}")
        
        return self.fitted_network
    
    def _create_uniform_cpd(self, node: str, data: pd.DataFrame) -> Optional[TabularCPD]:
        """Create uniform CPD for a node that couldn't be fitted normally"""
        try:
            node_values = sorted(data[node].unique())
            cardinality = len(node_values)
            
            parents = list(self.network.predecessors(node))
            
            if not parents:
                # No parents - uniform marginal
                values = np.ones(cardinality) / cardinality
                cpd = TabularCPD(variable=node, variable_card=cardinality, values=values.reshape(-1, 1))
            else:
                # With parents - uniform conditional
                parent_cards = [len(data[p].unique()) for p in parents if p in data.columns]
                if not parent_cards:
                    # Parents not in data, treat as no parents
                    values = np.ones(cardinality) / cardinality
                    cpd = TabularCPD(variable=node, variable_card=cardinality, values=values.reshape(-1, 1))
                else:
                    total_combinations = np.prod(parent_cards)
                    values = np.ones((cardinality, total_combinations)) / cardinality
                    cpd = TabularCPD(variable=node, variable_card=cardinality,
                                     values=values, evidence=parents, evidence_card=parent_cards)
            
            cpd.normalize()
            return cpd
            
        except Exception as e:
            print(f"Error creating uniform CPD for {node}: {e}")
            return None
    
    def perform_inference(self, evidence: Dict[str, Union[str, int]], 
                     query_variables: Optional[List[str]] = None) -> Optional[Dict]:
        """
        FIXED: Perform inference with better validation
        """
        if not PGMPY_AVAILABLE or self.fitted_network is None:
            print("Cannot perform inference: network not fitted or pgmpy unavailable")
            return None
        
        try:
            inference_engine = VariableElimination(self.fitted_network)
            
            # FIXED: Get nodes from the actual fitted network
            available_nodes = list(self.fitted_network.nodes())
            print(f"Available nodes in fitted network: {sorted(available_nodes)}")
            
            # Validate and set query variables
            if query_variables is None:
                default_queries = ['a1503', 'a1210', 'a0301', 'a0401', 'a0701p']
                query_variables = [var for var in default_queries if var in available_nodes]
                
                if not query_variables and available_nodes:
                    # Use first 2-3 available nodes as queries
                    query_variables = available_nodes[:min(3, len(available_nodes))]
            
            # FIXED: Validate ALL query variables exist
            valid_queries = [var for var in query_variables if var in available_nodes]
            if not valid_queries:
                print(f"ERROR: No valid query variables found")
                print(f"Requested: {query_variables}")
                print(f"Available: {available_nodes}")
                return None
            
            print(f"Using query variables: {valid_queries}")
            
            # FIXED: Process evidence with validation
            processed_evidence = {}
            for var, value in evidence.items():
                if var not in available_nodes:
                    print(f"Warning: Evidence variable '{var}' not in network, skipping")
                    continue
                
                try:
                    # Convert to integer (all our processed data should be integers)
                    if isinstance(value, str):
                        if var in self.encoders and hasattr(self.encoders[var], 'transform'):
                            processed_evidence[var] = int(self.encoders[var].transform([value])[0])
                        else:
                            processed_evidence[var] = int(float(value))
                    else:
                        processed_evidence[var] = int(value)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not process evidence {var}={value}: {e}")
                    continue
            
            if not processed_evidence:
                print("Warning: No valid evidence could be processed")
                # Try inference without evidence
                print("Attempting inference without evidence...")
            
            print(f"Processed evidence: {processed_evidence}")
            
            # Perform inference
            if processed_evidence:
                result = inference_engine.query(variables=valid_queries, evidence=processed_evidence)
            else:
                result = inference_engine.query(variables=valid_queries)
            
            print(f"\nInference Results:")
            print("-" * 50)
            print(result)
            
            return {
                'result': result, 
                'evidence': processed_evidence, 
                'query_vars': valid_queries
            }
            
        except Exception as e:
            print(f"Error in inference: {e}")
            import traceback
            traceback.print_exc()
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
        color_map = {
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
        
        for layer in sorted(unique_layers):
            if layer in color_map:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color_map[layer], markersize=10, 
                              label=layer.replace('_', ' ').title())
                )
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def _print_network_stats(self) -> None:
        """Print comprehensive network statistics"""
        print(f"\n{'='*50}")
        print("NETWORK STATISTICS")
        print(f"{'='*50}")
        print(f"Nodes: {len(self.network.nodes())}")
        print(f"Edges: {len(self.network.edges())}")
        
        if hasattr(self.network, 'number_of_nodes') and self.network.number_of_nodes() > 1:
            print(f"Density: {nx.density(self.network):.3f}")
        
        if isinstance(self.network, nx.DiGraph):
            print(f"Is DAG: {nx.is_directed_acyclic_graph(self.network)}")
            
            # Node statistics
            in_degrees = dict(self.network.in_degree())
            out_degrees = dict(self.network.out_degree())
            
            if in_degrees and out_degrees:
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
            'encoders': {},  # Encoders need special handling
            'variable_layers': self.variable_layers,
            'variable_meanings': self.variable_meanings,
        }
        
        # Save encoders with special handling for sklearn objects
        for key, encoder in self.encoders.items():
            if hasattr(encoder, 'classes_'):
                model_data['encoders'][key] = {
                    'type': 'LabelEncoder',
                    'classes_': encoder.classes_.tolist()
                }
            elif isinstance(encoder, np.ndarray):
                model_data['encoders'][key] = {
                    'type': 'bins',
                    'values': encoder.tolist()
                }
        
        # Save fitted network separately if available
        if self.fitted_network and PGMPY_AVAILABLE:
            try:
                cpds_data = []
                for cpd in self.fitted_network.get_cpds():
                    cpd_data = {
                        'variable': cpd.variable,
                        'variable_card': cpd.variable_card,
                        'values': cpd.get_values().tolist(),
                    }
                    
                    if hasattr(cpd, 'evidence') and cpd.evidence:
                        cpd_data['evidence'] = list(cpd.evidence)
                        cpd_data['evidence_card'] = list(cpd.evidence_card)
                    
                    cpds_data.append(cpd_data)
                    
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
        self.variable_layers = model_data.get('variable_layers', {})
        self.variable_meanings = model_data.get('variable_meanings', {})
        
        # Restore encoders
        for key, encoder_data in model_data.get('encoders', {}).items():
            if encoder_data.get('type') == 'LabelEncoder':
                le = LabelEncoder()
                le.classes_ = np.array(encoder_data['classes_'])
                self.encoders[key] = le
            elif encoder_data.get('type') == 'bins':
                self.encoders[key] = np.array(encoder_data['values'])
        
        # Restore network structure
        edges = model_data.get('network_edges', [])
        if edges and PGMPY_AVAILABLE:
            try:
                self.network = BayesianNetwork(edges)
                
                # Restore CPDs if available
                if 'cpds' in model_data:
                    for cpd_data in model_data['cpds']:
                        try:
                            if 'evidence' in cpd_data:
                                cpd = TabularCPD(
                                    variable=cpd_data['variable'],
                                    variable_card=int(cpd_data['variable_card']),
                                    values=cpd_data['values'],
                                    evidence=cpd_data['evidence'],
                                    evidence_card=cpd_data['evidence_card']
                                )
                            else:
                                cpd = TabularCPD(
                                    variable=cpd_data['variable'],
                                    variable_card=int(cpd_data['variable_card']),
                                    values=cpd_data['values']
                                )
                            self.network.add_cpds(cpd)
                        except Exception as cpd_err:
                            print(f"Warning: Could not load CPD for {cpd_data.get('variable')}: {cpd_err}")
                    
                    self.fitted_network = self.network
                    print("Loaded fitted network with CPDs")
                else:
                    print("Loaded network structure only")
                    
            except Exception as e:
                print(f"Error loading BayesianNetwork: {e}")
                self.network = nx.DiGraph()
                self.network.add_edges_from(edges)
                print("Fallback: Loaded as NetworkX DiGraph (no CPDs)")
        else:
            # If pgmpy not available, fallback to simple graph
            self.network = nx.DiGraph()
            if edges:
                self.network.add_edges_from(edges)
            print("Loaded as NetworkX DiGraph")
        
        print(f"Model loaded from {filepath}")
    
    def get_network_summary(self) -> Dict:
        """Get a comprehensive summary of the network"""
        summary = {
            'nodes': len(self.network.nodes()) if self.network else 0,
            'edges': len(self.network.edges()) if self.network else 0,
            'is_fitted': self.fitted_network is not None,
            'data_processed': self.df_processed is not None,
            'available_variables': list(self.df.columns) if self.df is not None else [],
            'network_variables': list(self.network.nodes()) if self.network else [],
        }
        
        if self.network and hasattr(self.network, 'nodes') and len(self.network.nodes()) > 0:
            summary['density'] = nx.density(self.network) if hasattr(nx, 'density') else 0
            summary['is_dag'] = nx.is_directed_acyclic_graph(self.network) if isinstance(self.network, nx.DiGraph) else False
        
        if self.fitted_network and PGMPY_AVAILABLE:
            try:
                cpds = self.fitted_network.get_cpds()
                summary['fitted_cpds'] = len(cpds)
                summary['cpd_variables'] = [cpd.variable for cpd in cpds]
            except:
                summary['fitted_cpds'] = 0
                summary['cpd_variables'] = []
        
        return summary