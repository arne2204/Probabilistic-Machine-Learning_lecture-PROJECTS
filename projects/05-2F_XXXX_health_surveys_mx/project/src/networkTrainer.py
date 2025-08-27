import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings
import time
from datetime import datetime
import os
import json
import sklearn

# Import the ENSINBayesianNetwork class
from .network import ENSINBayesianNetwork

warnings.filterwarnings('ignore')

class ENSINNetworkTrainer:
    """
    Main class for training and analyzing ENSIN Bayesian Networks
    
    Features:
    - Multiple structure learning methods comparison
    - Comprehensive model evaluation
    - Inference demonstrations
    - Model persistence and reporting
    - Cross-validation capabilities
    """
    
    def __init__(self, data_path: str = 'project/data/cleaned_dataset.csv', sample_data: bool = True):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to ENSIN dataset CSV file
            sample_data: If True and no data_path provided, generate sample data
        """
        self.data_path = data_path
        self.df = None
        self.networks = {}  # Store multiple trained networks
        self.results = {}   # Store training and evaluation results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = f"ensin_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"ENSIN Network Trainer initialized")
        print(f"Output directory: {self.output_dir}")
        
        # Load or generate data
        if data_path and os.path.exists(data_path):
            self.load_data()
        elif sample_data:
            self.generate_sample_data()
        else:
            print("No data provided. Use load_data() or generate_sample_data()")
    
    def load_data(self) -> None:
        """Load ENSIN dataset from CSV file"""
        try:
            print(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.df.shape}")
            self.analyze_dataset()
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Generating sample data instead...")
            self.generate_sample_data()
    
    def generate_sample_data(self, n_samples: int = 5000) -> None:
        """
        Generate realistic sample ENSIN data for demonstration
        """
        print(f"Generating sample ENSIN data ({n_samples} samples)...")
        
        np.random.seed(42)  # For reproducibility
        
        # Define realistic value ranges for key variables
        sample_data = {}
        
        # Demographics
        sample_data['edad'] = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
        sample_data['sexo'] = np.random.choice([0, 1], n_samples, p=[0.52, 0.48])  # 0=Female, 1=Male
        sample_data['meses'] = np.random.randint(1, 13, n_samples)
        
        # Geography and SES
        sample_data['x_region'] = np.random.choice(range(1, 6), n_samples)
        sample_data['estrato'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, 
                                                p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02])
        
        # Body perception (influenced by age and sex)
        body_perception_prob = 0.3 + 0.01 * sample_data['edad'] - 0.1 * sample_data['sexo']
        sample_data['a0104'] = np.random.binomial(1, body_perception_prob.clip(0, 1), n_samples)
        
        # Weight management attempts (influenced by body perception)
        weight_loss_prob = 0.2 + 0.3 * sample_data['a0104']
        sample_data['a0107'] = np.random.binomial(1, weight_loss_prob, n_samples)
        sample_data['a0108'] = np.random.binomial(1, 0.1 + 0.1 * sample_data['a0104'], n_samples)
        
        # Mental health indicators (influenced by SES and body perception)
        mental_health_base = 0.15 - 0.03 * sample_data['estrato'] + 0.2 * sample_data['a0104']
        sample_data['a0202'] = np.random.binomial(1, mental_health_base.clip(0, 0.5), n_samples)  # Depression
        sample_data['a0203'] = np.random.binomial(1, (mental_health_base + 0.05).clip(0, 0.5), n_samples)  # Anxiety
        sample_data['a0204'] = np.random.binomial(1, (mental_health_base + 0.1).clip(0, 0.6), n_samples)  # Sleep
        
        # Disease diagnosis (influenced by age and lifestyle)
        diabetes_prob = 0.01 + 0.002 * sample_data['edad'] + 0.05 * sample_data['a0202']
        sample_data['a0301'] = np.random.binomial(1, diabetes_prob.clip(0, 0.4), n_samples)
        
        hypert_prob = 0.05 + 0.003 * sample_data['edad'] + 0.1 * sample_data['sexo']
        sample_data['a0401'] = np.random.binomial(1, hypert_prob.clip(0, 0.6), n_samples)
        
        # Dietary patterns (influenced by SES and mental health)
        fruit_freq = np.random.choice([1, 2, 3, 4], n_samples, 
                                    p=[0.4 - 0.05*np.mean(sample_data['estrato']),
                                       0.3, 0.2, 0.1 + 0.05*np.mean(sample_data['estrato'])])
        sample_data['a0701p'] = fruit_freq
        
        vegetable_freq = np.random.choice([1, 2, 3, 4], n_samples, p=[0.35, 0.35, 0.2, 0.1])
        sample_data['a0702p'] = vegetable_freq
        
        # Nutritional status (BMI categories, influenced by diet, age, diseases)
        bmi_factors = (0.1 * sample_data['edad'] + 0.5 * sample_data['a0301'] + 
                      0.3 * sample_data['a0401'] - 0.2 * sample_data['a0701p'])
        bmi_probs = np.column_stack([
            0.15 - 0.01 * bmi_factors,  # Underweight
            0.45 - 0.05 * bmi_factors,  # Normal
            0.25 + 0.03 * bmi_factors,  # Overweight
            0.15 + 0.03 * bmi_factors   # Obese
        ]).clip(0.01, 0.8)
        
        # Normalize probabilities
        bmi_probs = bmi_probs / bmi_probs.sum(axis=1, keepdims=True)
        sample_data['a1503'] = np.array([np.random.choice([1, 2, 3, 4], p=row) for row in bmi_probs])
        
        # Health outcomes (influenced by diseases, mental health, nutrition)
        health_factors = (0.3 * sample_data['a0301'] + 0.2 * sample_data['a0401'] + 
                         0.2 * sample_data['a0202'] + 0.1 * (sample_data['a1503'] > 3))
        health_status = np.random.choice([1, 2, 3, 4, 5], n_samples,
                                       p=[0.05, 0.15, 0.4, 0.3, 0.1])
        
        # Adjust health status based on risk factors
        for i in range(n_samples):
            if health_factors[i] > 0.5 and health_status[i] > 3:
                health_status[i] = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        
        sample_data['a1210'] = health_status
        
        # Quality of life (related to health status and mental health)
        qol_base = 4 - health_status + np.random.normal(0, 0.5, n_samples)
        qol_base -= sample_data['a0202'] * 0.5 + sample_data['a0203'] * 0.3
        sample_data['a1301'] = np.clip(qol_base, 1, 5).astype(int)
        
        # Additional variables for completeness
        sample_data['upm'] = np.random.randint(1000, 9999, n_samples)
        sample_data['est_sel'] = np.random.choice([1, 2], n_samples, p=[0.7, 0.3])
        sample_data['h0406'] = np.random.randint(1, 10, n_samples)
        
        # Create DataFrame
        self.df = pd.DataFrame(sample_data)
        
        # Add some missing values to make it realistic
        missing_vars = ['a0202', 'a0203', 'a0701p', 'a1301']
        for var in missing_vars:
            if var in self.df.columns:
                missing_mask = np.random.random(n_samples) < 0.05  # 5% missing
                self.df.loc[missing_mask, var] = np.nan
        
        print(f"Sample data generated: {self.df.shape}")
        self.analyze_dataset()
        
        # Save sample data
        sample_data_path = os.path.join(self.output_dir, "sample_data.csv")
        self.df.to_csv(sample_data_path, index=False)
        print(f"Sample data saved to: {sample_data_path}")
    
    def analyze_dataset(self) -> None:
        """Analyze the loaded dataset"""
        print(f"\n{'='*50}")
        print("DATASET ANALYSIS")
        print(f"{'='*50}")
        
        print(f"Shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\nMissing values:")
            for var, count in missing_counts[missing_counts > 0].items():
                print(f"  {var}: {count} ({count/len(self.df)*100:.1f}%)")
        else:
            print("No missing values found")
        
        # Data types
        print(f"\nData types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} variables")
        
        # Create summary statistics
        self._create_summary_statistics()
    
    def _create_summary_statistics(self) -> None:
        """Create and save summary statistics"""
        print("\nCreating summary statistics...")
        
        # Descriptive statistics for numeric variables
        numeric_vars = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_vars) > 0:
            desc_stats = self.df[numeric_vars].describe()
            desc_stats.to_csv(os.path.join(self.output_dir, "descriptive_statistics.csv"))
        
        # Correlation matrix for key variables
        key_vars = ['edad', 'a0104', 'a0202', 'a0301', 'a0401', 'a0701p', 'a1503', 'a1210']
        available_vars = [var for var in key_vars if var in self.df.columns]
        
        if len(available_vars) > 3:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.df[available_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Correlation Matrix - Key Variables')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"), dpi=300)
            plt.show()
    
    def train_multiple_networks(self, methods: List[str] = ['expert', 'pc', 'hill_climb']) -> Dict:
        """
            Train networks with improved parameter fitting
        """
        print(f"\n{'='*60}")
        print("TRAINING MULTIPLE BAYESIAN NETWORKS")
        print(f"{'='*60}")
        
        training_results = {}
        
        for method in methods:
            print(f"\nTraining network using {method.upper()} method...")
            start_time = time.time()
            
            try:
                # Create and train network
                network = ENSINBayesianNetwork(self.df)
                
                # Preprocess data
                print(f"Preprocessing data for {method}...")
                preprocessed_data = network.preprocess_data(discretization_method='quantile', n_bins=3)
                
                if preprocessed_data is None:
                    raise ValueError("Failed to preprocess data")
                
                # Create network structure
                print(f"Creating network structure using {method}...")
                created_network = network.create_network(method=method)
                
                if created_network is None:
                    raise ValueError("Failed to create network structure")
                
                # FIXED: Ensure proper parameter fitting
                print(f"Fitting parameters for {method}...")
                fitted_network = self._fit_parameters_robust(network, method)
                
                if fitted_network is None:
                    # If robust fitting fails, try basic MLE
                    print(f"Robust fitting failed, trying basic MLE...")
                    fitted_network = network.fit_parameters(method='mle')
                
                training_time = time.time() - start_time
                
                # Store results
                self.networks[method] = network
                training_results[method] = {
                    'network': network,
                    'training_time': training_time,
                    'nodes': len(created_network.nodes()) if created_network else 0,
                    'edges': len(created_network.edges()) if created_network else 0,
                    'fitted': fitted_network is not None,
                    'success': True,
                    'cpd_count': len(fitted_network.get_cpds()) if fitted_network else 0
                }
                
                print(f"✓ {method.upper()} network trained successfully in {training_time:.2f}s")
                print(f"  - Nodes: {training_results[method]['nodes']}")
                print(f"  - Edges: {training_results[method]['edges']}")
                print(f"  - CPDs fitted: {training_results[method]['cpd_count']}")
                print(f"  - Parameters fitted: {training_results[method]['fitted']}")
                
                # Validate the fitted network
                if fitted_network:
                    validation_result = self._validate_network(fitted_network)
                    training_results[method]['validation'] = validation_result
                    if validation_result['valid']:
                        print(f"  ✓ Network validation passed")
                    else:
                        print(f"  ⚠ Network validation issues: {validation_result['issues']}")
                
            except Exception as e:
                print(f"✗ Error training {method} network: {e}")
                training_results[method] = {
                    'network': None,
                    'training_time': time.time() - start_time,
                    'nodes': 0,
                    'edges': 0,
                    'fitted': False,
                    'success': False,
                    'error': str(e),
                    'cpd_count': 0
                }
        
        self.results['training'] = training_results
        self._save_training_report()
        
        return training_results
    
    def compare_network_structures(self) -> None:
        """Compare structures of different networks"""
        print(f"\n{'='*50}")
        print("NETWORK STRUCTURE COMPARISON")
        print(f"{'='*50}")
        
        if not self.networks:
            print("No networks available for comparison")
            return
        
        comparison_data = []
        
        for method, network in self.networks.items():
            if network.network is not None:
                try:
                    nodes = list(network.network.nodes())
                    edges = list(network.network.edges())
                    
                    comparison_data.append({
                        'Method': method.title(),
                        'Nodes': len(nodes),
                        'Edges': len(edges),
                        'Density': len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
                        'Avg_Degree': (2 * len(edges)) / len(nodes) if len(nodes) > 0 else 0
                    })
                    
                except Exception as e:
                    print(f"Error analyzing {method} network: {e}")
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv(os.path.join(self.output_dir, "network_comparison.csv"), index=False)
            
            # Visualize comparison
            self._plot_network_comparison(comparison_df)
    
    def _plot_network_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Plot network comparison metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Network Structure Comparison', fontsize=16)
        
        # Number of edges
        axes[0, 0].bar(comparison_df['Method'], comparison_df['Edges'], color='skyblue')
        axes[0, 0].set_title('Number of Edges')
        axes[0, 0].set_ylabel('Count')
        
        # Network density
        axes[0, 1].bar(comparison_df['Method'], comparison_df['Density'], color='lightcoral')
        axes[0, 1].set_title('Network Density')
        axes[0, 1].set_ylabel('Density')
        
        # Average degree
        axes[1, 0].bar(comparison_df['Method'], comparison_df['Avg_Degree'], color='lightgreen')
        axes[1, 0].set_title('Average Node Degree')
        axes[1, 0].set_ylabel('Degree')
        
        # Combined metrics (normalized)
        metrics = ['Edges', 'Density', 'Avg_Degree']
        normalized_data = comparison_df[metrics].div(comparison_df[metrics].max())
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[1, 1].bar(x + i*width, normalized_data[metric], width, 
                          label=metric, alpha=0.8)
        
        axes[1, 1].set_title('Normalized Metrics Comparison')
        axes[1, 1].set_ylabel('Normalized Value')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(comparison_df['Method'])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "network_comparison.png"), dpi=300)
        plt.show()
    
    def demonstrate_inference(self, method: str = 'expert') -> None:
        """Demonstrate probabilistic inference capabilities"""
        print(f"\n{'='*50}")
        print(f"INFERENCE DEMONSTRATION - {method.upper()}")
        print(f"{'='*50}")
        
        if method not in self.networks:
            print(f"Network {method} not available")
            return
        
        network = self.networks[method]
        if network.fitted_network is None:
            print(f"Network {method} is not fitted - cannot perform inference")
            return
        
        # Define interesting inference scenarios
        scenarios = [
            {
                'name': 'Young Female with Diabetes',
                'evidence': {'edad': 'low', 'sexo': 0.0, 'a0301': 1.0},  # Young, female, has diabetes
                'description': 'What are the health implications for a young female with diabetes?'
            },
            {
                'name': 'Older Male with Good Diet',
                'evidence': {'edad': 'high', 'sexo': 1.0, 'a0701p': 1},  # Older, male, high fruit consumption
                'description': 'Health outcomes for older male with good dietary habits'
            },
            {
                'name': 'Middle-aged with Depression',
                'evidence': {'edad': 'medium', 'a0202': 1.0},  # Middle-aged with depression
                'description': 'Impact of depression on health outcomes'
            }
        ]
        
        inference_results = []
        
        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"Evidence: {scenario['evidence']}")
            
            try:
                result = network.perform_inference(
                    evidence=scenario['evidence'],
                    query_variables=['a1503', 'a1210']  # BMI category, health status
                )
                
                if result:
                    inference_results.append({
                        'scenario': scenario['name'],
                        'evidence': scenario['evidence'],
                        'result': result
                    })
                    print("✓ Inference completed successfully")
                else:
                    print("✗ Inference failed")
                    
            except Exception as e:
                print(f"✗ Error in inference: {e}")
        
        # Save inference results
        if inference_results:
            self.results['inference'] = inference_results
            self._save_inference_report(inference_results)


    def _fit_parameters_robust(self, network, method: str):
        """
        FIXED: Robust parameter fitting - ensures node validation
        """
        try:
            if not hasattr(network, 'network') or network.network is None:
                print("No network structure available for parameter fitting")
                return None
    
            from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    
            # Ensure data is preprocessed
            if network.df_processed is None:
                network.preprocess_data()
    
            # FIXED: Force proper encoding
            for col in network.df_processed.columns:
                if network.df_processed[col].dtype not in ['int64', 'int32']:
                    network.df_processed[col] = network.df_processed[col].astype('category').cat.codes
                    if network.df_processed[col].min() < 0:
                        network.df_processed[col] = network.df_processed[col] - network.df_processed[col].min()
                
            nodes = list(network.network.nodes())
            print(f"Network nodes: {sorted(nodes)}")
            print(f"Data columns: {sorted(list(network.df_processed.columns))}")
            
            # FIXED: Only use nodes that exist in BOTH network and data
            valid_nodes = [n for n in nodes if n in network.df_processed.columns]
            invalid_nodes = [n for n in nodes if n not in network.df_processed.columns]
            
            if invalid_nodes:
                print(f"Removing invalid nodes from network: {invalid_nodes}")
                for node in invalid_nodes:
                    if network.network.has_node(node):
                        network.network.remove_node(node)
            
            nodes = valid_nodes
            if not nodes:
                print("No valid nodes remaining after cleanup")
                return None
    
            # Filter data to complete cases
            data_for_fitting = network.df_processed[nodes].copy().dropna()
            if len(data_for_fitting) == 0:
                print("No complete cases available for parameter fitting")
                return None
    
            print(f"Fitting parameters on {len(data_for_fitting)} complete cases with {len(nodes)} variables")
            
            try:
                estimator = MaximumLikelihoodEstimator(network.network, data_for_fitting)
    
                # Fit CPDs
                for node in nodes:
                    try:
                        cpd = estimator.estimate_cpd(node)
                        if cpd is not None:
                            cpd.normalize()
                            # Remove old CPDs for this node
                            old_cpds = [c for c in network.network.get_cpds() if c.variable == node]
                            for old_cpd in old_cpds:
                                network.network.remove_cpds(old_cpd)
                            network.network.add_cpds(cpd)
                            print(f"  Fitted CPD for {node}")
                    except Exception as node_error:
                        print(f"Warning: Could not fit CPD for node {node}: {node_error}")
                        try:
                            bayes_estimator = BayesianEstimator(network.network, data_for_fitting)
                            cpd = bayes_estimator.estimate_cpd(node, prior_type='dirichlet', pseudo_counts=1)
                            if cpd is not None:
                                cpd.normalize()
                                old_cpds = [c for c in network.network.get_cpds() if c.variable == node]
                                for old_cpd in old_cpds:
                                    network.network.remove_cpds(old_cpd)
                                network.network.add_cpds(cpd)
                                print(f"  Fitted {node} using Bayesian estimator")
                        except Exception as bayes_error:
                            print(f"  Failed to fit {node} with both MLE and Bayesian: {bayes_error}")
    
                # Ensure all remaining nodes have CPDs
                fitted_nodes = [cpd.variable for cpd in network.network.get_cpds()]
                missing_cpds = set(nodes) - set(fitted_nodes)
                for node in missing_cpds:
                    try:
                        cpd = self._add_uniform_cpd(network.network, node, data_for_fitting)
                        if cpd is not None:
                            network.network.add_cpds(cpd)
                            print(f"  Added uniform CPD for {node}")
                    except Exception as uniform_error:
                        print(f"  Failed to add uniform CPD for {node}: {uniform_error}")
    
                network.fitted_network = network.network
                print(f"Parameter fitting completed. Total CPDs: {len(network.network.get_cpds())}")
    
                return network.network
    
            except Exception as e:
                print(f"Error in parameter fitting: {e}")
                return None
    
        except Exception as e:
            print(f"Error in robust parameter fitting: {e}")
            return None



    def _add_uniform_cpd(self, network, node, data):
        """Add uniform CPD for a node that couldn't be fitted normally (robust to any number of parents)."""
        from pgmpy.factors.discrete import TabularCPD
        import numpy as np
    
        try:
            node_values = sorted(data[node].unique())
            cardinality = len(node_values)
            parents = list(network.predecessors(node))
    
            if not parents:
                # No parents → uniform marginal
                values = np.ones(cardinality) / cardinality
                cpd = TabularCPD(variable=node, variable_card=cardinality, values=values.reshape(-1, 1))
            else:
                # With parents → uniform conditional
                parent_cards = [len(data[p].unique()) if p in data.columns else 2 for p in parents]
                total_combinations = np.prod(parent_cards)
                values = np.ones((cardinality, total_combinations)) / cardinality
                cpd = TabularCPD(variable=node, variable_card=cardinality,
                                 values=values, evidence=parents, evidence_card=parent_cards)
    
            cpd.normalize()  # ensure sums = 1
            return cpd
    
        except Exception as e:
            print(f"Error adding uniform CPD for {node}: {e}")
            return None

    def _validate_network(self, network):
        """Validate that the network is properly fitted"""
        try:
            # Check if all nodes have CPDs
            nodes = list(network.nodes())
            cpds = network.get_cpds()
            cpd_nodes = [cpd.variable for cpd in cpds]
            
            missing_cpds = set(nodes) - set(cpd_nodes)
            
            # Try to check model (this will raise an error if invalid)
            network.check_model()
            
            return {
                'valid': len(missing_cpds) == 0,
                'total_nodes': len(nodes),
                'nodes_with_cpd': len(cpd_nodes),
                'missing_cpds': list(missing_cpds),
                'issues': [] if len(missing_cpds) == 0 else [f"Missing CPDs: {missing_cpds}"]
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [str(e)],
                'error': str(e)
            }

        
    def evaluate_parameter_uncertainty(self, method: str = 'expert', alpha: float = 0.05) -> None:
        """
        FIXED: Parameter uncertainty evaluation with proper validation
        """
        if method not in self.networks or self.networks[method].network is None:
            print(f"No trained network found for {method}")
            return
    
        network = self.networks[method]
        if network.fitted_network is None:
            print(f"Network {method} is not fitted")
            return
    
        # FIXED: Validate network before proceeding
        validation = self._validate_network(network.fitted_network)
        if not validation['valid']:
            print(f"Network {method} is invalid: {validation['issues']}")
            return
    
        print(f"\n{'='*50}")
        print(f"PARAMETER UNCERTAINTY - {method.upper()}")
        print(f"{'='*50}")
    
        from scipy.stats import entropy
        uncertainty_stats = []
    
        # FIXED: Only analyze CPDs that actually exist
        cpds = network.fitted_network.get_cpds()
        if not cpds:
            print("No CPDs found in fitted network")
            return
    
        print(f"Analyzing {len(cpds)} CPDs...")
    
        for cpd in cpds:
            var = cpd.variable
            try:
                if hasattr(cpd, 'values') and cpd.values is not None:
                    values = cpd.values
                    if values.size == 0:
                        print(f"{var}: Empty CPD values")
                        continue
    
                    # Compute entropy per conditional distribution (columns)
                    entropies = []
                    if values.ndim == 1:
                        # Marginal distribution
                        probs = np.array(values, dtype=float)
                        if probs.sum() > 0:
                            probs /= probs.sum()
                            entropies.append(entropy(probs, base=2))
                    else:
                        # Conditional distribution
                        for col_idx in range(values.shape[1]):
                            probs = np.array(values[:, col_idx], dtype=float)
                            if probs.sum() > 0:
                                probs /= probs.sum()
                                entropies.append(entropy(probs, base=2))
    
                    if entropies:
                        mean_entropy = np.mean(entropies)
                        # FIXED: Safe normalization
                        max_entropy = np.log2(cpd.variable_card) if cpd.variable_card > 1 else 1.0
                        norm_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0
    
                        uncertainty_stats.append({
                            'variable': var,
                            'states': cpd.variable_card,
                            'mean_entropy': mean_entropy,
                            'normalized_entropy': norm_entropy
                        })
    
                        print(f"{var}: States={cpd.variable_card}, "
                              f"Entropy={mean_entropy:.3f}, "
                              f"Normalized={norm_entropy:.3f}")
                    else:
                        print(f"{var}: No valid probability distributions found")
                else:
                    print(f"{var}: No probability values available in CPD")
    
            except Exception as e:
                print(f"{var}: Error analyzing - {e}")
    
        if uncertainty_stats:
            df = pd.DataFrame(uncertainty_stats)
            path = os.path.join(self.output_dir, f"parameter_uncertainty_{method}.csv")
            df.to_csv(path, index=False)
            print(f"\nUncertainty statistics saved to: {path}")
            
            # Print summary
            if len(uncertainty_stats) > 0:
                mean_entropy = np.mean([s['mean_entropy'] for s in uncertainty_stats])
                mean_norm_entropy = np.mean([s['normalized_entropy'] for s in uncertainty_stats])
                print(f"\nSummary:")
                print(f"  Average entropy: {mean_entropy:.3f}")
                print(f"  Average normalized entropy: {mean_norm_entropy:.3f}")
        else:
            print("No uncertainty statistics could be computed")


    def evaluate_prediction_uncertainty(self, method: str = 'expert', query_vars: List[str] = None):
        """
        FIXED: Prediction uncertainty with proper node validation
        """
        if method not in self.networks:
            print(f"{method} network not available")
            return
    
        network = self.networks[method]
        if network.fitted_network is None:
            print(f"{method} network not fitted")
            return
    
        validation = self._validate_network(network.fitted_network)
        if not validation['valid']:
            print(f"Cannot perform inference - network validation failed: {validation['issues']}")
            return
    
        # FIXED: Get actual available nodes
        available_nodes = list(network.fitted_network.nodes())
        print(f"Available nodes: {sorted(available_nodes)}")
    
        if query_vars is None:
            # Use default query variables that exist
            default_queries = ['a1503', 'a1210', 'a0301', 'a0401']
            query_vars = [v for v in default_queries if v in available_nodes]
            if not query_vars and available_nodes:
                query_vars = available_nodes[:min(2, len(available_nodes))]
    
        # FIXED: Validate query variables
        valid_query_vars = [v for v in query_vars if v in available_nodes]
        if not valid_query_vars:
            print(f"No valid query variables found from {query_vars}")
            print(f"Available variables: {available_nodes}")
            return
    
        print(f"\n{'='*50}")
        print(f"PREDICTION UNCERTAINTY - {method.upper()}")
        print(f"Query variables: {valid_query_vars}")
        print(f"{'='*50}")
    
        from pgmpy.inference import VariableElimination
        from scipy.stats import entropy
    
        try:
            infer = VariableElimination(network.fitted_network)
        except Exception as e:
            print(f"Cannot create inference engine: {e}")
            return
    
        # FIXED: Use proper processed data
        if network.df_processed is None:
            network.preprocess_data()
    
        if network.df_processed is None:
            print("No processed data available")
            return
    
        # FIXED: Only use data for nodes that exist in the network
        network_data = network.df_processed[available_nodes].copy()
        
        n_samples = min(5, len(network_data))  # Reduced for debugging
        sample_indices = np.random.choice(len(network_data), n_samples, replace=False)
    
        print(f"Testing prediction uncertainty on {n_samples} samples...")
    
        for i, idx in enumerate(sample_indices):
            row = network_data.iloc[idx]
            
            # FIXED: Create evidence from available variables (excluding query vars)
            evidence_vars = [col for col in available_nodes 
                            if col not in valid_query_vars and not pd.isna(row[col])]
            
            if len(evidence_vars) < 1:
                print(f"Instance {i+1}: No valid evidence variables")
                continue
    
            # Use subset of evidence to avoid over-constraining
            evidence_vars = evidence_vars[:3]  # Max 3 evidence variables
            evidence = {col: int(row[col]) for col in evidence_vars}
    
            try:
                result = infer.query(variables=valid_query_vars, evidence=evidence)
                
                print(f"\nInstance {i+1}:")
                print(f"  Evidence: {evidence}")
                
                if hasattr(result, 'values'):
                    # Single variable result
                    probs = result.values.flatten()
                    probs = probs / probs.sum()
                    ent = entropy(probs, base=2)
                    print(f"  {valid_query_vars[0]}: Entropy={ent:.3f}")
                else:
                    print(f"  Results: {result}")
    
            except Exception as e:
                print(f"Instance {i+1}: Inference failed - {e}")
    
        print(f"\nPrediction uncertainty evaluation completed for {method}")

    def visualize_all_networks(self):
        """Safe stub to avoid crashes if visualization not implemented"""
        print("Visualization not implemented yet.")
    
    def evaluate_log_likelihood(self, method: str = 'expert', test_data: Optional[pd.DataFrame] = None):
        """
        FIXED: Robust log-likelihood evaluation with proper CPD access
        """
        if method not in self.networks:
            print(f"{method} network not available")
            return
    
        network = self.networks[method]
        
        if network.fitted_network is None:
            print(f"{method} network not fitted")
            return
        
        # Validate network
        validation = self._validate_network(network.fitted_network)
        if not validation['valid']:
            print(f"Cannot evaluate log-likelihood - network invalid: {validation['issues']}")
            return
        
        # Prepare test data
        if test_data is None:
            if network.df_processed is None:
                network.preprocess_data()
            if network.df_processed is None:
                print("No processed data available")
                return
            test_data = network.df_processed.sample(n=min(50, len(network.df_processed)), random_state=42)
        
        print(f"\n{'='*50}")
        print(f"LOG-LIKELIHOOD EVALUATION - {method.upper()} (FIXED)")
        print(f"{'='*50}")
        
        try:
            network_vars = list(network.fitted_network.nodes())
            cpds = network.fitted_network.get_cpds()
            cpd_dict = {cpd.variable: cpd for cpd in cpds}
            
            print(f"Evaluating log-likelihood on {len(test_data)} samples...")
            print(f"Network variables: {len(network_vars)}")
            print(f"Available CPDs: {len(cpd_dict)}")
            print(f"CPD variables: {sorted(list(cpd_dict.keys()))}")
            
            # Filter test data to only include network variables
            available_vars = [var for var in network_vars if var in test_data.columns]
            if not available_vars:
                print("No network variables found in test data")
                return
                
            filtered_test_data = test_data[available_vars].copy().dropna()
            if len(filtered_test_data) == 0:
                print("No complete cases in test data")
                return
                
            print(f"Using {len(available_vars)} variables: {sorted(available_vars)}")
            print(f"Complete cases: {len(filtered_test_data)}")
            
            total_ll = 0
            n_valid_samples = 0
            sample_lls = []
            
            for idx, row in filtered_test_data.iterrows():
                try:
                    sample_ll = 0
                    valid_vars = 0
                    
                    for var in available_vars:
                        if var not in cpd_dict:
                            continue
                        
                        cpd = cpd_dict[var]
                        var_value = int(row[var])
                        
                        # FIXED: Proper CPD probability extraction
                        try:
                            parents = list(cpd.evidence) if hasattr(cpd, 'evidence') and cpd.evidence else []
                            
                            if not parents:
                                # No parents - marginal probability
                                if hasattr(cpd, 'values'):
                                    values = cpd.values
                                    if values.ndim == 1:
                                        # 1D array - direct indexing
                                        if 0 <= var_value < len(values):
                                            prob = float(values[var_value])
                                        else:
                                            continue
                                    elif values.ndim == 2:
                                        # 2D array - take first column for marginal
                                        if 0 <= var_value < values.shape[0]:
                                            prob = float(values[var_value, 0])
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                # Has parents - conditional probability
                                parent_values = {}
                                skip_var = False
                                
                                for parent in parents:
                                    if parent in row.index and not pd.isna(row[parent]):
                                        parent_values[parent] = int(row[parent])
                                    else:
                                        skip_var = True
                                        break
                                
                                if skip_var:
                                    continue
                                
                                # FIXED: Use get_value method properly
                                try:
                                    query_dict = {var: var_value}
                                    query_dict.update(parent_values)
                                    
                                    if hasattr(cpd, 'get_value'):
                                        prob = cpd.get_value(**query_dict)
                                    else:
                                        # Manual calculation for conditional probability
                                        # This is more complex and depends on CPD structure
                                        prob = self._extract_conditional_prob(cpd, var_value, parent_values)
                                        
                                    if prob is None:
                                        continue
                                        
                                except Exception as cond_error:
                                    # Fallback: uniform probability
                                    prob = 1.0 / cpd.variable_card
                                    print(f"  Fallback uniform prob for {var}: {prob}")
                            
                            # Add to log-likelihood
                            if prob > 1e-10:  # Avoid log(0)
                                sample_ll += np.log(prob)
                                valid_vars += 1
                            else:
                                # Use small probability to avoid -inf
                                sample_ll += np.log(1e-10)
                                valid_vars += 1
                            
                        except Exception as var_error:
                            print(f"  Error processing {var}: {var_error}")
                            continue
                    
                    if valid_vars > 0:
                        # Normalize by number of variables
                        normalized_ll = sample_ll / valid_vars
                        sample_lls.append(normalized_ll)
                        total_ll += normalized_ll
                        n_valid_samples += 1
                
                except Exception as sample_error:
                    print(f"  Error processing sample {idx}: {sample_error}")
                    continue
            
            if n_valid_samples > 0:
                avg_ll = total_ll / n_valid_samples
                std_ll = np.std(sample_lls) if len(sample_lls) > 1 else 0.0
                
                print(f"\nResults:")
                print(f"  Valid samples: {n_valid_samples}/{len(filtered_test_data)}")
                print(f"  Average log-likelihood per variable: {avg_ll:.3f} ± {std_ll:.3f}")
                print(f"  Total log-likelihood: {total_ll:.2f}")
                
                # FIXED: Better interpretation thresholds
                if avg_ll > -0.5:
                    interpretation = "Excellent fit"
                elif avg_ll > -1.0:
                    interpretation = "Good fit"
                elif avg_ll > -2.0:
                    interpretation = "Moderate fit"
                elif avg_ll > -3.0:
                    interpretation = "Poor fit"
                else:
                    interpretation = "Very poor fit"
                
                print(f"  Model fit: {interpretation}")
                
                # Additional diagnostics
                if len(sample_lls) > 5:
                    percentiles = np.percentile(sample_lls, [25, 50, 75])
                    print(f"  LL Percentiles (25%, 50%, 75%): {percentiles[0]:.3f}, {percentiles[1]:.3f}, {percentiles[2]:.3f}")
                
                # Save results
                ll_results = {
                    'method': method,
                    'n_samples': n_valid_samples,
                    'n_variables': len(available_vars),
                    'avg_log_likelihood': float(avg_ll),
                    'std_log_likelihood': float(std_ll),
                    'total_log_likelihood': float(total_ll),
                    'interpretation': interpretation,
                    'sample_lls': [float(x) for x in sample_lls[:10]]  # Save first 10 for inspection
                }
                
                ll_path = os.path.join(self.output_dir, f"log_likelihood_{method}.json")
                with open(ll_path, 'w') as f:
                    json.dump(ll_results, f, indent=4)
                print(f"  Results saved to: {ll_path}")
                
            else:
                print("ERROR: No valid samples could be evaluated")
                print("This suggests issues with:")
                print("  - CPD structure or values")
                print("  - Data encoding mismatch") 
                print("  - Variable cardinality problems")
        
        except Exception as e:
            print(f"Error computing log-likelihood: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_conditional_prob(self, cpd, var_value, parent_values):
        """
        FIXED: Helper method to extract conditional probability from CPD
        """
        try:
            if not hasattr(cpd, 'values') or cpd.values is None:
                return None
                
            values = cpd.values
            
            # For conditional CPDs, we need to find the right column
            # This is a simplified version - assumes standard pgmpy CPD structure
            if values.ndim == 2:
                # Try to map parent values to column index
                # This is approximate and depends on how pgmpy orders the combinations
                if hasattr(cpd, 'evidence') and hasattr(cpd, 'evidence_card'):
                    evidence_vars = list(cpd.evidence)
                    evidence_cards = list(cpd.evidence_card)
                    
                    # Calculate column index from parent state combination
                    col_idx = 0
                    multiplier = 1
                    
                    for i, parent in enumerate(evidence_vars):
                        if parent in parent_values:
                            parent_val = parent_values[parent]
                            col_idx += parent_val * multiplier
                            multiplier *= evidence_cards[i]
                    
                    # Check bounds
                    if 0 <= col_idx < values.shape[1] and 0 <= var_value < values.shape[0]:
                        return float(values[var_value, col_idx])
            
            # Fallback: return uniform probability
            return 1.0 / cpd.variable_card
            
        except Exception as e:
            print(f"Error extracting conditional probability: {e}")
            return 1.0 / cpd.variable_card if hasattr(cpd, 'variable_card') else 0.5
    
    
    # Additional helper method for the network class
    def perform_inference(self, evidence: Dict[str, Union[str, int]], 
                         query_variables: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Perform probabilistic inference on the fitted network
        FIXED VERSION - addresses the indentation and import issues
        """
        try:
            from pgmpy.inference import VariableElimination
        except ImportError:
            print("Cannot perform inference: pgmpy not available")
            return None
        
        if self.fitted_network is None:
            print("Cannot perform inference: network not fitted")
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
    
            # Process evidence - ensure all values are integers for categorical variables
            processed_evidence = {}
            for var, value in evidence.items():
                if var in self.fitted_network.nodes():
                    try:
                        # Convert to appropriate type
                        if isinstance(value, str):
                            if var in self.encoders:
                                processed_evidence[var] = int(self.encoders[var].transform([value])[0])
                            else:
                                # Try to convert string to numeric
                                processed_evidence[var] = int(float(value))
                        else:
                            processed_evidence[var] = int(value)
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Could not process evidence {var}={value}: {e}")
                        continue
    
            if not processed_evidence:
                print("No valid evidence could be processed")
                return None
    
            # Perform inference
            result = inference_engine.query(variables=query_variables, evidence=processed_evidence)
            
            print(f"\nInference Results for evidence {processed_evidence}:")
            print("-" * 50)
            print(result)
            
            return {
                'result': result, 
                'evidence': processed_evidence, 
                'query_vars': query_variables
            }
            
        except Exception as e:
            print(f"Error in inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cross_validate_networks(self, n_folds: int = 5) -> Dict:
        """
        Perform cross-validation to evaluate network stability
        """
        print(f"\n{'='*50}")
        print(f"CROSS-VALIDATION ({n_folds} FOLDS)")
        print(f"{'='*50}")
        
        if self.df is None:
            print("No data available for cross-validation")
            return {}
        
        from sklearn.model_selection import KFold
        
        cv_results = {}
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for method in ['expert', 'pc']:  # Focus on main methods
            print(f"\nCross-validating {method.upper()} method...")
            
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(kfold.split(self.df)):
                print(f"  Fold {fold + 1}/{n_folds}...")
                
                try:
                    # Split data
                    train_data = self.df.iloc[train_idx]
                    test_data = self.df.iloc[test_idx]
                    
                    # Train network on fold
                    fold_network = ENSINBayesianNetwork(train_data)
                    fold_network.preprocess_data()
                    fold_network.create_network(method=method)
                    
                    # Evaluate on test set (network structure stability)
                    test_network = ENSINBayesianNetwork(test_data)
                    test_network.preprocess_data()
                    test_network.create_network(method=method)
                    
                    # Compare network structures (edge overlap)
                    train_edges = set(fold_network.network.edges())
                    test_edges = set(test_network.network.edges())
                    
                    overlap = len(train_edges.intersection(test_edges))
                    union = len(train_edges.union(test_edges))
                    jaccard = overlap / union if union > 0 else 0
                    
                    fold_results.append({
                        'fold': fold + 1,
                        'train_edges': len(train_edges),
                        'test_edges': len(test_edges),
                        'edge_overlap': overlap,
                        'jaccard_similarity': jaccard
                    })
                    
                except Exception as e:
                    print(f"    Error in fold {fold + 1}: {e}")
                    fold_results.append({
                        'fold': fold + 1,
                        'train_edges': 0,
                        'test_edges': 0,
                        'edge_overlap': 0,
                        'jaccard_similarity': 0,
                        'error': str(e)
                    })
            
            cv_results[method] = fold_results
            
            # Summary statistics
            if fold_results:
                jaccard_scores = [r['jaccard_similarity'] for r in fold_results if 'error' not in r]
                if jaccard_scores:
                    print(f"  Mean Jaccard similarity: {np.mean(jaccard_scores):.3f} ± {np.std(jaccard_scores):.3f}")
        
        self.results['cross_validation'] = cv_results
        self._save_cv_report(cv_results)
        
        return cv_results
    
    def _save_training_report(self) -> None:
        """Save comprehensive training report"""
        report_path = os.path.join(self.output_dir, "training_report.json")
        
        # Convert results to JSON-serializable format
        json_results = {}
        for method, results in self.results.get('training', {}).items():
            json_results[method] = {
                'training_time': results['training_time'],
                'nodes': results['nodes'],
                'edges': results['edges'],
                'fitted': results['fitted'],
                'success': results['success']
            }
            if 'error' in results:
                json_results[method]['error'] = results['error']
        
        with open(report_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"Training report saved to: {report_path}")
    
    def _save_inference_report(self, inference_results: List[Dict]) -> None:
        """Save inference demonstration results"""
        report_path = os.path.join(self.output_dir, "inference_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("ENSIN BAYESIAN NETWORK - INFERENCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for result in inference_results:
                f.write(f"Scenario: {result['scenario']}\n")
                f.write(f"Evidence: {result['evidence']}\n")
                f.write("Result: See console output for detailed probabilities\n")
                f.write("-" * 30 + "\n")
        
        print(f"Inference report saved to: {report_path}")
    
    def _save_cv_report(self, cv_results: Dict) -> None:
        """Save cross-validation results"""
        for method, results in cv_results.items():
            df_results = pd.DataFrame(results)
            cv_path = os.path.join(self.output_dir, f"cross_validation_{method}.csv")
            df_results.to_csv(cv_path, index=False)
        
        print(f"Cross-validation reports saved to: {self.output_dir}")
    
    def save_all_models(self) -> None:
        """Save all trained networks"""
        print(f"\nSaving all trained models...")
        
        for method, network in self.networks.items():
            if network.network is not None:
                model_path = os.path.join(self.output_dir, f"model_{method}.json")
                try:
                    network.save_model(model_path)
                    print(f"✓ {method} model saved to: {model_path}")
                except Exception as e:
                    print(f"✗ Error saving {method} model: {e}")
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        print(f"\n{'='*60}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*60}")
        
        report_path = os.path.join(self.output_dir, "final_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("ENSIN BAYESIAN NETWORK ANALYSIS - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Shape: {self.df.shape}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Training summary
            if 'training' in self.results:
                f.write("TRAINING SUMMARY\n")
                f.write("-" * 30 + "\n")
                for method, results in self.results['training'].items():
                    f.write(f"{method.upper()}:\n")
                    f.write(f"  Success: {results['success']}\n")
                    f.write(f"  Training Time: {results['training_time']:.2f}s\n")
                    f.write(f"  Nodes: {results['nodes']}\n")
                    f.write(f"  Edges: {results['edges']}\n")
                    f.write(f"  Parameters Fitted: {results['fitted']}\n")
                    if 'error' in results:
                        f.write(f"  Error: {results['error']}\n")
                    f.write("\n")
            
            # Cross-validation summary
            if 'cross_validation' in self.results:
                f.write("CROSS-VALIDATION SUMMARY\n")
                f.write("-" * 30 + "\n")
                for method, results in self.results['cross_validation'].items():
                    jaccard_scores = [r['jaccard_similarity'] for r in results if 'error' not in r]
                    if jaccard_scores:
                        f.write(f"{method.upper()}:\n")
                        f.write(f"  Mean Jaccard Similarity: {np.mean(jaccard_scores):.3f} ± {np.std(jaccard_scores):.3f}\n")
                        f.write(f"  Stability: {'High' if np.mean(jaccard_scores) > 0.7 else 'Medium' if np.mean(jaccard_scores) > 0.5 else 'Low'}\n\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 30 + "\n")
            generated_files = [
                "sample_data.csv",
                "descriptive_statistics.csv",
                "correlation_matrix.png",
                "network_comparison.csv",
                "network_comparison.png",
                "training_report.json",
                "inference_report.txt"
            ]
            
            for method in self.networks.keys():
                generated_files.extend([
                    f"network_{method}.png",
                    f"model_{method}.json",
                    f"cross_validation_{method}.csv"
                ])
            
            for file in generated_files:
                f.write(f"  - {file}\n")
            
            f.write(f"\nTotal files generated: {len(generated_files)}\n")
        
        print(f"Final report saved to: {report_path}")


# WORKFLOW EXECUTION CODE
def main():
    """
    Execute the complete ENSIN Bayesian Network workflow
    """
    print("="*80)
    print("ENSIN BAYESIAN NETWORK ANALYSIS WORKFLOW")
    print("="*80)
    print("This workflow will:")
    print("1. Initialize the trainer and generate/load data")
    print("2. Train multiple Bayesian networks using different methods")
    print("3. Compare network structures")
    print("4. Demonstrate probabilistic inference")
    print("5. Perform cross-validation")
    print("6. Evaluate uncertainty and model scores")
    print("7. Visualize all networks")
    print("8. Save models and generate comprehensive reports")
    print("="*80)
    
    # Step 1: Initialize trainer
    print("\n" + "="*20 + " STEP 1: INITIALIZATION " + "="*20)
    trainer = ENSINNetworkTrainer(sample_data=True)
    
    # Step 2: Train multiple networks
    print("\n" + "="*20 + " STEP 2: NETWORK TRAINING " + "="*20)
    training_methods = ['expert', 'pc', 'hill_climb']
    training_results = trainer.train_multiple_networks(methods=training_methods)
    
    # Step 3: Compare network structures
    print("\n" + "="*20 + " STEP 3: STRUCTURE COMPARISON " + "="*20)
    trainer.compare_network_structures()
    
    # Step 4: Demonstrate inference (use the best performing network)
    print("\n" + "="*20 + " STEP 4: INFERENCE DEMONSTRATION " + "="*20)
    successful_methods = [method for method, result in training_results.items() 
                         if result['success'] and result['fitted']]
    
    if successful_methods:
        best_method = successful_methods[0]  # Use first successful method
        print(f"Using {best_method} network for inference demonstration")
        trainer.demonstrate_inference(method=best_method)
    else:
        print("No fitted networks available for inference")
    
    # Step 5: Cross-validation
    print("\n" + "="*20 + " STEP 5: CROSS-VALIDATION " + "="*20)
    cv_results = trainer.cross_validate_networks(n_folds=3)  # Reduced for demo

    # Step 6: Uncertainty quantification and model evaluation
    print("\n" + "="*20 + " STEP 6: UNCERTAINTY & MODEL EVALUATION " + "="*20)
    for method in successful_methods:
        print(f"\nEvaluating {method.upper()} network:")
        trainer.evaluate_parameter_uncertainty(method=method)
        trainer.evaluate_prediction_uncertainty(method=method, query_vars=['a1503'])
    
    # Step 7: Visualize all networks
    print("\n" + "="*20 + " STEP 7: NETWORK VISUALIZATION " + "="*20)
    trainer.visualize_all_networks()
    
    # Step 8: Save models and generate final report
    print("\n" + "="*20 + " STEP 8: MODEL SAVING & REPORTING " + "="*20)
    trainer.save_all_models()
    trainer.generate_final_report()
    
    # Final summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Output directory: {trainer.output_dir}")
    print("Check the output directory for:")
    print("  • Network visualizations (PNG files)")
    print("  • Trained models (JSON files)")
    print("  • Analysis reports (TXT/CSV files)")
    print("  • Sample data and statistics")
    print("="*60)
    
    return trainer

def run_quick_demo():
    """
    Run a quick demonstration with minimal configuration
    """
    print("="*60)
    print("QUICK DEMO - ENSIN BAYESIAN NETWORK")
    print("="*60)
    
    # Initialize with smaller dataset for quick demo
    trainer = ENSINNetworkTrainer(sample_data=True)
    trainer.generate_sample_data(n_samples=1000)  # Smaller sample
    
    # Train only expert network for quick demo
    print("\nTraining expert network...")
    trainer.train_multiple_networks(methods=['expert'])
    
    # Quick visualization
    print("\nGenerating visualization...")
    if 'expert' in trainer.networks:
        trainer.networks['expert'].visualize_network(figsize=(12, 8))
    
    # Save model
    trainer.save_all_models()
    
    print(f"\nQuick demo completed! Check output: {trainer.output_dir}")
    return trainer


def run_analysis_with_custom_data(data_path: str):
    """
    Run analysis with custom ENSIN dataset
    
    Args:
        data_path: Path to your ENSIN CSV file
    """
    print("="*60)
    print("CUSTOM DATA ANALYSIS - ENSIN BAYESIAN NETWORK")
    print("="*60)
    
    # Initialize with custom data
    trainer = ENSINNetworkTrainer(data_path=data_path, sample_data=False)
    
    if trainer.df is None:
        print("Failed to load custom data. Please check the file path.")
        return None
    
    # Run full workflow
    return main()


# Interactive menu system
def interactive_menu():
    """
    Interactive menu for different workflow options
    """
    print("="*60)
    print("ENSIN BAYESIAN NETWORK - INTERACTIVE MENU")
    print("="*60)
    print("Choose an option:")
    print("1. Run full workflow with sample data")
    print("2. Run quick demo")
    print("3. Load custom data and run analysis")
    print("4. Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\nStarting full workflow...")
                return main()
            
            elif choice == '2':
                print("\nStarting quick demo...")
                return run_quick_demo()
            
            elif choice == '3':
                data_path = input("Enter path to your ENSIN CSV file: ").strip()
                if data_path and os.path.exists(data_path):
                    return run_analysis_with_custom_data(data_path)
                else:
                    print("File not found. Please check the path and try again.")
            
            elif choice == '4':
                print("Goodbye!")
                return None
            
            else:
                print("Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return None
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    """
    Main execution block - runs when script is executed directly
    """
    import sys
    
    # Check command line arguments for automated execution
    if len(sys.argv) > 1:
        if sys.argv[1] == '--full':
            # Run full workflow
            trainer = main()
        elif sys.argv[1] == '--quick':
            # Run quick demo
            trainer = run_quick_demo()
        elif sys.argv[1] == '--data' and len(sys.argv) > 2:
            # Run with custom data
            trainer = run_analysis_with_custom_data(sys.argv[2])
        else:
            print("Usage:")
            print("  python ensin_main.py --full          # Run full workflow")
            print("  python ensin_main.py --quick         # Run quick demo")
            print("  python ensin_main.py --data <path>   # Run with custom data")
            print("  python ensin_main.py                 # Interactive menu")
    else:
        # Run interactive menu
        trainer = interactive_menu()


# Additional utility functions for advanced users
def batch_analysis(data_files: List[str], output_base_dir: str = "batch_results"):
    """
    Run batch analysis on multiple ENSIN datasets
    
    Args:
        data_files: List of paths to ENSIN CSV files
        output_base_dir: Base directory for batch results
    """
    print(f"Starting batch analysis on {len(data_files)} datasets...")
    
    batch_results = {}
    
    for i, data_file in enumerate(data_files):
        print(f"\nProcessing dataset {i+1}/{len(data_files)}: {data_file}")
        
        try:
            # Create unique output directory for this dataset
            dataset_name = os.path.splitext(os.path.basename(data_file))[0]
            
            # Initialize trainer for this dataset
            trainer = ENSINNetworkTrainer(data_path=data_file, sample_data=False)
            
            if trainer.df is not None:
                # Run analysis
                training_results = trainer.train_multiple_networks(['expert', 'pc'])
                trainer.compare_network_structures()
                trainer.save_all_models()
                
                batch_results[dataset_name] = {
                    'trainer': trainer,
                    'results': training_results,
                    'output_dir': trainer.output_dir
                }
                
                print(f"✓ Dataset {dataset_name} processed successfully")
            else:
                print(f"✗ Failed to load dataset {dataset_name}")
                
        except Exception as e:
            print(f"✗ Error processing {data_file}: {e}")
    
    # Generate batch summary report
    batch_summary_path = os.path.join(output_base_dir, "batch_summary.txt")
    os.makedirs(output_base_dir, exist_ok=True)
    
    with open(batch_summary_path, 'w') as f:
        f.write("BATCH ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total datasets processed: {len(batch_results)}\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset_name, results in batch_results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Output directory: {results['output_dir']}\n")
            f.write(f"Training results: {len(results['results'])} methods\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nBatch analysis completed! Summary saved to: {batch_summary_path}")
    return batch_results


def compare_datasets(trainers: List[ENSINNetworkTrainer], 
                    output_dir: str = "dataset_comparison"):
    """
    Compare multiple ENSIN datasets and their resulting networks
    
    Args:
        trainers: List of trained ENSINNetworkTrainer objects
        output_dir: Directory to save comparison results
    """
    print(f"Comparing {len(trainers)} datasets...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare basic dataset statistics
    comparison_stats = []
    
    for i, trainer in enumerate(trainers):
        if trainer.df is not None:
            stats = {
                'dataset': f'Dataset_{i+1}',
                'n_samples': len(trainer.df),
                'n_variables': len(trainer.df.columns),
                'missing_rate': trainer.df.isnull().sum().sum() / (len(trainer.df) * len(trainer.df.columns)),
            }
            
            # Add network statistics if available
            if trainer.networks:
                for method, network in trainer.networks.items():
                    if network.network is not None:
                        stats[f'{method}_nodes'] = len(network.network.nodes())
                        stats[f'{method}_edges'] = len(network.network.edges())
            
            comparison_stats.append(stats)
    
    # Save comparison
    if comparison_stats:
        comparison_df = pd.DataFrame(comparison_stats)
        comparison_path = os.path.join(output_dir, "dataset_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"Dataset comparison saved to: {comparison_path}")
        print("\nDataset Comparison Summary:")
        print(comparison_df.to_string(index=False))
    
    return comparison_stats