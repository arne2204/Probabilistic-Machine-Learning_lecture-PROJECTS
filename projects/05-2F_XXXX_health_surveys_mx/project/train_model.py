
def main():
    """
    Main function for ENSIN Layered Bayesian Network Analysis + Inference
    Focus: Diabetes (A0301) and Weight Change (A0107) scenarios
    """
    print("ENSIN Layered Bayesian Network Analysis with Pruning & Inference")
    print("=" * 60)

    inspect_data()
    
    try:
        # Load dataset
        print("Loading data from 'cleaned_dataset.csv'...")
        df = pd.read_csv('project/cleaned_dataset.csv')
        print(f"Data loaded successfully! Shape: {df.shape}")
        print(df.head())
        
        # Initialize ENSIN Bayesian Network
        ensin_bn = ENSINBayesianNetwork(df)
        
        # Preprocess dataset (build encoders)
        # ensin_bn.preprocess_data()
        encoders = ensin_bn.encoders
        
        # Build layered network with pruning
        network = ensin_bn.create_layered_network(
            pruning_method='mutual_info', 
            keep_top_percent=0.25
        )
        if network is None:
            print("Failed to create network.")
            return
        
        # Analyze edge patterns
        ensin_bn.analyze_edge_patterns()
        
        # Fit network parameters
        fitted_network = ensin_bn.fit_network_parameters(method='mle')
        if fitted_network is None or not isinstance(fitted_network, BayesianNetwork):
            print("Network not fitted; cannot run inference.")
            return
        
        network_nodes = set(fitted_network.nodes())
        
        print(f"Dataset shape: {df.shape}")
        print(f"Network nodes: {len(ensin_bn.network.nodes())}")
        print(f"Network edges: {len(ensin_bn.network.edges())}")

        print("Saving model....")
        save_bn_model(fitted_network, "model.json")
        
    except FileNotFoundError:
        print("Error: 'cleaned_dataset.csv' not found!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
