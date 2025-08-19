from causalnex.inference import InferenceEngine

def main():
    # Load your trained Bayesian network
    bn = load_bn("your_network_file.json")  # Replace with your loading method

    # Define your evidence dictionary (all observed variables)
    evidence = {
        'a0104': 0,
        'a0202': 1,
        # Add all other observed variables here
    }

    # Initialize inference engine
    ie = InferenceEngine(bn)

    # Target variables you want distributions for
    targets = ['a0811d', 'a1001a']  # replace with your actual target variable names

    # Compute posterior distributions
    posteriors = {}
    for var in targets:
        posteriors[var] = ie.query()[var].values(evidence=evidence)

    # Print posterior distributions
    for var, dist in posteriors.items():
        print(f"Posterior distribution for {var}:")
        print(dist)
        print()

    return posteriors

if __name__ == "__main__":
    main()