#!/usr/bin/env python3

import os
import sys
import time
import warnings
from pathlib import Path
import sklearn

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom classes
try:
    from src.networkTrainer import ENSINNetworkTrainer, main, run_quick_demo, interactive_menu
    from src.networkTrainer import run_analysis_with_custom_data, batch_analysis, compare_datasets
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure 'ensin_bayesian_network.py' and 'ensin_main_trainer.py' are in the same directory")
    IMPORTS_SUCCESS = False

warnings.filterwarnings('ignore')


def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy', 
        'networkx': 'pip install networkx',
        'matplotlib': 'pip install matplotlib',
        'seaborn': 'pip install seaborn',
        'scikit-learn': 'pip install scikit-learn',
        'scipy': 'pip install scipy'
    }
    
    optional_packages = {
        'pgmpy': 'pip install pgmpy  # For advanced Bayesian network features'
    }
    
    missing_required = []
    missing_optional = []
    
    for package, install_cmd in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_required.append((package, install_cmd))
    
    for package, install_cmd in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, install_cmd))
    
    if missing_required:
        print("ERROR: Missing required packages:")
        for package, cmd in missing_required:
            print(f"  {cmd}")
        return False
    
    if missing_optional:
        print("WARNING: Missing optional packages (reduced functionality):")
        for package, cmd in missing_optional:
            print(f"  {cmd}")
    
    print("✓ All required packages available")
    return True


def example_1_full_workflow():
    print("\n" + "="*60)
    print("EXAMPLE 1: FULL WORKFLOW WITH SAMPLE DATA")
    print("="*60)
    
    print("This will:")
    print("• Train 3 different Bayesian networks (Expert, PC, Hill Climb)")
    print("• Compare network structures")
    print("• Demonstrate probabilistic inference")
    print("• Perform 3-fold cross-validation")
    print("• Generate comprehensive visualizations and reports")
    print("\nExpected runtime: 2-5 minutes")
    
    input("Press Enter to continue...")
    
    start_time = time.time()
    
    # Run the main workflow
    trainer = main()
    
    end_time = time.time()
    
    print(f"\n✓ Full workflow completed in {end_time - start_time:.1f} seconds")
    print(f"Results saved to: {trainer.output_dir}")
    
    return trainer


def example_2_quick_demo():
    """
    Example 2: Quick demonstration
    Fast demo with minimal configuration
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: QUICK DEMO")
    print("="*60)
    
    print("This quick demo will:")
    print("• Train one expert-based network")
    print("• Create basic visualization")
    print("\nExpected runtime: 30 seconds")
    
    input("Press Enter to continue...")
    
    start_time = time.time()
    
    # Run quick demo
    trainer = run_quick_demo()
    
    end_time = time.time()
    
    print(f"\n✓ Quick demo completed in {end_time - start_time:.1f} seconds")
    if trainer:
        print(f"Results saved to: {trainer.output_dir}")
    
    return trainer


def example_3_step_by_step():
    """
    Example 3: Step-by-step workflow
    Manual control over each step for educational purposes
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: STEP-BY-STEP WORKFLOW")
    print("="*60)
    
    print("This example demonstrates manual control over each workflow step.")
    print("You'll see detailed output for each step.\n")
    
    input("Press Enter to start step-by-step workflow...")
    
    # Step 1: Initialize
    print("\n[STEP 1] Initializing trainer...")
    trainer = ENSINNetworkTrainer(sample_data=True)
    trainer.generate_sample_data(n_samples=2000)
    print("✓ Initialization complete")
    
    input("\nPress Enter for Step 2...")
    
    # Step 2: Train networks
    print("\n[STEP 2] Training networks...")
    methods_to_try = ['expert', 'pc']  # Reduced set for demo
    
    for method in methods_to_try:
        print(f"\n  Training {method.upper()} network...")
        result = trainer.train_multiple_networks(methods=[method])
        if result[method]['success']:
            print(f"  ✓ {method} network trained successfully")
        else:
            print(f"  ✗ {method} network failed")
    
    input("\nPress Enter for Step 3...")
    
    # Step 3: Compare structures
    print("\n[STEP 3] Comparing network structures...")
    trainer.compare_network_structures()
    print("✓ Structure comparison complete")
    
    input("\nPress Enter for Step 4...")
    
    # Step 4: Inference demo
    print("\n[STEP 4] Demonstrating inference...")
    successful_methods = [method for method, network in trainer.networks.items() 
                         if network.fitted_network is not None]
    
    if successful_methods:
        print(f"Using {successful_methods[0]} network for inference...")
        trainer.demonstrate_inference(method=successful_methods[0])
    else:
        print("No fitted networks available for inference")
    
    input("\nPress Enter for Step 5...")
    
    # Step 5: Visualization
    print("\n[STEP 5] Creating visualizations...")
    trainer.visualize_all_networks()
    print("✓ Visualizations complete")
    
    input("\nPress Enter for final step...")
    
    # Step 6: Save and report
    print("\n[STEP 6] Saving models and generating reports...")
    trainer.save_all_models()
    trainer.generate_final_report()
    print("✓ All results saved")
    
    print(f"\n✓ Step-by-step workflow completed!")
    print(f"Results saved to: {trainer.output_dir}")
    
    return trainer


def example_4_custom_analysis():
    """
    Example 4: Custom analysis configuration
    Demonstrate advanced customization options
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: CUSTOM ANALYSIS CONFIGURATION")
    print("="*60)
    
    print("This example shows how to customize the analysis:")
    print("• Specific variable selection")
    print("• Custom inference scenarios")
    
    input("Press Enter to continue...")
    
    # Initialize with custom settings
    trainer = ENSINNetworkTrainer(sample_data=True)
    
    # Custom data generation
    print("\nGenerating custom dataset...")
    trainer.generate_sample_data(n_samples=3000)
    
    # Create network with custom preprocessing
    print("\nCreating custom network...")
    network = ENSINNetworkTrainer.ENSINBayesianNetwork(trainer.df)
    
    # Custom discretization
    print("Applying custom discretization (uniform, 4 bins)...")
    network.preprocess_data(discretization_method='uniform', n_bins=4)
    
    # Create expert network
    print("Building expert network...")
    network.create_network(method='expert')
    
    # Custom inference scenarios
    if network.network:
        print("Fitting parameters...")
        fitted = network.fit_parameters(method='mle')
        
        if fitted:
            print("\nRunning custom inference scenarios...")
            
            custom_scenarios = [
                {
                    'name': 'High SES Individual',
                    'evidence': {'estrato': 3, 'edad': 1},  # High SES, middle-aged
                },
                {
                    'name': 'Low SES with Poor Diet',
                    'evidence': {'estrato': 0, 'a0701p': 0},  # Low SES, poor fruit intake
                }
            ]
            
            for scenario in custom_scenarios:
                print(f"\nScenario: {scenario['name']}")
                result = network.perform_inference(
                    evidence=scenario['evidence'],
                    query_variables=['a1503', 'a1210', 'a0301']
                )
    
    # Visualization
    print("\nCreating custom visualization...")
    if network.network:
        network.visualize_network(figsize=(14, 10), layout='spring')
    
    print("✓ Custom analysis completed!")
    return network


def example_5_batch_processing():
    """
    Example 5: Batch processing simulation
    Simulate processing multiple datasets
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: BATCH PROCESSING SIMULATION")
    print("="*60)
    
    print("This example simulates batch processing of multiple datasets:")
    print("• Processes each independently")
    print("• Compares results across datasets")
    
    input("Press Enter to continue...")
    
    # Create multiple trainers with different data characteristics
    trainers = []
    dataset_configs = [
        {'n_samples': 1500, 'name': 'Small_Urban'},
        {'n_samples': 2500, 'name': 'Medium_Rural'}, 
        {'n_samples': 2000, 'name': 'Mixed_Population'}
    ]
    
    print("\nCreating and processing datasets...")
    for i, config in enumerate(dataset_configs):
        print(f"\n  Processing {config['name']} dataset...")
        
        # Create trainer
        trainer = ENSINNetworkTrainer(sample_data=True)
        trainer.generate_sample_data(n_samples=config['n_samples'])
        
        # Train networks
        results = trainer.train_multiple_networks(methods=['expert'])
        
        trainers.append(trainer)
        print(f"  ✓ {config['name']} processed")
    
    # Compare datasets
    print("\nComparing datasets...")
    comparison_results = compare_datasets(trainers, output_dir="batch_comparison")
    
    print("✓ Batch processing completed!")
    print("Check 'batch_comparison' directory for comparison results")
    
    return trainers


def interactive_examples():
    """Interactive menu for different examples"""
    examples = {
        '1': ('Full Workflow Demo', example_1_full_workflow),
        '2': ('Quick Demo', example_2_quick_demo), 
        '3': ('Step-by-Step Tutorial', example_3_step_by_step),
        '4': ('Custom Analysis', example_4_custom_analysis),
        '5': ('Batch Processing', example_5_batch_processing)
    }
    
    while True:
        print("\n" + "="*60)
        print("ENSIN BAYESIAN NETWORK - EXAMPLE WORKFLOWS")
        print("="*60)
        print("Choose an example to run:")
        
        for key, (name, _) in examples.items():
            print(f"{key}. {name}")
        
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            
            elif choice in examples:
                name, func = examples[choice]
                print(f"\nStarting: {name}")
                result = func()
                
                print(f"\n{'='*40}")
                print(f"EXAMPLE COMPLETED: {name}")
                print(f"{'='*40}")
                
                input("\nPress Enter to return to menu...")
            
            else:
                print("Invalid choice. Please enter 0-5.")
        
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            input("Press Enter to continue...")


def run_performance_benchmark():
    """
    Run performance benchmarks for different dataset sizes
    """
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    sizes = [500, 1000, 2000, 5000]
    methods = ['expert', 'pc']
    
    results = []
    
    for size in sizes:
        print(f"\nBenchmarking with {size} samples...")
        
        for method in methods:
            print(f"  Testing {method} method...")
            
            start_time = time.time()
            
            try:
                # Create and train network
                trainer = ENSINNetworkTrainer(sample_data=True)
                trainer.generate_sample_data(n_samples=size)
                
                training_result = trainer.train_multiple_networks(methods=[method])
                
                end_time = time.time()
                
                results.append({
                    'dataset_size': size,
                    'method': method,
                    'time_seconds': end_time - start_time,
                    'success': training_result[method]['success'],
                    'nodes': training_result[method]['nodes'],
                    'edges': training_result[method]['edges']
                })
                
                print(f"    ✓ Completed in {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                results.append({
                    'dataset_size': size,
                    'method': method,
                    'time_seconds': -1,
                    'success': False,
                    'nodes': 0,
                    'edges': 0,
                    'error': str(e)
                })
    
    # Display results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    import pandas as pd
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success'] == True]
    
    if len(successful_results) > 0:
        print("\nSuccessful runs:")
        print(successful_results[['dataset_size', 'method', 'time_seconds', 'nodes', 'edges']].to_string(index=False))
        
        # Save benchmark results
        benchmark_dir = "performance_benchmark"
        os.makedirs(benchmark_dir, exist_ok=True)
        results_df.to_csv(os.path.join(benchmark_dir, "benchmark_results.csv"), index=False)
        print(f"\nBenchmark results saved to: {benchmark_dir}/benchmark_results.csv")
    else:
        print("No successful benchmark runs")
    
    return results


def main_execution():
    """Main execution function with command line argument handling"""
    
    if not IMPORTS_SUCCESS:
        print("Cannot continue due to import errors.")
        return

    '''
    if not check_requirements():
        print("\nPlease install missing required packages and try again.")
        return
    '''
    
    print("\nENSIN Bayesian Network Analysis Tool Ready!")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--full':
            print("Running full workflow...")
            return example_1_full_workflow()
        
        elif arg == '--quick':
            print("Running quick demo...")
            return example_2_quick_demo()
        
        elif arg == '--step':
            print("Running step-by-step tutorial...")
            return example_3_step_by_step()
        
        elif arg == '--custom':
            print("Running custom analysis...")
            return example_4_custom_analysis()
        
        elif arg == '--batch':
            print("Running batch processing demo...")
            return example_5_batch_processing()
        
        elif arg == '--benchmark':
            print("Running performance benchmark...")
            return run_performance_benchmark()
        
        elif arg == '--data' and len(sys.argv) > 2:
            data_path = sys.argv[2]
            if os.path.exists(data_path):
                print(f"Running analysis with custom data: {data_path}")
                return run_analysis_with_custom_data(data_path)
            else:
                print(f"Error: Data file not found: {data_path}")
                return None
        
        elif arg in ['--help', '-h']:
            print_help()
            return None
        
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return None
    
    else:
        # No arguments - run interactive menu
        return interactive_examples()


def print_help():
    """Print help information"""
    help_text = """
ENSIN Bayesian Network Analysis Tool

USAGE:
    python run_ensin_workflow.py [OPTIONS]

OPTIONS:
    --full          Run complete workflow with sample data (5-10 min)
    --quick         Run quick demo with minimal features (30 sec)
    --step          Run step-by-step tutorial with pauses
    --custom        Run custom analysis with advanced options
    --batch         Run batch processing simulation
    --benchmark     Run performance benchmark tests
    --data <file>   Run analysis with custom ENSIN CSV file
    --help, -h      Show this help message

INTERACTIVE MODE:
    python run_ensin_workflow.py
    (Runs interactive menu with all examples)

EXAMPLES:
    # Quick demo
    python run_ensin_workflow.py --quick
    
    # Full analysis
    python run_ensin_workflow.py --full
    
    # Custom data analysis
    python run_ensin_workflow.py --data my_ensin_data.csv
    
    # Interactive mode
    python run_ensin_workflow.py

REQUIREMENTS:
    Required: pandas, numpy, networkx, matplotlib, seaborn, scikit-learn, scipy
    Optional: pgmpy (for advanced Bayesian network features)

OUTPUT:
    All results are saved to timestamped directories:
    - Network visualizations (.png files)
    - Trained models (.json files) 
    - Analysis reports (.txt/.csv files)
    - Sample/processed data (.csv files)

For more information, visit: https://github.com/your-repo/ensin-bayesian-network
"""
    print(help_text)


def validate_environment():
    """
    Comprehensive environment validation
    """
    print("Validating environment...")
    
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append("Python 3.7 or higher required")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages with version info
    package_checks = {
        'pandas': {'min_version': '1.0', 'import_name': 'pandas'},
        'numpy': {'min_version': '1.18', 'import_name': 'numpy'},
        'networkx': {'min_version': '2.4', 'import_name': 'networkx'},
        'matplotlib': {'min_version': '3.1', 'import_name': 'matplotlib'},
        'seaborn': {'min_version': '0.10', 'import_name': 'seaborn'},
        'scikit-learn': {'min_version': '0.22', 'import_name': 'sklearn'},
        'scipy': {'min_version': '1.4', 'import_name': 'scipy'}
    }
    
    for package, info in package_checks.items():
        try:
            module = __import__(info['import_name'])
            if hasattr(module, '__version__'):
                print(f"✓ {package} {module.__version__}")
            else:
                print(f"✓ {package} (version unknown)")
        except ImportError:
            issues.append(f"Missing required package: {package}")
    
    # Check optional packages
    optional_packages = {
        'pgmpy': 'Advanced Bayesian network structure learning and inference'
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            warnings.append(f"Optional package missing: {package} - {description}")
    
    # Check write permissions
    try:
        test_dir = "test_write_permissions"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✓ Write permissions OK")
    except Exception:
        issues.append("No write permissions in current directory")
    
    # Display results
    if issues:
        print(f"\n❌ {len(issues)} critical issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    print(f"\n✅ Environment validation {'passed' if not issues else 'failed'}")
    return len(issues) == 0


def create_sample_config():
    """
    Create sample configuration files for advanced users
    """
    config_dir = "ensin_configs"
    os.makedirs(config_dir, exist_ok=True)
    
    # Sample network configuration
    network_config = {
        "preprocessing": {
            "discretization_method": "quantile",
            "n_bins": 3,
            "handle_missing": "median"
        },
        "structure_learning": {
            "methods": ["expert", "pc", "hill_climb"],
            "pc_significance": 0.05,
            "hill_climb_scoring": "bic"
        },
        "parameter_estimation": {
            "method": "mle",
            "bayesian_prior": "uniform"
        },
        "inference": {
            "default_queries": ["a1503", "a1210", "a0301"],
            "evidence_scenarios": [
                {"name": "young_female", "evidence": {"edad": 0, "sexo": 0}},
                {"name": "elderly_male", "evidence": {"edad": 2, "sexo": 1}}
            ]
        },
        "visualization": {
            "layout": "hierarchical",
            "figsize": [16, 12],
            "node_size": 1000,
            "save_format": "png"
        }
    }
    
    # Save configuration
    import json
    config_path = os.path.join(config_dir, "network_config.json")
    with open(config_path, 'w') as f:
        json.dump(network_config, f, indent=4)
    
    # Create readme file
    readme_content = """
# ENSIN Bayesian Network Configuration

This directory contains configuration files for customizing the ENSIN analysis.

## Files:
- network_config.json: Main configuration for network analysis
- custom_variables.json: Custom variable definitions and mappings

## Usage:
Modify the configuration files and pass them to the analysis functions:

```python
from ensin_main_trainer import ENSINNetworkTrainer
import json

# Load custom config
with open('ensin_configs/network_config.json', 'r') as f:
    config = json.load(f)

# Use in analysis
trainer = ENSINNetworkTrainer(sample_data=True)
# Apply custom settings based on config
```

## Configuration Options:

### Preprocessing:
- discretization_method: 'quantile', 'uniform', or 'kmeans'
- n_bins: Number of bins for discretization
- handle_missing: 'median', 'mean', 'mode', or 'drop'

### Structure Learning:
- methods: List of methods to try
- pc_significance: Significance level for PC algorithm
- hill_climb_scoring: Scoring method for Hill Climbing

### Visualization:
- layout: 'hierarchical', 'spring', 'circular'
- figsize: [width, height] in inches
- node_size: Size of network nodes
"""
    
    readme_path = os.path.join(config_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Configuration files created in: {config_dir}/")
    print(f"  • network_config.json")
    print(f"  • README.md")
    
    return config_dir


def run_diagnostic_tests():
    """
    Run diagnostic tests to ensure everything works correctly
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS")
    print("="*60)
    
    tests = []
    
    # Test 1: Basic import and initialization
    print("\n[TEST 1] Basic functionality...")
    try:
        trainer = ENSINNetworkTrainer(sample_data=True)
        trainer.generate_sample_data(n_samples=100)
        tests.append(("Basic initialization", True, ""))
        print("✓ PASSED")
    except Exception as e:
        tests.append(("Basic initialization", False, str(e)))
        print(f"✗ FAILED: {e}")
    
    # Test 2: Network creation
    print("\n[TEST 2] Network creation...")
    try:
        from ensin_bayesian_network import ENSINBayesianNetwork
        network = ENSINBayesianNetwork(trainer.df)
        network.preprocess_data()
        network.create_network(method='expert')
        tests.append(("Network creation", True, ""))
        print("✓ PASSED")
    except Exception as e:
        tests.append(("Network creation", False, str(e)))
        print(f"✗ FAILED: {e}")
    
    # Test 3: Visualization
    print("\n[TEST 3] Visualization...")
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Test Plot")
        plt.close()
        tests.append(("Visualization", True, ""))
        print("✓ PASSED")
    except Exception as e:
        tests.append(("Visualization", False, str(e)))
        print(f"✗ FAILED: {e}")
    
    # Test 4: File I/O
    print("\n[TEST 4] File operations...")
    try:
        test_dir = "diagnostic_test_output"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test CSV writing
        import pandas as pd
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        test_df.to_csv(os.path.join(test_dir, "test.csv"), index=False)
        
        # Test JSON writing
        import json
        test_data = {'test': 'data'}
        with open(os.path.join(test_dir, "test.json"), 'w') as f:
            json.dump(test_data, f)
        
        # Clean up
        os.remove(os.path.join(test_dir, "test.csv"))
        os.remove(os.path.join(test_dir, "test.json"))
        os.rmdir(test_dir)
        
        tests.append(("File operations", True, ""))
        print("✓ PASSED")
    except Exception as e:
        tests.append(("File operations", False, str(e)))
        print(f"✗ FAILED: {e}")
    
    # Test 5: Advanced features (pgmpy)
    print("\n[TEST 5] Advanced features (pgmpy)...")
    try:
        import pgmpy
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        
        # Create simple test network
        model = BayesianNetwork([('A', 'B')])
        cpd_a = TabularCPD('A', 2, [[0.3], [0.7]])
        cpd_b = TabularCPD('B', 2, [[0.4, 0.9], [0.6, 0.1]], evidence=['A'], evidence_card=[2])
        model.add_cpds(cpd_a, cpd_b)
        
        tests.append(("Advanced features", True, ""))
        print("✓ PASSED (pgmpy available)")
    except ImportError:
        tests.append(("Advanced features", False, "pgmpy not installed"))
        print("⚠ SKIPPED (pgmpy not available - reduced functionality)")
    except Exception as e:
        tests.append(("Advanced features", False, str(e)))
        print(f"✗ FAILED: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for test in tests if test[1])
    total = len(tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System ready for analysis.")
    else:
        print("⚠️  Some tests failed. Check errors above.")
        print("\nFailed tests:")
        for name, success, error in tests:
            if not success:
                print(f"  • {name}: {error}")
    
    return tests


if __name__ == "__main__":
    """
    Main entry point - handles all execution paths
    """
    try:
        # Print banner
        print("="*80)
        print("ENSIN BAYESIAN NETWORK ANALYSIS TOOLKIT")
        print("Advanced Health & Nutrition Data Analysis")
        print("="*80)
        
        # Validate environment first
        if not validate_environment():
            print("\n❌ Environment validation failed. Please fix issues above.")
            sys.exit(1)
        
        # Handle special commands
        if len(sys.argv) > 1:
            if sys.argv[1] == '--diagnostic':
                run_diagnostic_tests()
                sys.exit(0)
            elif sys.argv[1] == '--create-config':
                create_sample_config()
                sys.exit(0)
        
        # Run main execution
        print("\n🚀 Starting analysis workflow...")
        result = main_execution()
        
        if result:
            print(f"\n🎉 Workflow completed successfully!")
            if hasattr(result, 'output_dir'):
                print(f"📁 Results saved to: {result.output_dir}")
            elif isinstance(result, list):
                print(f"📁 Multiple results generated (check individual output directories)")
        else:
            print("\n⚠️  Workflow completed with warnings or was cancelled.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nFor help, run: python run_ensin_workflow.py --help")
        print("For diagnostics, run: python run_ensin_workflow.py --diagnostic")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("Thank you for using ENSIN Bayesian Network Analysis Toolkit!")
    print("For support: https://github.com/your-repo/ensin-bayesian-network")
    print("="*80)