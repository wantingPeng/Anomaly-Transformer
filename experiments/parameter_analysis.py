"""
Parameter Analysis for Anomaly Transformer
This script trains models with different parameter values and visualizes F1 score performance.
Supports analyzing: win_size, k, anormly_ratio
"""

import os
import sys
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# Select which parameter to analyze (choose one: 'win_size', 'k', 'anormly_ratio')
PARAM_TO_ANALYZE = 'k'  # Change this to 'k' or 'anormly_ratio' for other analyses

# Datasets to test
DATASETS = ['contact', 'ring', 'pcb']

# Parameter values to test (modify based on PARAM_TO_ANALYZE)
PARAM_VALUES = {
    'win_size': [10, 20, 30, 40, 50, 60],
    'k': [3, 5, 7, 10, 15],
    'anormly_ratio': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'batch_size': [32, 64, 128, 256, 512]
}

# Fixed parameters (used when not being analyzed)
FIXED_PARAMS = {
    'num_epochs': 10,
    'batch_size': 64,
    'lr': 1e-4,
    'win_size': 30,      # Used when analyzing k or anormly_ratio
    'k': 3,             # Used when analyzing win_size or anormly_ratio
    'anormly_ratio': 3.0 # Used when analyzing win_size or k
}

# Output configuration
OUTPUT_DIR = 'experiments/results'
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {
    'contact': {
        'data_path': 'dataset/dataset/pca_analysis_and_result_1/contact/pca_features_contact.parquet',
        'input_c': 7,
        'output_c': 7,
    },
    'ring': {
        'data_path': 'dataset/dataset/pca_analysis_and_result_1/ring/pca_features_ring.parquet',
        'input_c': 6,
        'output_c': 6,
    },
    'pcb': {
        'data_path': 'dataset/dataset/pca_analysis_and_result_1/pcb/pca_features_pcb.parquet',
        'input_c': 10,
        'output_c': 10,
    }
}

# Plot styling
PLOT_CONFIG = {
    'colors': {'contact': 'red', 'ring': 'green', 'pcb': 'blue'},
    'markers': {'contact': 'o', 'ring': 's', 'pcb': '^'},
    'labels': {'contact': 'Contact', 'ring': 'Ring', 'pcb': 'PCB'}
}

PARAM_DISPLAY_NAMES = {
    'win_size': 'Window Size',
    'k': 'K Parameter',
    'anormly_ratio': 'Anomaly Ratio',
    'batch_size': 'Batch Size'
}

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def get_dataset_config(dataset_name):
    """Get configuration for each dataset."""
    return DATASET_CONFIGS.get(dataset_name)


def run_single_experiment(dataset_name, param_name, param_value, fixed_params):
    """
    Run a single experiment with specified parameter value.
    
    Args:
        dataset_name: Name of the dataset ('contact', 'ring', 'pcb')
        param_name: Name of the parameter being tested ('win_size', 'k', 'anormly_ratio')
        param_value: Value of the parameter to test
        fixed_params: Dictionary of fixed parameters
        
    Returns:
        dict: Results containing precision, recall, f_score, accuracy
    """
    config = get_dataset_config(dataset_name)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create unique model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f'checkpoints_{param_name}_analysis/{dataset_name}_{param_name}{param_value}_{timestamp}'
    
    # Build parameters dictionary
    params = fixed_params.copy()
    params[param_name] = param_value
    
    # Build command
    cmd = [
        'python', 'main.py',
        '--dataset', 'Custom',
        '--data_path', config['data_path'],
        '--num_epochs', str(params['num_epochs']),
        '--win_size', str(params['win_size']),
        '--batch_size', str(params['batch_size']),
        '--mode', 'train',
        '--input_c', str(config['input_c']),
        '--output_c', str(config['output_c']),
        '--lr', str(params['lr']),
        '--k', str(params['k']),
        '--anormly_ratio', str(params['anormly_ratio']),
        '--model_save_path', model_save_path
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {dataset_name} | {param_name}={param_value}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        # Find the latest checkpoint directory
        if not os.path.exists(model_save_path):
            print(f"Warning: Model save path not found: {model_save_path}")
            return None
            
        checkpoint_dirs = [d for d in os.listdir(model_save_path) 
                          if os.path.isdir(os.path.join(model_save_path, d))]
        
        if not checkpoint_dirs:
            print(f"Warning: No checkpoint directory found in {model_save_path}")
            return None
            
        # Get the most recent checkpoint
        checkpoint_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)))
        latest_checkpoint = checkpoint_dirs[-1]
        result_json_path = os.path.join(model_save_path, latest_checkpoint, 'result.json')
        
        # Read results
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as f:
                results = json.load(f)
                return results['summary']
        else:
            print(f"Warning: Result file not found at {result_json_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def run_all_experiments(datasets, param_name, param_values, fixed_params):
    """
    Run experiments for all datasets and parameter values.
    
    Args:
        datasets: List of dataset names
        param_name: Name of the parameter being tested
        param_values: List of parameter values to test
        fixed_params: Dictionary of fixed parameters
        
    Returns:
        dict: Results organized by dataset and parameter value
    """
    results = {dataset: {} for dataset in datasets}
    
    total_experiments = len(datasets) * len(param_values)
    current_experiment = 0
    
    for dataset in datasets:
        for param_value in param_values:
            current_experiment += 1
            print(f"\n{'#'*80}")
            print(f"# Experiment {current_experiment}/{total_experiments}")
            print(f"# Dataset: {dataset}, {param_name}: {param_value}")
            print(f"{'#'*80}\n")
            
            result = run_single_experiment(
                dataset_name=dataset,
                param_name=param_name,
                param_value=param_value,
                fixed_params=fixed_params
            )
            
            if result:
                results[dataset][param_value] = result
                print(f"\n✓ Results: Precision={result['precision']:.4f}, "
                      f"Recall={result['recall']:.4f}, F-score={result['f_score']:.4f}")
            else:
                print(f"\n✗ Failed to get results for {dataset} with {param_name}={param_value}")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_results(results, param_name, output_dir=OUTPUT_DIR):
    """
    Plot F1 scores for different parameter values and datasets.
    
    Args:
        results: Dictionary containing results for each dataset and parameter value
        param_name: Name of the parameter being analyzed
        output_dir: Directory to save output files
    """
    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.style.use(PLOT_STYLE)
    
    colors = PLOT_CONFIG['colors']
    markers = PLOT_CONFIG['markers']
    labels = PLOT_CONFIG['labels']
    
    # Plot each dataset
    for dataset, dataset_results in results.items():
        if not dataset_results:
            continue
            
        # Extract parameter values and f1 scores
        param_vals = sorted(dataset_results.keys())
        f1_scores = [dataset_results[pv]['f_score'] * 100 for pv in param_vals]
        
        # Plot line with markers
        plt.plot(param_vals, f1_scores, 
                marker=markers[dataset], 
                color=colors[dataset], 
                linewidth=2, 
                markersize=8,
                label=labels[dataset])
        
        # Add value annotations next to each point
        for param_val, f1_score in zip(param_vals, f1_scores):
            plt.annotate(f'{f1_score:.1f}', 
                        xy=(param_val, f1_score),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[dataset],
                        weight='bold')
    
    # Customize plot
    param_display_name = PARAM_DISPLAY_NAMES.get(param_name, param_name)
    plt.xlabel(param_display_name, fontsize=12, weight='bold')
    plt.ylabel('F1 Score (%)', fontsize=12, weight='bold')
    plt.title(f'F1 Score vs {param_display_name} for Different Datasets', 
             fontsize=14, weight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    all_param_vals = []
    all_f1_scores = []
    for dataset_results in results.values():
        for pv, metrics in dataset_results.items():
            all_param_vals.append(pv)
            all_f1_scores.append(metrics['f_score'] * 100)
    
    # Ensure x-axis always shows configured parameter values
    configured_param_vals = PARAM_VALUES.get(param_name, [])
    if configured_param_vals:
        # Convert configured values to same type as experiment outputs when possible
        normalized_config_vals = []
        for val in configured_param_vals:
            if isinstance(val, (int, float)):
                normalized_config_vals.append(val)
            else:
                try:
                    # Attempt numeric conversion (covers cases like "10")
                    numeric_val = float(val)
                    # Cast back to int if it represents an integer value
                    if numeric_val.is_integer():
                        numeric_val = int(numeric_val)
                    normalized_config_vals.append(numeric_val)
                except (TypeError, ValueError):
                    normalized_config_vals.append(val)
        all_param_vals.extend(normalized_config_vals)

    if all_param_vals:
        x_min, x_max = min(all_param_vals), max(all_param_vals)
        x_range = x_max - x_min
        plt.xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)
        unique_ticks = sorted(set(all_param_vals))
        plt.xticks(unique_ticks)
    
    if all_f1_scores:
        y_min = max(0, min(all_f1_scores) - 5)
        y_max = min(100, max(all_f1_scores) + 5)
        plt.ylim(y_min, y_max)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, f'{param_name}_f1_scores.png')
    pdf_path = os.path.join(output_dir, f'{param_name}_f1_scores.pdf')
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\n{'='*80}")
    print(f"Plot saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*80}\n")
    
    plt.show()




def save_results_to_json(results, param_name, output_dir=OUTPUT_DIR):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary containing results
        param_name: Name of the parameter being analyzed
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f'{param_name}_results.json')
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to JSON: {json_path}")


def load_results_from_json(param_name, output_dir=OUTPUT_DIR):
    """
    Load results from JSON file.
    
    Args:
        param_name: Name of the parameter being analyzed
        output_dir: Directory containing output files
        
    Returns:
        dict: Loaded results
    """
    json_path = os.path.join(output_dir, f'{param_name}_results.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Convert string keys back to appropriate types
    converted_results = {}
    for dataset, dataset_results in results.items():
        converted_results[dataset] = {}
        for param_val_str, metrics in dataset_results.items():
            # Try to convert to int first, then float, otherwise keep as string
            try:
                param_val = int(param_val_str)
            except ValueError:
                try:
                    param_val = float(param_val_str)
                except ValueError:
                    param_val = param_val_str
            converted_results[dataset][param_val] = metrics
    
    return converted_results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the parameter analysis."""
    
    print("\n" + "="*80)
    print("ANOMALY TRANSFORMER - PARAMETER ANALYSIS")
    print("="*80)
    print(f"Parameter to analyze: {PARAM_TO_ANALYZE}")
    print(f"Parameter values: {PARAM_VALUES[PARAM_TO_ANALYZE]}")
    print(f"Datasets: {DATASETS}")
    print(f"Fixed parameters: {FIXED_PARAMS}")
    print("="*80 + "\n")
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Get parameter values to test
    param_values = PARAM_VALUES[PARAM_TO_ANALYZE]
    
    # Run experiments
    print(f"\nStarting experiments...")
    results = run_all_experiments(
        datasets=DATASETS,
        param_name=PARAM_TO_ANALYZE,
        param_values=param_values,
        fixed_params=FIXED_PARAMS
    )
    
    # Save results
    print(f"\nSaving results...")
    save_results_to_json(results, PARAM_TO_ANALYZE, OUTPUT_DIR)
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_results(results, PARAM_TO_ANALYZE, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"All results saved in: {OUTPUT_DIR}")
    print("="*80 + "\n")


# ============================================================================
# UTILITY: PLOT ONLY MODE
# ============================================================================

def plot_only_mode(param_name):
    """
    Load existing results and regenerate plots only.
    
    Args:
        param_name: Name of the parameter to plot
    """
    print(f"\n{'='*80}")
    print("PLOT ONLY MODE")
    print(f"Loading results for: {param_name}")
    print(f"{'='*80}\n")
    
    try:
        results = load_results_from_json(param_name, OUTPUT_DIR)
        print("✓ Results loaded successfully\n")
        
        plot_results(results, param_name, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print("PLOT GENERATION COMPLETE!")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please run the full analysis first.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Set to True to only regenerate plots from existing results
    PLOT_ONLY = False
    
    if PLOT_ONLY:
        plot_only_mode(PARAM_TO_ANALYZE)
    else:
        main()

