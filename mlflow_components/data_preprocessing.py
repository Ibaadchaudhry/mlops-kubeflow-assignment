#!/usr/bin/env python3
"""
Standalone Data Preprocessing MLflow Component

This module can be executed independently to perform data preprocessing
with full MLflow tracking and logging.
"""

import os
import sys
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

# Add parent directory to path to import pipeline components
sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline_components_mlflow import data_preprocessing_component


def main():
    """Main function to run data preprocessing as standalone component."""
    parser = argparse.ArgumentParser(description='Data Preprocessing MLflow Component')
    parser.add_argument('--input-data-path', type=str, required=True,
                       help='Path to the input data file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--experiment-name', type=str, default='data_preprocessing_standalone',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Running standalone Data Preprocessing component")
    print(f"Experiment: {args.experiment_name}")
    print(f"Input data: {args.input_data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print("=" * 50)
    
    try:
        # Run data preprocessing component
        processed_data_paths = data_preprocessing_component(
            input_data_path=args.input_data_path,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        print(f"\nData preprocessing completed successfully!")
        print(f"Output files:")
        for key, path in processed_data_paths.items():
            print(f"  {key}: {path}")
        print(f"Check MLflow UI for detailed tracking information")
        
        return processed_data_paths
        
    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()