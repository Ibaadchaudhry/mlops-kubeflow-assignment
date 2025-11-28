#!/usr/bin/env python3
"""
Data Preprocessing Component Entry Point for MLflow Projects

This script serves as an entry point for the data preprocessing component
in MLflow Projects pipeline orchestration.
"""

import os
import sys
import argparse
import mlflow
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline_components_mlflow import data_preprocessing_component


def main():
    """Main entry point for data preprocessing component."""
    parser = argparse.ArgumentParser(description='Data Preprocessing Component')
    parser.add_argument('--input-data-path', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--experiment-name', type=str, default='data_preprocessing',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Starting Data Preprocessing Component")
    print(f"Input data: {args.input_data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Experiment: {args.experiment_name}")
    print("-" * 50)
    
    try:
        # Run data preprocessing
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
        
        # Save output paths for pipeline orchestration
        output_info = {
            'processed_data_paths': processed_data_paths,
            'status': 'success'
        }
        
        with open('data_preprocessing_output.json', 'w') as f:
            json.dump(output_info, f)
        
        return processed_data_paths
        
    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()