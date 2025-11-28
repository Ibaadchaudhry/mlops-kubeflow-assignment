#!/usr/bin/env python3
"""
Standalone Data Extraction MLflow Component

This module can be executed independently to perform data extraction
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
from src.pipeline_components_mlflow import data_extraction_component


def main():
    """Main function to run data extraction as standalone component."""
    parser = argparse.ArgumentParser(description='Data Extraction MLflow Component')
    parser.add_argument('--data-path', type=str, default='data/raw_data.csv',
                       help='Path to the raw data file')
    parser.add_argument('--output-dir', type=str, default='data/extracted',
                       help='Directory to save extracted data')
    parser.add_argument('--experiment-name', type=str, default='data_extraction_standalone',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Running standalone Data Extraction component")
    print(f"Experiment: {args.experiment_name}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Run data extraction component
        extracted_data_path = data_extraction_component(
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        print(f"\nData extraction completed successfully!")
        print(f"Output file: {extracted_data_path}")
        print(f"Check MLflow UI for detailed tracking information")
        
        return extracted_data_path
        
    except Exception as e:
        print(f"Data extraction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()