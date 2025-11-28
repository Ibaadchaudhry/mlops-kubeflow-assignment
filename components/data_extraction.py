#!/usr/bin/env python3
"""
Data Extraction Component Entry Point for MLflow Projects

This script serves as an entry point for the data extraction component
in MLflow Projects pipeline orchestration.
"""

import os
import sys
import argparse
import mlflow
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline_components import data_extraction_component


def main():
    """Main entry point for data extraction component."""
    parser = argparse.ArgumentParser(description='Data Extraction Component')
    parser.add_argument('--data-path', type=str, default='data/raw_data.csv',
                       help='Path to raw data file')
    parser.add_argument('--output-dir', type=str, default='data/extracted',
                       help='Output directory for extracted data')
    parser.add_argument('--experiment-name', type=str, default='data_extraction',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Starting Data Extraction Component")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment: {args.experiment_name}")
    print("-" * 50)
    
    try:
        # Run data extraction
        extracted_data_path = data_extraction_component(
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        print(f"\nData extraction completed successfully!")
        print(f"Output: {extracted_data_path}")
        
        # Save output path to file for pipeline orchestration
        output_info = {
            'extracted_data_path': extracted_data_path,
            'status': 'success'
        }
        
        import json
        with open('data_extraction_output.json', 'w') as f:
            json.dump(output_info, f)
        
        return extracted_data_path
        
    except Exception as e:
        print(f"Data extraction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()