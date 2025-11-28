#!/usr/bin/env python3
"""
Standalone Model Evaluation MLflow Component

This module can be executed independently to perform model evaluation
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
from src.pipeline_components_mlflow import model_evaluation_component


def main():
    """Main function to run model evaluation as standalone component."""
    parser = argparse.ArgumentParser(description='Model Evaluation MLflow Component')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--test-features-path', type=str, required=True,
                       help='Path to test features CSV')
    parser.add_argument('--test-target-path', type=str, required=True,
                       help='Path to test target CSV')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--experiment-name', type=str, default='model_evaluation_standalone',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Running standalone Model Evaluation component")
    print(f"Experiment: {args.experiment_name}")
    print(f"Model path: {args.model_path}")
    print(f"Test features: {args.test_features_path}")
    print(f"Test target: {args.test_target_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Run model evaluation component
        evaluation_metrics = model_evaluation_component(
            model_path=args.model_path,
            test_features_path=args.test_features_path,
            test_target_path=args.test_target_path,
            output_dir=args.output_dir
        )
        
        print(f"\nModel evaluation completed successfully!")
        print(f"Evaluation metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"Check MLflow UI for detailed tracking information")
        
        return evaluation_metrics
        
    except Exception as e:
        print(f"Model evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()