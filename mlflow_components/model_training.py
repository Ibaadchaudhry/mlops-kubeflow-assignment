#!/usr/bin/env python3
"""
Standalone Model Training MLflow Component

This module can be executed independently to perform model training
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
from src.pipeline_components_mlflow import model_training_component


def main():
    """Main function to run model training as standalone component."""
    parser = argparse.ArgumentParser(description='Model Training MLflow Component')
    parser.add_argument('--train-features-path', type=str, required=True,
                       help='Path to training features CSV')
    parser.add_argument('--train-target-path', type=str, required=True,
                       help='Path to training target CSV')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save the trained model')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in the forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of trees')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--experiment-name', type=str, default='model_training_standalone',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Running standalone Model Training component")
    print(f"Experiment: {args.experiment_name}")
    print(f"Train features: {args.train_features_path}")
    print(f"Train target: {args.train_target_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print("=" * 50)
    
    try:
        # Run model training component
        model_path = model_training_component(
            train_features_path=args.train_features_path,
            train_target_path=args.train_target_path,
            output_dir=args.output_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        
        print(f"\nModel training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Check MLflow UI for detailed tracking information")
        
        return model_path
        
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()