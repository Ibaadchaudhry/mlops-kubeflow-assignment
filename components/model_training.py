#!/usr/bin/env python3
"""
Model Training Component Entry Point for MLflow Projects

This script serves as an entry point for the model training component
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
from src.pipeline_components import model_training_component


def main():
    """Main entry point for model training component."""
    parser = argparse.ArgumentParser(description='Model Training Component')
    parser.add_argument('--train-features-path', type=str, required=True,
                       help='Path to training features file')
    parser.add_argument('--train-target-path', type=str, required=True,
                       help='Path to training target file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for model')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth of trees')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--experiment-name', type=str, default='model_training',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Starting Model Training Component")
    print(f"Train features: {args.train_features_path}")
    print(f"Train target: {args.train_target_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"Experiment: {args.experiment_name}")
    print("-" * 50)
    
    try:
        # Run model training
        model_path = model_training_component(
            train_features_path=args.train_features_path,
            train_target_path=args.train_target_path,
            model_output_dir=args.output_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        
        print(f"\nModel training completed successfully!")
        print(f"Model saved to: {model_path}")
        
        # Save output path for pipeline orchestration
        output_info = {
            'model_path': model_path,
            'status': 'success'
        }
        
        with open('model_training_output.json', 'w') as f:
            json.dump(output_info, f)
        
        return model_path
        
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()