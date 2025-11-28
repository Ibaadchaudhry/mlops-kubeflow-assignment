#!/usr/bin/env python3
"""
Model Evaluation Component Entry Point for MLflow Projects

This script serves as an entry point for the model evaluation component
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
from src.pipeline_components import model_evaluation_component


def main():
    """Main entry point for model evaluation component."""
    parser = argparse.ArgumentParser(description='Model Evaluation Component')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test-features-path', type=str, required=True,
                       help='Path to test features file')
    parser.add_argument('--test-target-path', type=str, required=True,
                       help='Path to test target file')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--experiment-name', type=str, default='model_evaluation',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    
    print(f"Starting Model Evaluation Component")
    print(f"Model: {args.model_path}")
    print(f"Test features: {args.test_features_path}")
    print(f"Test target: {args.test_target_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Experiment: {args.experiment_name}")
    print("-" * 50)
    
    try:
        # Run model evaluation
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
        
        # Save results for pipeline orchestration
        output_info = {
            'evaluation_metrics': evaluation_metrics,
            'status': 'success'
        }
        
        with open('model_evaluation_output.json', 'w') as f:
            json.dump(output_info, f)
        
        return evaluation_metrics
        
    except Exception as e:
        print(f"Model evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()