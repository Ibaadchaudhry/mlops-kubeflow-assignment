#!/usr/bin/env python3
"""
MLflow Workflow Orchestrator

This script demonstrates how to run the MLflow components both individually
and as a complete workflow. Each component creates its own MLflow run for
full tracking and reproducibility.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any


def run_component_subprocess(component_name: str, args: Dict[str, Any]) -> str:
    """
    Run a standalone MLflow component as a subprocess.
    
    Args:
        component_name (str): Name of the component to run
        args (Dict[str, Any]): Arguments to pass to the component
        
    Returns:
        str: Output from the component
    """
    script_path = f"mlflow_components/{component_name}.py"
    
    cmd = ["python", script_path]
    
    # Convert arguments to command line format
    for key, value in args.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {component_name}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise e


def run_individual_components_workflow():
    """
    Demonstrate running each component individually with their own MLflow runs.
    """
    print("Running Individual Components Workflow")
    print("Each component will create its own MLflow run")
    print("=" * 60)
    
    # Component 1: Data Extraction
    print("\n1. Running Data Extraction Component")
    print("-" * 40)
    
    extraction_args = {
        "data_path": "data/raw_data.csv",
        "output_dir": "data/extracted_individual",
        "experiment_name": "individual_components_workflow"
    }
    
    run_component_subprocess("data_extraction", extraction_args)
    
    # Component 2: Data Preprocessing
    print("\n2. Running Data Preprocessing Component")
    print("-" * 40)
    
    preprocessing_args = {
        "input_data_path": "data/extracted_individual/extracted_data.csv",
        "output_dir": "data/processed_individual",
        "test_size": 0.2,
        "random_state": 42,
        "experiment_name": "individual_components_workflow"
    }
    
    run_component_subprocess("data_preprocessing", preprocessing_args)
    
    # Component 3: Model Training
    print("\n3. Running Model Training Component")
    print("-" * 40)
    
    training_args = {
        "train_features_path": "data/processed_individual/X_train.csv",
        "train_target_path": "data/processed_individual/y_train.csv",
        "output_dir": "models_individual",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "experiment_name": "individual_components_workflow"
    }
    
    run_component_subprocess("model_training", training_args)
    
    # Component 4: Model Evaluation
    print("\n4. Running Model Evaluation Component")
    print("-" * 40)
    
    evaluation_args = {
        "model_path": "models_individual/random_forest_model.pkl",
        "test_features_path": "data/processed_individual/X_test.csv",
        "test_target_path": "data/processed_individual/y_test.csv",
        "output_dir": "evaluation_individual",
        "experiment_name": "individual_components_workflow"
    }
    
    run_component_subprocess("model_evaluation", evaluation_args)
    
    print("\n" + "=" * 60)
    print("Individual Components Workflow Completed!")
    print("Each component created its own MLflow run in the 'individual_components_workflow' experiment")


def run_integrated_pipeline_workflow():
    """
    Demonstrate running the complete integrated pipeline.
    """
    print("Running Integrated Pipeline Workflow")
    print("All components will run in a single coordinated pipeline")
    print("=" * 60)
    
    # Import and run the integrated pipeline
    sys.path.append(str(Path(__file__).parent))
    from src.pipeline_components_mlflow import run_full_pipeline
    
    results = run_full_pipeline(
        data_path="data/raw_data.csv",
        experiment_name="integrated_pipeline_workflow"
    )
    
    print("\n" + "=" * 60)
    print("Integrated Pipeline Workflow Completed!")
    print(f"Pipeline created multiple runs in the 'integrated_pipeline_workflow' experiment")
    
    return results


def demonstrate_reproducibility():
    """
    Demonstrate how the workflows can be reproduced and tracked.
    """
    print("\n" + "=" * 80)
    print("MLflow Tracking and Reproducibility Demonstration")
    print("=" * 80)
    
    print("\nTo view the MLflow UI and see all tracked experiments:")
    print("1. Run: mlflow ui")
    print("2. Open: http://localhost:5000")
    print("3. Browse experiments:")
    print("   - individual_components_workflow")
    print("   - integrated_pipeline_workflow")
    
    print("\nTo reproduce a specific run:")
    print("1. Find the run ID in MLflow UI")
    print("2. Use: mlflow run <run_id>")
    print("3. Or download artifacts and rerun with same parameters")
    
    print("\nTo run individual components with different parameters:")
    print("python mlflow_components/data_extraction.py --data-path <path> --experiment-name <name>")
    print("python mlflow_components/data_preprocessing.py --input-data-path <path> --test-size 0.3")
    print("python mlflow_components/model_training.py --n-estimators 200 --max-depth 15")
    print("python mlflow_components/model_evaluation.py --model-path <path>")


def main():
    """
    Main function to run workflow demonstrations.
    """
    parser = argparse.ArgumentParser(description='MLflow Workflow Orchestrator')
    parser.add_argument('--workflow', type=str, choices=['individual', 'integrated', 'both'], 
                       default='both', help='Which workflow to run')
    
    args = parser.parse_args()
    
    print("MLflow Workflow Orchestrator")
    print("Demonstrating reproducible ML workflows with full tracking")
    print("=" * 80)
    
    if args.workflow in ['individual', 'both']:
        run_individual_components_workflow()
    
    if args.workflow in ['integrated', 'both']:
        print("\n" + "=" * 80)
        run_integrated_pipeline_workflow()
    
    demonstrate_reproducibility()
    
    print("\n" + "=" * 80)
    print("Workflow demonstration completed!")
    print("Check the MLflow UI to see all tracked experiments, parameters, metrics, and artifacts.")


if __name__ == "__main__":
    main()