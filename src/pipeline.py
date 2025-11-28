#!/usr/bin/env python3
"""
Complete MLOps Pipeline with Component Scripts Orchestration

This script orchestrates the entire machine learning pipeline by sequentially 
calling component scripts located in the components/ directory:
- Data extraction
- Data preprocessing  
- Model training
- Model evaluation

Each component is executed as a separate process with MLflow tracking,
and artifacts are passed between components via shared storage.
"""

import os
import sys
import json
import logging
import mlflow
import subprocess
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_component_script(script_path, args, step_name):
    """
    Run a component script as a subprocess and capture output.
    """
    logger.info(f"Executing {step_name}...")
    
    # Use the current Python executable (should be from venv)
    python_executable = sys.executable
    command = [python_executable, script_path] + args
    
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()  # Ensure we're in the right directory
        )
        
        logger.info(f"✓ {step_name} completed successfully")
        if result.stdout:
            logger.info("STDOUT:")
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")
        
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"X {step_name} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error("STDOUT:")
            for line in e.stdout.splitlines():
                logger.error(f"  {line}")
        if e.stderr:
            logger.error("STDERR:")
            for line in e.stderr.splitlines():
                logger.error(f"  {line}")
        raise e


def run_pipeline_with_components():
    """
    Run the complete ML pipeline using component scripts with MLflow tracking.
    """
    logger.info("Starting Pipeline with Component Scripts Orchestration")
    logger.info("=" * 70)
    
    try:
        # Set the main experiment for pipeline orchestration
        mlflow.set_experiment("component_pipeline_orchestration")
        
        # Start main pipeline run
        with mlflow.start_run(run_name="component_pipeline_execution") as main_run:
            logger.info(f"Main pipeline run ID: {main_run.info.run_id}")
            
            # Pipeline configuration
            pipeline_config = {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 10
            }
            
            # Log pipeline parameters
            mlflow.log_params(pipeline_config)
            
            logger.info("Pipeline Configuration:")
            for key, value in pipeline_config.items():
                logger.info(f"  {key}: {value}")
            logger.info("-" * 50)
            
            # Get project root and component paths
            project_root = Path.cwd()
            components_dir = project_root / "components"
            
            logger.info(f"Project root: {project_root}")
            logger.info(f"Components directory: {components_dir}")
            
            # Shared artifact paths
            artifacts = {
                'data_path': 'data/raw_data.csv',
                'extracted_dir': 'data/extracted',
                'processed_dir': 'data/processed', 
                'models_dir': 'models',
                'evaluation_dir': 'evaluation'
            }
            
            # Step 1: Data Extraction
            logger.info("\nSTEP 1: Data Extraction via Component Script")
            logger.info("-" * 50)
            
            data_extraction_args = [
                '--data-path', artifacts['data_path'],
                '--output-dir', artifacts['extracted_dir'],
                '--experiment-name', 'pipeline_data_extraction'
            ]
            
            run_component_script(
                script_path=str(components_dir / 'data_extraction.py'),
                args=data_extraction_args,
                step_name="Data Extraction"
            )
            
            # Determine extracted data path
            extracted_data_path = os.path.join(artifacts['extracted_dir'], 'extracted_data.csv')
            if not os.path.exists(extracted_data_path):
                # Try alternative naming
                extracted_data_path = os.path.join(artifacts['extracted_dir'], 'raw_data.csv')
            
            logger.info(f"Extracted data path: {extracted_data_path}")
            mlflow.log_param("extracted_data_path", extracted_data_path)
            
            # Step 2: Data Preprocessing  
            logger.info("\nSTEP 2: Data Preprocessing via Component Script")
            logger.info("-" * 50)
            
            data_preprocessing_args = [
                '--input-data-path', extracted_data_path,
                '--output-dir', artifacts['processed_dir'],
                '--test-size', str(pipeline_config['test_size']),
                '--random-state', str(pipeline_config['random_state']),
                '--experiment-name', 'pipeline_data_preprocessing'
            ]
            
            run_component_script(
                script_path=str(components_dir / 'data_preprocessing.py'),
                args=data_preprocessing_args,
                step_name="Data Preprocessing"
            )
            
            # Define processed data paths based on actual output  
            processed_data_paths = {
                'X_train_scaled': os.path.join(artifacts['processed_dir'], 'X_train.csv'),
                'X_test_scaled': os.path.join(artifacts['processed_dir'], 'X_test.csv'),
                'y_train': os.path.join(artifacts['processed_dir'], 'y_train.csv'),
                'y_test': os.path.join(artifacts['processed_dir'], 'y_test.csv')
            }
            
            logger.info("Processed data paths:")
            for key, path in processed_data_paths.items():
                logger.info(f"  {key}: {path}")
                mlflow.log_param(f"processed_path_{key}", path)
            
            # Step 3: Model Training
            logger.info("\nSTEP 3: Model Training via Component Script")
            logger.info("-" * 50)
            
            model_training_args = [
                '--train-features-path', processed_data_paths['X_train_scaled'],
                '--train-target-path', processed_data_paths['y_train'],
                '--output-dir', artifacts['models_dir'],
                '--n-estimators', str(pipeline_config['n_estimators']),
                '--max-depth', str(pipeline_config['max_depth']),
                '--random-state', str(pipeline_config['random_state']),
                '--experiment-name', 'pipeline_model_training'
            ]
            
            run_component_script(
                script_path=str(components_dir / 'model_training.py'),
                args=model_training_args,
                step_name="Model Training"
            )
            
            # Define model path
            model_path = os.path.join(artifacts['models_dir'], 'random_forest_model.pkl')
            logger.info(f"Model path: {model_path}")
            mlflow.log_param("model_path", model_path)
            
            # Step 4: Model Evaluation
            logger.info("\nSTEP 4: Model Evaluation via Component Script")
            logger.info("-" * 50)
            
            model_evaluation_args = [
                '--model-path', model_path,
                '--test-features-path', processed_data_paths['X_test_scaled'],
                '--test-target-path', processed_data_paths['y_test'],
                '--output-dir', artifacts['evaluation_dir'],
                '--experiment-name', 'pipeline_model_evaluation'
            ]
            
            run_component_script(
                script_path=str(components_dir / 'model_evaluation.py'),
                args=model_evaluation_args,
                step_name="Model Evaluation"
            )
            
            # Try to read evaluation metrics from output file
            evaluation_metrics = {}
            evaluation_output_file = "model_evaluation_output.json"
            if os.path.exists(evaluation_output_file):
                try:
                    with open(evaluation_output_file, 'r') as f:
                        evaluation_output = json.load(f)
                    evaluation_metrics = evaluation_output.get('evaluation_metrics', {})
                    logger.info("Retrieved evaluation metrics:")
                    for metric, value in evaluation_metrics.items():
                        logger.info(f"  {metric}: {value:.4f}")
                        mlflow.log_metric(f"final_{metric}", value)
                except Exception as e:
                    logger.warning(f"Could not read evaluation output: {e}")
            
            # Create pipeline summary
            pipeline_summary = {
                'pipeline_status': 'success',
                'timestamp': datetime.now().isoformat(),
                'extracted_data_path': extracted_data_path,
                'processed_data_paths': processed_data_paths,
                'model_path': model_path,
                'final_metrics': evaluation_metrics,
                'pipeline_config': pipeline_config
            }
            
            # Save pipeline summary as artifact
            summary_path = "component_pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
            mlflow.log_artifact(summary_path)
            
            # Log execution status
            mlflow.log_param("pipeline_execution_method", "component_scripts")
            mlflow.log_param("components_directory", str(components_dir))
            
            # Pipeline Summary
            logger.info("\n" + "=" * 70)
            logger.info("COMPONENT PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            if evaluation_metrics:
                logger.info(f"Final Model Accuracy: {evaluation_metrics.get('accuracy', 'N/A')}")
            logger.info(f"Main Run ID: {main_run.info.run_id}")
            logger.info(f"Execution Time: {datetime.now()}")
            logger.info("All components executed with individual MLflow tracking")
            logger.info("Check MLflow UI for detailed component runs and metrics")
            
            return pipeline_summary
            
    except Exception as e:
        logger.error(f"Component pipeline execution failed: {str(e)}")
        logger.error("Pipeline terminated with errors")
        
        # Log error to MLflow if we have an active run
        try:
            mlflow.log_param("pipeline_status", "failed")
            mlflow.log_param("error_message", str(e))
        except:
            pass
            
        raise e


def cleanup_output_files():
    """Clean up temporary output files."""
    output_files = [
        "component_pipeline_summary.json",
        "model_evaluation_output.json",
        "data_extraction_output.json",
        "data_preprocessing_output.json",
        "model_training_output.json"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                logger.info(f"Cleaned up: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")


def main():
    """
    Main function to run the component pipeline.
    """
    try:
        # Get current directory and project root
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Ensure we're in the project root
        if current_dir.name == 'src':
            project_root = current_dir.parent
            os.chdir(project_root)
            logger.info(f"Changed to project root: {project_root}")
        
        # Clean up any existing output files
        cleanup_output_files()
        
        # Run the component pipeline
        results = run_pipeline_with_components()
        
        logger.info("\nComponent Pipeline executed successfully!")
        logger.info("Check MLflow UI for detailed run information:")
        logger.info("  mlflow ui")
        
        # Clean up output files
        cleanup_output_files()
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        cleanup_output_files()
        sys.exit(1)


if __name__ == "__main__":
    main()