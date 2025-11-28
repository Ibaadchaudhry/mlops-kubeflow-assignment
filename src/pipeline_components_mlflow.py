"""
MLFlow Component Definitions for Kubeflow Pipeline

This module contains MLFlow component definitions for the machine learning pipeline.
Components include data extraction, preprocessing, training, and evaluation with full MLflow tracking.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def data_extraction_component(
    data_path: str = "data/raw_data.csv",
    output_dir: str = "data/extracted"
) -> str:
    """
    Data Extraction Component with MLflow tracking
    
    Fetches the versioned dataset from DVC remote storage and saves it locally.
    
    Args:
        data_path (str): Path to the DVC-tracked data file
        output_dir (str): Directory to save extracted data
        
    Returns:
        str: Path to the extracted data file
    """
    import subprocess
    import shutil
    
    with mlflow.start_run(run_name="data_extraction"):
        print(f"Starting data extraction...")
        print(f"Data path: {data_path}")
        print(f"Output directory: {output_dir}")
        
        # Log parameters
        mlflow.log_param("input_data_path", data_path)
        mlflow.log_param("output_directory", output_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Pull data from DVC remote (if needed)
            print("Checking DVC status...")
            result = subprocess.run(['dvc', 'status'], 
                                  capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print("DVC status check completed")
                mlflow.log_param("dvc_status", "success")
            
            # Check if data file exists
            if not os.path.exists(data_path):
                error_msg = f"Data file not found at {data_path}"
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", error_msg)
                raise FileNotFoundError(error_msg)
            
            # Copy to output directory
            output_file = os.path.join(output_dir, "extracted_data.csv")
            shutil.copy2(data_path, output_file)
            
            # Load and analyze data
            data = pd.read_csv(output_file)
            
            # Log data metrics
            mlflow.log_metric("data_rows", data.shape[0])
            mlflow.log_metric("data_columns", data.shape[1])
            mlflow.log_metric("missing_values", data.isnull().sum().sum())
            mlflow.log_metric("file_size_mb", os.path.getsize(output_file) / (1024 * 1024))
            
            # Log data info as parameters
            mlflow.log_param("data_columns", str(list(data.columns)))
            mlflow.log_param("data_types", str(data.dtypes.to_dict()))
            
            # Log the extracted data file as artifact
            mlflow.log_artifact(output_file, "extracted_data")
            
            # Log success
            mlflow.log_param("status", "success")
            mlflow.log_metric("extraction_success", 1)
            
            print(f"Data extraction completed successfully!")
            print(f"Dataset shape: {data.shape}")
            print(f"Output file: {output_file}")
            
            return output_file
            
        except Exception as e:
            error_msg = f"Error during data extraction: {str(e)}"
            print(f"Error: {error_msg}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", error_msg)
            mlflow.log_metric("extraction_success", 0)
            raise Exception(error_msg)


def data_preprocessing_component(
    input_data_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Data Preprocessing Component with MLflow tracking
    
    Handles cleaning, scaling, and splitting data into train/test sets.
    
    Args:
        input_data_path (str): Path to the input data file
        output_dir (str): Directory to save processed data
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random state for reproducibility
        
    Returns:
        Dict[str, str]: Paths to processed data files
    """
    with mlflow.start_run(run_name="data_preprocessing"):
        print(f"Starting data preprocessing...")
        print(f"Input data: {input_data_path}")
        
        # Log parameters
        mlflow.log_param("input_data_path", input_data_path)
        mlflow.log_param("output_directory", output_dir)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load data
            data = pd.read_csv(input_data_path)
            print(f"Loaded data shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            
            # Log original data metrics
            mlflow.log_metric("original_rows", data.shape[0])
            mlflow.log_metric("original_columns", data.shape[1])
            mlflow.log_param("original_columns_list", str(list(data.columns)))
            
            # Data cleaning
            print("Performing data cleaning...")
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            print(f"Total missing values: {missing_values}")
            mlflow.log_metric("original_missing_values", missing_values)
            
            # Remove any rows with missing values
            data_clean = data.dropna()
            print(f"Data shape after removing missing values: {data_clean.shape}")
            mlflow.log_metric("cleaned_rows", data_clean.shape[0])
            mlflow.log_metric("rows_removed", data.shape[0] - data_clean.shape[0])
            
            # Separate features and target
            feature_columns = [col for col in data_clean.columns if col != 'target']
            X = data_clean[feature_columns]
            y = data_clean['target']
            
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            
            mlflow.log_metric("features_count", X.shape[1])
            mlflow.log_param("feature_columns", str(feature_columns))
            
            # Convert regression target to classification (binning)
            # First ensure target is numeric
            y_numeric = pd.to_numeric(y, errors='coerce')
            
            # Create 3 classes based on target value percentiles
            try:
                y_binned = pd.cut(y_numeric, bins=3, labels=['Low', 'Medium', 'High'])
            except Exception as e:
                print(f"Warning: Error with pd.cut: {e}")
                # Fallback: use quantile-based binning
                quantiles = y_numeric.quantile([0.33, 0.67])
                y_binned = pd.Series(['Low'] * len(y_numeric), index=y_numeric.index)
                y_binned[y_numeric > quantiles.iloc[0]] = 'Medium'
                y_binned[y_numeric > quantiles.iloc[1]] = 'High'
            
            print(f"Target distribution:\n{y_binned.value_counts()}")
            
            # Log target distribution
            target_distribution = y_binned.value_counts().to_dict()
            for class_name, count in target_distribution.items():
                mlflow.log_metric(f"class_{class_name}_count", count)
            
            # Split data
            print(f"Splitting data with test_size={test_size}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binned, test_size=test_size, random_state=random_state, stratify=y_binned
            )
            
            print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Log split metrics
            mlflow.log_metric("train_samples", X_train.shape[0])
            mlflow.log_metric("test_samples", X_test.shape[0])
            mlflow.log_metric("actual_test_ratio", X_test.shape[0] / data_clean.shape[0])
            
            # Feature scaling
            print("Applying feature scaling...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame for easier handling
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Save processed data
            train_features_path = os.path.join(output_dir, "X_train.csv")
            test_features_path = os.path.join(output_dir, "X_test.csv")
            train_target_path = os.path.join(output_dir, "y_train.csv")
            test_target_path = os.path.join(output_dir, "y_test.csv")
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            
            X_train_scaled_df.to_csv(train_features_path, index=False)
            X_test_scaled_df.to_csv(test_features_path, index=False)
            y_train.to_csv(train_target_path, index=False)
            y_test.to_csv(test_target_path, index=False)
            joblib.dump(scaler, scaler_path)
            
            # Log artifacts
            mlflow.log_artifact(train_features_path, "processed_data")
            mlflow.log_artifact(test_features_path, "processed_data")
            mlflow.log_artifact(train_target_path, "processed_data")
            mlflow.log_artifact(test_target_path, "processed_data")
            mlflow.log_artifact(scaler_path, "model_artifacts")
            
            # Log success
            mlflow.log_param("status", "success")
            mlflow.log_metric("preprocessing_success", 1)
            
            output_paths = {
                'train_features': train_features_path,
                'test_features': test_features_path,
                'train_target': train_target_path,
                'test_target': test_target_path,
                'scaler': scaler_path
            }
            
            print(f"Data preprocessing completed successfully!")
            print(f"Output files: {output_paths}")
            
            return output_paths
            
        except Exception as e:
            error_msg = f"Error during data preprocessing: {str(e)}"
            print(f"Error: {error_msg}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", error_msg)
            mlflow.log_metric("preprocessing_success", 0)
            raise Exception(error_msg)


def model_training_component(
    train_features_path: str,
    train_target_path: str,
    output_dir: str = "models",
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> str:
    """
    Model Training Component with MLflow tracking
    
    Trains a Random Forest classifier on the processed training data.
    
    Args:
        train_features_path (str): Path to training features CSV
        train_target_path (str): Path to training target CSV
        output_dir (str): Directory to save the trained model
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        random_state (int): Random state for reproducibility
        
    Returns:
        str: Path to the saved model file
    """
    with mlflow.start_run(run_name="model_training"):
        print(f"Starting model training...")
        
        # Log parameters
        mlflow.log_param("train_features_path", train_features_path)
        mlflow.log_param("train_target_path", train_target_path)
        mlflow.log_param("output_directory", output_dir)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("algorithm", "Random Forest")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load training data
            X_train = pd.read_csv(train_features_path)
            y_train = pd.read_csv(train_target_path).squeeze()
            
            print(f"Training data loaded: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Log training data metrics
            mlflow.log_metric("train_samples", X_train.shape[0])
            mlflow.log_metric("train_features", X_train.shape[1])
            
            # Log target distribution
            target_distribution = y_train.value_counts().to_dict()
            for class_name, count in target_distribution.items():
                mlflow.log_metric(f"train_class_{class_name}_count", count)
            
            # Initialize and train model
            print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Log training metrics
            train_score = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_score)
            
            # Get feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            print(f"Top 5 important features:")
            for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {feature}: {importance:.4f}")
                mlflow.log_metric(f"feature_importance_{feature}", importance)
            
            # Save model
            model_path = os.path.join(output_dir, "random_forest_model.pkl")
            joblib.dump(model, model_path)
            
            # Log model and artifacts
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(model_path, "model_files")
            
            # Save feature importance
            importance_path = os.path.join(output_dir, "feature_importance.json")
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
            mlflow.log_artifact(importance_path, "model_artifacts")
            
            # Log success
            mlflow.log_param("status", "success")
            mlflow.log_metric("training_success", 1)
            
            print(f"Model training completed successfully!")
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            print(f"Error: {error_msg}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", error_msg)
            mlflow.log_metric("training_success", 0)
            raise Exception(error_msg)


def model_evaluation_component(
    model_path: str,
    test_features_path: str,
    test_target_path: str,
    output_dir: str = "evaluation"
) -> Dict[str, float]:
    """
    Model Evaluation Component with MLflow tracking
    
    Evaluates the trained model on the test set and saves metrics.
    
    Args:
        model_path (str): Path to the trained model file
        test_features_path (str): Path to test features CSV
        test_target_path (str): Path to test target CSV
        output_dir (str): Directory to save evaluation results
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    with mlflow.start_run(run_name="model_evaluation"):
        print(f"Starting model evaluation...")
        
        # Log parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("test_features_path", test_features_path)
        mlflow.log_param("test_target_path", test_target_path)
        mlflow.log_param("output_directory", output_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load model
            model = joblib.load(model_path)
            print(f"Model loaded from: {model_path}")
            
            # Load test data
            X_test = pd.read_csv(test_features_path)
            y_test = pd.read_csv(test_target_path).squeeze()
            
            print(f"Test data loaded: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Log test data metrics
            mlflow.log_metric("test_samples", X_test.shape[0])
            mlflow.log_metric("test_features", X_test.shape[1])
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"Evaluation Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
                mlflow.log_metric(metric_name, value)
            
            # Log per-class metrics
            class_report = classification_report(y_test, y_pred, output_dict=True)
            for class_name, class_metrics in class_report.items():
                if isinstance(class_metrics, dict):
                    for metric_name, metric_value in class_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(f"{class_name}_{metric_name}", metric_value)
            
            # Save detailed results
            results_path = os.path.join(output_dir, "evaluation_metrics.json")
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save classification report
            report_path = os.path.join(output_dir, "classification_report.json")
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=2)
            
            # Save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_path = os.path.join(output_dir, "confusion_matrix.csv")
            pd.DataFrame(cm).to_csv(cm_path, index=False)
            
            # Log artifacts
            mlflow.log_artifact(results_path, "evaluation_results")
            mlflow.log_artifact(report_path, "evaluation_results")
            mlflow.log_artifact(cm_path, "evaluation_results")
            
            # Log success
            mlflow.log_param("status", "success")
            mlflow.log_metric("evaluation_success", 1)
            
            print(f"Model evaluation completed successfully!")
            print(f"Results saved to: {output_dir}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error during model evaluation: {str(e)}"
            print(f"Error: {error_msg}")
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", error_msg)
            mlflow.log_metric("evaluation_success", 0)
            raise Exception(error_msg)


def run_full_pipeline(
    data_path: str = "data/raw_data.csv",
    experiment_name: str = "california_housing_pipeline"
) -> Dict[str, Any]:
    """
    Run the complete MLflow pipeline with all components.
    
    Args:
        data_path (str): Path to the raw data
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        Dict[str, Any]: Results from all pipeline components
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    print(f"Starting complete MLflow pipeline...")
    print(f"Experiment: {experiment_name}")
    print("=" * 50)
    
    try:
        # Step 1: Data Extraction
        print("Step 1: Data Extraction")
        extracted_data_path = data_extraction_component(data_path)
        
        # Step 2: Data Preprocessing
        print("\nStep 2: Data Preprocessing")
        processed_data = data_preprocessing_component(extracted_data_path)
        
        # Step 3: Model Training
        print("\nStep 3: Model Training")
        model_path = model_training_component(
            processed_data['train_features'],
            processed_data['train_target']
        )
        
        # Step 4: Model Evaluation
        print("\nStep 4: Model Evaluation")
        metrics = model_evaluation_component(
            model_path,
            processed_data['test_features'],
            processed_data['test_target']
        )
        
        results = {
            'extracted_data_path': extracted_data_path,
            'processed_data': processed_data,
            'model_path': model_path,
            'evaluation_metrics': metrics
        }
        
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        print("Check MLflow UI for detailed tracking information")
        
        return results
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise e


if __name__ == "__main__":
    # Run the complete pipeline
    results = run_full_pipeline()
    print(f"Pipeline results: {results}")