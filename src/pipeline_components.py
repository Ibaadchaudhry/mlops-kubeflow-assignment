"""
MLFlow Component Definitions for Kubeflow Pipeline

This module contains MLFlow component definitions for the machine learning pipeline.
Components will be implemented here for data preprocessing, model training, and evaluation.
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
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from kfp import components


def data_extraction_component(
    data_path: str = "data/raw_data.csv",
    output_dir: str = "data/extracted"
) -> str:
    """
    Data Extraction Component with MLflow tracking
    
    Fetches the versioned dataset from DVC remote storage and saves it locally.
    Uses DVC to pull the latest version of the dataset.
    
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
            # Pull data from DVC remote
            print("Pulling data from DVC remote...")
            result = subprocess.run(['dvc', 'pull'], 
                                  capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                print(f"DVC pull warning: {result.stderr}")
                mlflow.log_param("dvc_pull_status", "warning")
                mlflow.log_param("dvc_pull_message", result.stderr)
            else:
                print("DVC pull completed successfully")
                mlflow.log_param("dvc_pull_status", "success")
            
            # Check if data file exists
            if not os.path.exists(data_path):
                error_msg = f"Data file not found at {data_path}"
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", error_msg)
                raise FileNotFoundError(error_msg)
            
            # Copy to output directory with timestamp
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
    Data Preprocessing Component
    
    Handles data cleaning, scaling, and splitting into train/test sets.
    
    Args:
        input_data_path (str): Path to the input data file
        output_dir (str): Directory to save processed data
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random state for reproducibility
        
    Returns:
        Dict[str, str]: Paths to processed data files
    """
    print(f"Starting data preprocessing...")
    print(f"Input data: {input_data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(input_data_path)
    print(f"Loaded data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Data cleaning
    print("Performing data cleaning...")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"Missing values:\n{missing_values}")
    
    # Remove any rows with missing values
    data_clean = data.dropna()
    print(f"Data shape after removing missing values: {data_clean.shape}")
    
    # Separate features and target
    feature_columns = [col for col in data_clean.columns if col != 'target']
    X = data_clean[feature_columns]
    y = data_clean['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
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
    
    # Split data
    print(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binned, test_size=test_size, random_state=random_state, stratify=y_binned
    )
    
    print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Feature scaling
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    # Save processed data
    output_files = {}
    
    # Save training data
    train_features_path = os.path.join(output_dir, "X_train.csv")
    train_target_path = os.path.join(output_dir, "y_train.csv")
    X_train_scaled.to_csv(train_features_path, index=False)
    y_train.to_csv(train_target_path, index=False)
    
    # Save test data
    test_features_path = os.path.join(output_dir, "X_test.csv")
    test_target_path = os.path.join(output_dir, "y_test.csv")
    X_test_scaled.to_csv(test_features_path, index=False)
    y_test.to_csv(test_target_path, index=False)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    output_files = {
        'train_features': train_features_path,
        'train_target': train_target_path,
        'test_features': test_features_path,
        'test_target': test_target_path,
        'scaler': scaler_path
    }
    
    print("Data preprocessing completed successfully!")
    print(f"Output files: {output_files}")
    
    return output_files


def model_training_component(
    train_features_path: str,
    train_target_path: str,
    model_output_dir: str = "models",
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> str:
    """
    Model Training Component
    
    Trains a Random Forest classifier on the training data and saves the model artifact.
    
    Args:
        train_features_path (str): Path to training features CSV
        train_target_path (str): Path to training target CSV
        model_output_dir (str): Directory to save trained model
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
        random_state (int): Random state for reproducibility
        
    Returns:
        str: Path to the saved model file
    """
    print("Starting model training...")
    
    # Create output directory
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {train_features_path} and {train_target_path}")
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_target_path).iloc[:, 0]  # First column
    
    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Target distribution:\n{y_train.value_counts()}")
    
    # Initialize MLflow tracking
    mlflow.set_experiment("california-housing-classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Initialize and train model
        print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        print("Model training completed!")
        
        # Calculate training accuracy
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        
        # Save model locally
        model_path = os.path.join(model_output_dir, "random_forest_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Model saved to: {model_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        return model_path


def model_evaluation_component(
    model_path: str,
    test_features_path: str,
    test_target_path: str,
    metrics_output_dir: str = "metrics"
) -> Dict[str, Any]:
    """
    Model Evaluation Component
    
    Loads the trained model, evaluates it on the test set, and saves metrics.
    
    Args:
        model_path (str): Path to the trained model file
        test_features_path (str): Path to test features CSV
        test_target_path (str): Path to test target CSV
        metrics_output_dir (str): Directory to save metrics
        
    Returns:
        Dict[str, Any]: Evaluation metrics
    """
    print("Starting model evaluation...")
    
    # Create output directory
    os.makedirs(metrics_output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {test_features_path} and {test_target_path}")
    X_test = pd.read_csv(test_features_path)
    y_test = pd.read_csv(test_target_path).iloc[:, 0]  # First column
    
    print(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Load trained model
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'precision_micro': precision_score(y_test, y_pred, average='micro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'recall_micro': recall_score(y_test, y_pred, average='micro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_micro': f1_score(y_test, y_pred, average='micro')
    }
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print("-" * 40)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save metrics to JSON file
    metrics_file = os.path.join(metrics_output_dir, "evaluation_metrics.json")
    metrics_data = {
        'metrics': metrics,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'test_set_size': len(y_test),
        'unique_classes': list(np.unique(y_test))
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Save detailed results
    results_file = os.path.join(metrics_output_dir, "test_predictions.csv")
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'prediction_confidence': np.max(y_pred_proba, axis=1)
    })
    results_df.to_csv(results_file, index=False)
    
    # Log metrics with MLflow (if run is active)
    try:
        if mlflow.active_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Log metrics file as artifact
            mlflow.log_artifact(metrics_file)
            mlflow.log_artifact(results_file)
    except:
        print("MLflow not available for logging")
    
    print(f"Metrics saved to: {metrics_file}")
    print(f"Predictions saved to: {results_file}")
    print("Model evaluation completed!")
    
    return metrics


if __name__ == "__main__":
    print("MLFlow pipeline components module")
    print("Available components:")
    print("1. data_extraction_component")
    print("2. data_preprocessing_component") 
    print("3. model_training_component")
    print("4. model_evaluation_component")