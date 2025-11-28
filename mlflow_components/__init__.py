"""
MLflow Components Package

Standalone MLflow components for the machine learning pipeline.
Each component can be executed independently with full MLflow tracking.

Components:
- data_extraction: Extract and track data from DVC storage
- data_preprocessing: Clean, scale, and split data
- model_training: Train models with hyperparameter tracking
- model_evaluation: Evaluate models and log metrics
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"
__email__ = "mlops@example.com"

# Import all components for easy access
from .data_extraction import main as run_data_extraction
from .data_preprocessing import main as run_data_preprocessing
from .model_training import main as run_model_training
from .model_evaluation import main as run_model_evaluation

__all__ = [
    'run_data_extraction',
    'run_data_preprocessing', 
    'run_model_training',
    'run_model_evaluation'
]