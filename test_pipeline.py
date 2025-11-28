"""
Test Pipeline Components

This script tests all the pipeline components to ensure they work correctly.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)


def test_pipeline():
    """
    Test the complete pipeline by running all components sequentially.
    """
    print("=" * 60)
    print("TESTING MLOPS PIPELINE COMPONENTS")
    print("=" * 60)
    
    try:
        # Step 1: Data Extraction
        print("\n1. TESTING DATA EXTRACTION COMPONENT")
        print("-" * 40)
        extracted_data_path = data_extraction_component(
            data_path="data/raw_data.csv",
            output_dir="data/extracted"
        )
        print(f"✓ Data extraction successful: {extracted_data_path}")
        
        # Step 2: Data Preprocessing
        print("\n2. TESTING DATA PREPROCESSING COMPONENT")
        print("-" * 40)
        processed_files = data_preprocessing_component(
            input_data_path=extracted_data_path,
            output_dir="data/processed",
            test_size=0.2,
            random_state=42
        )
        print(f"✓ Data preprocessing successful: {len(processed_files)} files created")
        
        # Step 3: Model Training
        print("\n3. TESTING MODEL TRAINING COMPONENT")
        print("-" * 40)
        model_path = model_training_component(
            train_features_path=processed_files['train_features'],
            train_target_path=processed_files['train_target'],
            model_output_dir="models",
            n_estimators=50,  # Reduced for faster testing
            max_depth=5,
            random_state=42
        )
        print(f"✓ Model training successful: {model_path}")
        
        # Step 4: Model Evaluation
        print("\n4. TESTING MODEL EVALUATION COMPONENT")
        print("-" * 40)
        metrics = model_evaluation_component(
            model_path=model_path,
            test_features_path=processed_files['test_features'],
            test_target_path=processed_files['test_target'],
            metrics_output_dir="metrics"
        )
        print(f"✓ Model evaluation successful: Test accuracy = {metrics['accuracy']:.4f}")
        
        print("\n" + "=" * 60)
        print("ALL PIPELINE COMPONENTS TESTED SUCCESSFULLY! ✓")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)