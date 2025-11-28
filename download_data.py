"""
Download California Housing Dataset

This script downloads the California housing dataset using scikit-learn
and saves it as a CSV file in the data directory for DVC tracking.
"""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np

def download_california_housing():
    """
    Download and save the California housing dataset.
    """
    print("Downloading California housing dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)
    
    # Combine features and target into a single DataFrame
    data = housing.data.copy()
    data['target'] = housing.target
    
    # Save to CSV
    output_path = 'data/raw_data.csv'
    data.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\nFirst few rows:")
    print(data.head())
    
    return output_path

if __name__ == "__main__":
    download_california_housing()