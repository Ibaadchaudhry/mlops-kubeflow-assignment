# MLOps Kubeflow Assignment

This project demonstrates MLOps practices using Kubeflow, DVC, MLflow, and Docker.

## Dataset

The project uses the **California Housing Dataset** from scikit-learn:
- **Source**: sklearn.datasets.fetch_california_housing()
- **Size**: 20,640 samples with 8 features + 1 target
- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: Median house value in hundreds of thousands of dollars

## Data Version Control (DVC) Setup

DVC has been configured to track the dataset:

### 1. DVC Initialization
```bash
dvc init -f
```

### 2. Remote Storage Setup
```bash
# Local folder as remote storage
dvc remote add -d myremote ./dvc-remote
```

### 3. Data Tracking
```bash
# Download the California housing dataset
python download_data.py

# Add dataset to DVC tracking
dvc add data/raw_data.csv

# Push to remote storage
dvc push
```

### 4. Git Integration
```bash
# Commit DVC files to Git
git add .dvc/ .dvcignore data/raw_data.csv.dvc
git commit -m "Setup DVC and add California housing dataset"
```

## Project Structure

```
mlops-kubeflow-assignment/
├── data/
│   ├── raw_data.csv           # Original dataset (DVC tracked)
│   └── raw_data.csv.dvc       # DVC metadata file
├── src/
│   ├── pipeline_components.py # MLflow component definitions
│   └── model_training.py     # Training script
├── .dvc/
│   ├── config                # DVC configuration
│   └── .gitignore           # DVC gitignore
├── dvc-remote/              # Local DVC remote storage
├── pipeline.py              # Main pipeline definition
├── download_data.py         # Dataset download script
├── requirements.txt         # Project dependencies
├── Dockerfile              # Container configuration
└── .github/workflows/       # CI/CD pipeline
```

## Usage

### Reproduce Data Download
```bash
python download_data.py
```

### Pull Data from DVC Remote
```bash
dvc pull
```

### Check DVC Status
```bash
dvc status
```

## Next Steps

1. Implement model training pipeline
2. Set up MLflow experiment tracking
3. Create Kubeflow pipeline components
4. Configure CI/CD with GitHub Actions