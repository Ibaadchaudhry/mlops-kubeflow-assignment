# MLOps Kubeflow Assignment - MLflow Pipeline

A comprehensive Machine Learning Operations (MLOps) pipeline implementing experiment tracking, model versioning, and automated CI/CD using MLflow, DVC, and GitHub Actions.

## 📋 Project Overview

### ML Problem
This project implements a **California Housing Price Classification** pipeline that transforms the regression problem into a multi-class classification task. The pipeline predicts housing price categories (Low, Medium, High) based on geographic and demographic features.

### MLflow Integration
- **Experiment Tracking**: Comprehensive logging of parameters, metrics, and artifacts across all pipeline stages
- **Model Registry**: Automatic model versioning and artifact storage
- **Pipeline Orchestration**: Component-based architecture with MLflow tracking integration
- **Reproducibility**: Full experiment lineage and parameter tracking for reproducible results

### Architecture
```
├── 📊 Data Pipeline
│   ├── Data Extraction (DVC versioned)
│   ├── Data Preprocessing (cleaning, scaling, splitting)
│   ├── Feature Engineering (categorical conversion)
│   └── Data Validation
│
├── 🤖 ML Pipeline  
│   ├── Model Training (Random Forest Classifier)
│   ├── Model Evaluation (metrics calculation)
│   ├── Model Validation (performance assessment)
│   └── Model Registry (MLflow model storage)
│
└── 🔄 MLOps Pipeline
    ├── Experiment Tracking (MLflow)
    ├── Version Control (DVC + Git)
    ├── CI/CD (GitHub Actions)
    └── Monitoring (MLflow UI)
```

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- Git
- Virtual Environment (recommended)

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ibaadchaudhry/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify MLflow installation
mlflow --version
```

### 3. Initialize DVC (Data Version Control)
```bash
# Initialize DVC for data versioning
dvc init

# Add data to DVC tracking (if using external data)
dvc add data/raw_data.csv
git add data/raw_data.csv.dvc .dvc/
git commit -m "Add data to DVC tracking"
```

### 4. MLflow Tracking Setup

#### Local File Storage (Default)
```bash
# MLflow will automatically create ./mlruns directory
# No additional configuration needed
```

#### Optional: External Storage Configuration

**MinIO (S3-Compatible)**
```bash
# Set environment variables
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Configure MLflow
mlflow server --host 0.0.0.0 --port 5000 \
  --default-artifact-root s3://mlflow-bucket/artifacts \
  --backend-store-uri sqlite:///mlflow.db
```

**AWS S3**
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_aws_access_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Run MLflow with S3 backend
mlflow server --host 0.0.0.0 --port 5000 \
  --default-artifact-root s3://your-mlflow-bucket/artifacts \
  --backend-store-uri sqlite:///mlflow.db
```

### 5. Launch MLflow Tracking UI
```bash
# Start MLflow UI (accessible at http://localhost:5000)
mlflow ui --host 127.0.0.1 --port 5000

# Or specify custom tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow ui
```

## 🔄 Pipeline Walkthrough

### Pipeline Architecture
The MLflow pipeline consists of four main components executed in sequence:

1. **Data Extraction** → 2. **Data Preprocessing** → 3. **Model Training** → 4. **Model Evaluation**

Each component is tracked as a separate MLflow experiment with detailed logging.

### Running the Pipeline

#### Method 1: Direct Python Execution (Recommended)
```bash
# Run the complete pipeline
python src/pipeline.py

# Pipeline creates the following experiments:
# - component_pipeline_orchestration (main orchestrator)
# - pipeline_data_extraction
# - pipeline_data_preprocessing  
# - pipeline_model_training
# - pipeline_model_evaluation
```

#### Method 2: MLflow Projects (Alternative)
```bash
# Run using MLflow Projects
mlflow run .

# Run with custom parameters
mlflow run . -P test_size=0.3 -P n_estimators=200
```

### Component Details

#### 🔍 Data Extraction (`pipeline_data_extraction`)
**Purpose**: Load and validate raw California housing dataset

**MLflow Logging**:
```python
# Parameters
mlflow.log_param("input_data_path", "data/raw_data.csv")
mlflow.log_param("output_directory", "data/extracted")

# Metrics  
mlflow.log_metric("data_rows", 20640)
mlflow.log_metric("data_columns", 9)
mlflow.log_metric("missing_values", 0)
mlflow.log_metric("file_size_mb", 1.4)

# Artifacts
mlflow.log_artifact("data/extracted/extracted_data.csv", "extracted_data")
```

**Outputs**: 
- `data/extracted/extracted_data.csv`: Validated raw dataset

#### 🧹 Data Preprocessing (`pipeline_data_preprocessing`)
**Purpose**: Clean, transform, and split data for training

**MLflow Logging**:
```python
# Parameters
mlflow.log_param("test_size", 0.2)
mlflow.log_param("random_state", 42)
mlflow.log_param("feature_columns", str(feature_list))

# Metrics
mlflow.log_metric("train_samples", 16512)
mlflow.log_metric("test_samples", 4128)
mlflow.log_metric("features_count", 8)
mlflow.log_metric("class_Low_count", 6880)
mlflow.log_metric("class_Medium_count", 6880)
mlflow.log_metric("class_High_count", 6880)

# Artifacts
mlflow.log_artifact("data/processed/X_train.csv", "processed_data")
mlflow.log_artifact("data/processed/X_test.csv", "processed_data")
mlflow.log_artifact("data/processed/y_train.csv", "processed_data")
mlflow.log_artifact("data/processed/y_test.csv", "processed_data")
mlflow.log_artifact("data/processed/scaler.pkl", "model_artifacts")
```

**Outputs**:
- Training/test feature matrices with StandardScaler normalization
- Categorical target labels (Low/Medium/High price categories)
- Fitted scaler for inference

#### 🤖 Model Training (`pipeline_model_training`)
**Purpose**: Train Random Forest classifier with hyperparameter tracking

**MLflow Logging**:
```python
# Parameters
mlflow.log_param("algorithm", "Random Forest")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_param("random_state", 42)

# Metrics
mlflow.log_metric("train_accuracy", 0.9856)
mlflow.log_metric("train_samples", 16512)
mlflow.log_metric("feature_importance_MedInc", 0.5247)
mlflow.log_metric("feature_importance_Latitude", 0.1089)
# ... additional feature importance scores

# Model & Artifacts
mlflow.sklearn.log_model(model, "model")
mlflow.log_artifact("models/random_forest_model.pkl", "model_files")
mlflow.log_artifact("models/feature_importance.json", "model_artifacts")
```

**Outputs**:
- Trained Random Forest model (`.pkl` file)
- Feature importance rankings
- MLflow model registry entry

#### 📊 Model Evaluation (`pipeline_model_evaluation`)
**Purpose**: Evaluate model performance on test set

**MLflow Logging**:
```python
# Parameters
mlflow.log_param("model_path", "models/random_forest_model.pkl")
mlflow.log_param("test_samples", 4128)

# Core Metrics
mlflow.log_metric("accuracy", 0.8038)
mlflow.log_metric("precision", 0.8045)
mlflow.log_metric("recall", 0.8038)
mlflow.log_metric("f1_score", 0.8036)

# Per-Class Metrics
mlflow.log_metric("Low_precision", 0.7923)
mlflow.log_metric("Low_recall", 0.8156)
mlflow.log_metric("Low_f1-score", 0.8037)
# ... Medium and High class metrics

# Artifacts
mlflow.log_artifact("evaluation/evaluation_metrics.json", "evaluation_results")
mlflow.log_artifact("evaluation/classification_report.json", "evaluation_results")
mlflow.log_artifact("evaluation/confusion_matrix.csv", "evaluation_results")
```

**Outputs**:
- Comprehensive evaluation metrics
- Classification report with per-class performance
- Confusion matrix analysis

### Viewing Results

#### MLflow UI Dashboard
1. **Open MLflow UI**: `http://localhost:5000`
2. **Navigate to Experiments**: View all pipeline experiments
3. **Compare Runs**: Analyze different parameter combinations
4. **Download Artifacts**: Access models, data, and evaluation results

#### Key Metrics to Monitor
- **Model Performance**: Accuracy (~80.4%), F1-Score (~80.4%)
- **Data Quality**: Missing values, feature distributions
- **Training Efficiency**: Training time, convergence
- **Feature Importance**: Top predictive features

#### Experiment Comparison
```python
# Load and compare experiments programmatically
import mlflow

client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

# Get runs from main experiment
main_exp = client.get_experiment_by_name("component_pipeline_orchestration")
runs = client.search_runs([main_exp.experiment_id])

# Compare metrics across runs
for run in runs:
    print(f"Run {run.info.run_id}: Accuracy = {run.data.metrics.get('accuracy', 'N/A')}")
```

## 🔧 Troubleshooting

### Common Issues

**MLflow UI not accessible**:
```bash
# Check if port is available
netstat -an | grep 5000

# Try different port
mlflow ui --port 5001
```

**Module import errors**:
```bash
# Ensure proper PYTHONPATH
export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"

# Or run from project root
cd /path/to/mlops-kubeflow-assignment
python src/pipeline.py
```

**DVC issues**:
```bash
# Reinitialize DVC if needed
rm -rf .dvc
dvc init
```

## 🚀 CI/CD Pipeline

The project includes automated testing via GitHub Actions:

**Workflow Features**:
- ✅ Environment setup and dependency installation
- ✅ Lightweight pipeline validation (1000 samples)
- ✅ MLflow tracking verification
- ✅ Unit tests with coverage reporting
- ✅ Artifact collection for debugging

**Trigger**: Automatic on push to `main` branch or manual via GitHub Actions tab

## 📁 Project Structure

```
mlops-kubeflow-assignment/
├── 📁 .github/workflows/
│   └── ci.yml                    # GitHub Actions CI pipeline
├── 📁 components/                # MLflow component scripts
│   ├── data_extraction.py        # Data extraction entry point
│   ├── data_preprocessing.py     # Data preprocessing entry point
│   ├── model_training.py         # Model training entry point
│   └── model_evaluation.py      # Model evaluation entry point
├── 📁 src/
│   ├── pipeline.py              # Main pipeline orchestrator
│   └── pipeline_components.py   # Core component implementations
├── 📁 data/                     # Data storage (DVC tracked)
│   ├── raw_data.csv            # Original California housing data
│   ├── extracted/              # Extracted and validated data
│   └── processed/              # Preprocessed train/test splits
├── 📁 models/                   # Trained model artifacts
│   ├── random_forest_model.pkl # Trained model
│   └── feature_importance.json # Feature importance scores
├── 📁 evaluation/              # Model evaluation results
│   ├── evaluation_metrics.json # Performance metrics
│   ├── classification_report.json # Detailed class performance
│   └── confusion_matrix.csv    # Confusion matrix
├── 📁 mlruns/                  # MLflow tracking data (local)
├── requirements.txt            # Python dependencies
├── .dvcignore                 # DVC ignore patterns
├── .gitignore                 # Git ignore patterns
└── README.md                  # This documentation
```

## 🎯 Key Features

- **🔬 Experiment Tracking**: Complete MLflow integration with parameter, metric, and artifact logging
- **📊 Data Versioning**: DVC-based data pipeline with reproducible datasets
- **🔄 Pipeline Automation**: Component-based architecture with automated orchestration
- **✅ Quality Assurance**: Comprehensive testing with GitHub Actions CI/CD
- **📈 Performance Monitoring**: Real-time metrics tracking and model performance analysis
- **🔧 Reproducibility**: Fully documented setup with environment management

## 📈 Results

**Final Model Performance**:
- **Accuracy**: 80.38%
- **Precision**: 80.45% (weighted average)
- **Recall**: 80.38% (weighted average)
- **F1-Score**: 80.36% (weighted average)

**Pipeline Execution Time**: ~2-3 minutes (full pipeline)

**Dataset**: 20,640 California housing samples → 3-class price prediction

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open Pull Request

## 📄 License

This project is part of an MLOps course assignment. See course guidelines for usage permissions.

---

**📧 Contact**: For questions about this MLOps pipeline implementation, please refer to the course materials or submit an issue in the GitHub repository.