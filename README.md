# Breast Cancer ML Pipeline

End-to-end Machine Learning pipeline for breast cancer classification using the Wisconsin dataset.
This project covers data processing, feature engineering, model training, evaluation, experiment tracking (MLflow), and deployment via FastAPI.

---

## Project Overview

This repository implements a **modular and reproducible ML pipeline** designed to:

* Load and preprocess structured medical data
* Perform feature selection and engineering
* Train and evaluate classification models
* Track experiments with MLflow
* Serve predictions via a REST API using FastAPI

---

## Project Structure

```bash
.
├── api.py                     # FastAPI application
├── test_api.py                # API tests
├── src/                       # Core ML pipeline
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Model configuration & export
│   ├── pipelines/             # Training pipeline
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Utilities (paths, seed)
├── data/
│   └── raw/wisconsin.csv      # Dataset
├── models/
│   └── wisconsin_best_model.pkl
├── notebooks/                 
│   ├── eda/                   # EDA and experiments
├── reports/                   # Reports of the experiments
├── pyproject.toml             # Package configuration
├── requirements.txt           # Dependencies
└── README.md
```

---

## Wisconsin Breast Cancer Dataset

The Wisconsin Breast Cancer Dataset is a widely used dataset for binary classification problems in machine learning. It contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. These features describe characteristics of the cell nuclei present in the images.

The objective of the dataset is to classify tumors into two categories:

- **Malignant (M) - Class 0**: cancerous tumors
- **Benign (B) - Class 1**: non-cancerous tumors

### Dataset Size

- **Total samples:** 569
- **Number of features:** 30 numerical features
- **Target variable:** Diagnosis (Malignant or Benign)

The dataset is moderately imbalanced, with approximately:
- **357 benign cases**
- **212 malignant cases**

### Feature Description

The 30 features represent measurements of cell nuclei characteristics. These measurements are computed from the images and grouped into three categories:

1. **Mean values** – average measurement across the nuclei
2. **Standard error (SE)** – variability of the measurement
3. **Worst values** – largest value observed

Examples of features include:

- **Radius** – mean distance from the center to points on the perimeter
- **Texture** – standard deviation of gray-scale values
- **Perimeter** – length of the boundary of the nucleus
- **Area** – area of the nucleus
- **Smoothness** – local variation in radius lengths
- **Compactness** – perimeter² / area − 1
- **Concavity** – severity of concave portions of the contour
- **Symmetry** – symmetry of the nucleus
- **Fractal dimension** – measure of contour complexity

**Reference:**  
https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset


---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/breast-cancer-ml-research.git
cd breast-cancer-ml-research
```

---

### 2. Create and activate virtual environment

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv/Scripts/activate
```

**Linux/Mac:**

```bash
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install as a package (recommended)

```bash
pip install -e .
```

This allows you to use imports like:

```python
from src.data.load_data import load_wisconsin_dataset
```

---

## Experiment Tracking with MLflow

Start MLflow UI:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## Model Training

Run the training pipeline:

```bash
python src/pipelines/train_pipeline.py
```

This will:

* Load data
* Train the model
* Evaluate performance
* Log metrics and artifacts in MLflow

---

## API Deployment (FastAPI)

Start the API:

```bash
uvicorn api:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

## API Testing (Batch Prediction)

Run:

```bash
python test_api.py
```

### Example Output

```
===== BATCH PREDICTION SUMMARY =====
Number of samples processed: 569
   Raw classes: Counter({1: 353, 0: 216})
   Labels:      Counter({'benign': 353, 'malignant': 216})
   Total latency: 0.006881 seconds
   Latency per sample: 0.000012 seconds
```

---

##  Results

Model performance and experiments are tracked using MLflow. Detailed experiment results, visualizations, and performance analysis can be found here:
**[View Results Report](reports/README.md)**

### Evaluation Metrics

Model evaluation focuses on metrics that better capture performance under potential class imbalance and diagnostic relevance.

The primary metrics considered are:

- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**  
  Measures the model’s ability to discriminate between benign and malignant cases across different thresholds.

- **F1-score (macro)**  
  Provides a balanced evaluation across classes and is particularly useful when class distributions are uneven.

- **Sensitivity**  
  In a Wisconsin context, it measures the proportion of bening cases correctly identified by the model.  

- **Specificity**  
  In a Wisconsin context, it measures the proportion of malignant cases correctly identified by the model.

### Error Analysis & Interpretability

To better understand model limitations and decision patterns, several analysis techniques are applied:

- Confusion matrix inspection to identify misclassification patterns  
- Feature importance analysis to determine key predictive variables  
- SHAP values to explain individual predictions, especially misclassified samples

### Optimization Objective

The training process was optimized to **maximize the Macro F1-score**, ensuring balanced performance across both classes.

**Note**:All experiments were conducted using the Wisconsin dataset and tracked with MLflow to ensure reproducibility and proper experiment management.

---
## End-to-End Workflow

1. Activate environment
2. Run the [data exploration notebook](notebooks/eda/wisconsin_exploration.ipynb)
3. Start MLflow UI
4. Train model
5. Launch FastAPI
6. Run inference via API
7. Execute the [error analysis & interpretability](https://github.com/malejav02/breast-cancer-ml-research/blob/main/notebooks/others/wisconsin_error_analysis.ipynb) 

---

## Reproducibility

To ensure reproducibility, random seeds are fixed across libraries.

---
### Data Splitting Strategy

The data splitting approach depends on the dataset size and the need for reliable performance estimation.

#### Standard Approach (Large Datasets)

For sufficiently large datasets, we use a conventional split:

* Train
* Validation
* Test

This ensures consistent evaluation and prevents information leakage.

#### Wisconsin Dataset Strategy

Given the relatively small size of the Wisconsin dataset, a fixed split may lead to unstable performance estimates. To address this:

* The dataset is split into:

  * 80% training
  * 20% test

* On the training set:

  * Cross-validation is performed for model training and selection
  * `cross_val_predict` is used to obtain out-of-fold predictions and compute reliable metrics

* The best model is then retrained on the full training set and evaluated on the held-out test set


This approach reduces variance, avoids overfitting to a single split, and ensures unbiased evaluation on unseen data.

---
## Responsible AI Considerations

Medical datasets often have important limitations:

* Small sample sizes
* Potential demographic bias
* Limited geographic diversity

These factors can impact model generalization and lead to biased results.

Additionally, in the Wisconsin dataset:

* **Class 0 = malignant**
* **Class 1 = benign**

This is the inverse of the usual convention, so metric interpretation (e.g., sensitivity, specificity) must be handled carefully.

---

## Dataset Usage

The dataset used in this repository is publicly available and is subject to its respective license.

- Breast Cancer Wisconsin Diagnostic Dataset — UCI Machine Learning Repository  

For reproducibility, data loading scripts are provided. Users should ensure compliance with the original dataset licenses when accessing and using the data.

---
## License

The source code in this repository is released under the **MIT License**.

This project is intended for **research and educational purposes**.

---
## Contributions

Contributions are welcome!
Feel free to open issues or submit pull requests.

---
## Author

Maria Alejandra Vélez Clavijo  
Machine Learning & Artificial Intelligence Research


