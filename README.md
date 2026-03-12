# Breast Cancer ML Research

Machine learning research on breast cancer detection using multiple publicly available datasets, combining **clinical biomarkers**, **diagnostic imaging features**, and **histopathology images**.

This repository implements reproducible pipelines for **data preprocessing, model training, evaluation, and error analysis** across several datasets commonly used in machine learning research.

The goal is to explore how different data modalities contribute to breast cancer prediction and to compare classical and deep learning approaches across datasets.

---

# Datasets

This repository uses three well-known breast cancer datasets.

## 1. Breast Cancer Coimbra Dataset

Clinical dataset containing metabolic and anthropometric variables associated with breast cancer.

- **Samples:** 116 patients  
- **Features:** 9 biomarkers  
- **Variables include:** Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP-1  
- **Task:** Binary classification (Cancer vs Control)

Source: UCI Machine Learning Repository

---

## 2. Breast Cancer Wisconsin Diagnostic Dataset

- **Samples:** 569 patients  
- **Features:** 30 computed features from digitized images of breast mass cell nuclei  
- **Task:** Binary classification (Malignant vs Benign)

Source: UCI Machine Learning Repository

---

## 3. BreakHis Dataset

Histopathological images of breast tumor tissue.

- **Images:** 7,900+ microscopy images  
- **Magnifications:** 40X, 100X, 200X, 400X  
- **Task:** Binary classification (Benign vs Malignant)

Source: Breast Cancer Histopathological Database (BreakHis)

---

# Project Structure

```
breast-cancer-ml-research/

data/
    coimbra/
    wisconsin/
    breakhis/

notebooks/
    coimbra_exploration.ipynb
    wisconsin_baseline_models.ipynb
    breakhis_cnn_experiments.ipynb

src/

    data/
        load_data.py
        preprocessing.py

    models/
        train_tabular_models.py
        cnn_model.py

    evaluation/
        metrics.py
        error_analysis.py

    utils/
        seed.py
        config.py

experiments/
    logs/
    model_checkpoints/

requirements.txt
environment.yml
README.md
```

---

# Machine Learning Tasks

The repository includes experiments for multiple machine learning tasks.

## Tabular Data Models

Applied to **Coimbra** and **Wisconsin** datasets.

Models explored:

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machines

Focus:

- feature importance
- biomarker relevance
- model comparison

---

## Deep Learning on Histopathology Images

Applied to **BreakHis** dataset.

Approaches include:

- Convolutional Neural Networks (CNN)
- Transfer learning with pretrained architectures

Focus:

- tumor pattern recognition
- image-level classification
- performance comparison across magnifications

---

# Reproducibility

To ensure reproducibility, random seeds are fixed across libraries.

Example:

```python
import numpy as np
import torch
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

Data splitting strategies depend on the dataset size.

For sufficiently large datasets, we use a standard split:

- Train
- Validation
- Test

This ensures consistent model evaluation and prevents information leakage.

For smaller datasets (such as the Coimbra dataset), where a fixed split may lead to unstable estimates, we instead use **cross-validation** and **cross_val_predict** to obtain more reliable performance estimates across folds.

This approach helps reduce variance in evaluation and provides a more robust estimate of model generalization.


# Environment Setup

Create environment using conda:

```
conda env create -f environment.yml
conda activate breast-cancer-ml
```

Or using pip:

```
pip install -r requirements.txt
```

---

# Evaluation

Model evaluation focuses on metrics that better capture performance under potential class imbalance and diagnostic relevance.

The primary metrics considered are:

- **ROC-AUC (Area Under the Receiver Operating Characteristic Curve)**  
  Measures the model’s ability to discriminate between benign and malignant cases across different thresholds.

- **F1-score (macro)**  
  Provides a balanced evaluation across classes and is particularly useful when class distributions are uneven.

- **Sensitivity (Recall / True Positive Rate)**  
  Measures the proportion of actual cancer cases correctly identified by the model.  
  In clinical settings, high sensitivity is critical to minimize missed diagnoses.

- **Specificity (True Negative Rate)**  
  Measures how well the model correctly identifies non-cancer cases.

Error analysis includes:

- confusion matrices
- inspection of misclassified samples
- feature importance analysis

---

# Responsible AI Considerations

Medical datasets often suffer from limitations such as:

- small sample sizes
- demographic bias
- lack of geographic diversity

These factors must be considered when interpreting model performance.

---

# License

The source code in this repository is released under the **MIT License**.

This project is intended for **research and educational purposes**.

## Dataset Usage

The datasets used in this repository are publicly available and are subject to their respective licenses.

- Breast Cancer Coimbra Dataset — UCI Machine Learning Repository  
- Breast Cancer Wisconsin Diagnostic Dataset — UCI Machine Learning Repository  
- BreakHis (Breast Cancer Histopathological Database)

Datasets are **not redistributed in this repository**.

For reproducibility, data loading scripts are provided.  
Some datasets can be automatically loaded using Python libraries or direct URLs, while others must be downloaded from their original sources due to their size.

Users should ensure compliance with the original dataset licenses when accessing and using the data.


---

# Author

Maria Alejandra Vélez Clavijo  
Machine Learning & Artificial Intelligence Research

