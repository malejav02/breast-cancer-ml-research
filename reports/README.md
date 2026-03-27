# Experiment Reports

This directory contains the results of model experimentation, evaluation, and deployment performance analysis.

All experiments are tracked using **MLflow**, ensuring reproducibility, traceability, and structured comparison across different models and configurations.

---

## Overview

The experimentation process is divided into three main stages:

1. **Cross-validation experiments (model selection)**
2. **Final evaluation on test set**
3. **API performance and latency analysis**

Each section below includes metrics, visualizations, and key insights.

---

## Cross-Validation Results (Model Selection)

During training, multiple models and configurations were evaluated using:

- **5-fold cross-validation**
- `cross_val_predict` for out-of-fold predictions
- Optimization objective: **Macro F1-score**

These experiments help ensure robust model selection and reduce variance due to dataset size.

### Key Metrics

- ROC-AUC
- F1-score (macro)
- Sensitivity
- Specificity

### MLflow Comparison (Cross-Validation)

The following metrics are tracked and visualized in MLflow to compare model performance across cross-validation folds:

- **Model Tracking**  
  Provides an overview of all experiment runs.
  <img width="1641" height="785" alt="image" src="https://github.com/user-attachments/assets/4e152084-cbc7-41c2-b82d-8ffb1a7d3012" />

- **CV Score**  
  Represents the aggregated performance metric across the 5 folds aligned with the optimization objective. 
  This gives an estimate of how well the model generalizes.
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/eee5cdce-0ffd-490c-a010-158ce3855cb9" />

- **F1 Macro**  
  Measures the harmonic mean of precision and recall, averaged across both classes.  
  It is the primary optimization metric, ensuring balanced performance between malignant and benign predictions.
  <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/5b32cd1a-8acb-4976-bc6d-687df55e598e" />


- **ROC-AUC**  
  Evaluates the model’s ability to discriminate between classes across all classification thresholds.  
  Higher values indicate better separability between malignant and benign cases.
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/7915c223-036b-4d9e-8502-75495496b270" />

- **Sensitivity**  
  Measures the proportion of correctly identified positive cases.  
  In this context, it reflects how well the model detects **benign cases** (based on label encoding).
  <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/37c6e111-4ff7-4de0-825b-71650a9441e3" />


- **Specificity**  
  Measures the proportion of correctly identified negative cases.  
  It reflects how well the model identifies **malignant cases**.
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/708b7260-18e6-4893-ba33-e8e7de3aef05" />

- **Training Time (seconds)**  
  Captures the computational cost of training each model.  
  This metric is useful to evaluate the trade-off between performance and efficiency, especially for deployment considerations.
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/3a4638fd-b71d-4e89-81ba-06bbcd543de4" />

- **Confusion Matrix**  
  Displayed for the model with the best performance (XGBClassifier in this case) based on cross-validation predictions (cross_val_predict), showing its results   using this technique
<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/8f7bc38f-063e-434c-9883-9b588ebb6039" />

- **Classification report**
  Presented for the model with the best performance (XGBClassifier in this case) based on cross-validation predictions (cross_val_predict), summarizing its evaluation metrics obtained through this technique.
<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/f3495129-7355-4276-a8ee-6372be5530ef" />


## Test Set Evaluation & Error Analysis (Final Model)

After selecting the best-performing model through cross-validation (XGBClassifier in this case), a final evaluation is conducted on the **held-out test set**. This step not only assesses generalization performance but also provides deeper insights into model behavior through error analysis and interpretability techniques.

### Evaluation Metrics

The model is evaluated using standard classification metrics derived from test predictions:

- Confusion matrix
- Classification report
- Overall performance metrics

### Error Analysis & Interpretability

To better understand model limitations and decision patterns, several analysis techniques are applied:

- Confusion matrix inspection to identify misclassification patterns  
- Feature importance analysis to determine key predictive variables  
- SHAP values to explain individual predictions, especially misclassified samples  

### Example Visualizations

**Confusion Matrix**

<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/8f956fe6-ab3c-4601-b713-f667ec4889ce" />

**Feature Impact Distribution (Shap Beeswarm Plot)**

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/be1bf5ca-9f63-4868-a80a-0d5d46c222a1" />

**Misclassified Samples Analysis**

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/b6e28bf6-85f1-4804-93fc-279d5315a1f1" />

### Key Insights

- Identification of the most influential features in model predictions  
- Detection of patterns among misclassified instances  
- Understanding potential sources of bias, noise, or model weaknesses  

---

This analysis is available in [wisconsin_error_analysis.ipynb ](https://github.com/malejav02/breast-cancer-ml-research/blob/main/notebooks/others/wisconsin_error_analysis.ipynb) which includes:

- Test set evaluation with confusion matrix, classification report, and key metrics  
- SHAP-based interpretability:
  - Feature importance (bar plot)  
  - Feature impact distribution (beeswarm)  
  - Decision-level analysis (decision plot)  
- Comparison between misclassified and correctly classified samples  

## API Performance & Latency

After deployment using FastAPI, the model was evaluated in a production-like setting.

### Results
```
===== BATCH PREDICTION SUMMARY =====
Number of samples processed: 569
   Raw classes: Counter({1: 353, 0: 216})
   Labels:      Counter({'benign': 353, 'malignant': 216})
   Total latency: 0.006881 seconds
   Latency per sample: 0.000012 seconds
```
--- 
## System Specifications

The experiments were executed on the following device:

| Specification        | Details                                                   |
|----------------------|-----------------------------------------------------------|
| **Device Name**      | DESKTOP-UP8PQV6                                          |
| **Processor**        | AMD Ryzen 7 7730U with Radeon Graphics (2.00 GHz)       |
| **Installed RAM**    | 16.0 GB (13.8 GB usable)                                 |
| **Device ID**        | 19A20620-28DD-42CA-AAB0-0F90FE4628CC                     |
| **Product ID**       | 00330-80000-00000-AA557                                  |
| **System Type**      | 64-bit operating system, x64-based processor            |
| **Pen and Touch**    | No pen or touch input available                           |
