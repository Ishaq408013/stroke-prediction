# Stroke Prediction Using Machine Learning

## Project Overview

This project uses machine-learning methods to predict stroke occurrence from demographic, clinical, and lifestyle features.

The dataset contains 5,110 patient records and 12 original columns. The target variable is `stroke`, where `1` represents a stroke case and `0` represents a non-stroke case.

A major challenge in this dataset is class imbalance. Only about 4.87% of the records are stroke-positive, so model evaluation focuses on metrics beyond accuracy.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── tabular/
│   ├── Stroke_Prediction_FINAL_Code.ipynb
│   └── healthcare-dataset-stroke-data.csv
└── computer_vision/
    └── .gitkeep
```

## Dataset

The tabular dataset includes features such as age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status, and stroke outcome.

The `id` column is removed before modeling because it is only a record identifier.

## Preprocessing Pipeline

The project uses an 80/20 stratified train-test split so that the stroke-to-non-stroke ratio remains similar in both sets.

Preprocessing is performed inside a machine-learning pipeline to reduce data leakage.

Main preprocessing steps:

1. Remove the `id` column.
2. Convert BMI to numeric.
3. Impute missing BMI values using the median.
4. Keep `Unknown` smoking status as its own category.
5. One-hot encode categorical features.
6. Standardize numerical features including age, average glucose level, and BMI.
7. Apply preprocessing using training data only.

## Class Imbalance

Class imbalance is handled using class weighting and SMOTE.

SMOTE is applied only inside the training pipeline so synthetic minority-class samples do not leak into the test set.

## Models Evaluated

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Gradient Boosting
- Logistic Regression with SMOTE

## Model Validation

The project uses 5-fold stratified cross-validation and GridSearchCV. Average precision is used for hyperparameter selection, and the test set remains untouched until final evaluation.

## Evaluation Metrics

Because the target is highly imbalanced, model performance is evaluated using accuracy, precision, recall, F1 score, ROC-AUC, PR-AUC, a confusion matrix, an ROC curve, and a Precision-Recall curve.

## Final Workflow

```text
Raw Data
→ Data Audit
→ Stratified Train/Test Split
→ Preprocessing Pipeline
→ Baseline Model
→ Model Comparison
→ Class-Imbalance Handling
→ Cross-Validation
→ Hyperparameter Tuning
→ Final Test Evaluation
→ Feature Interpretation
→ Error Analysis
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open:

```text
tabular/Stroke_Prediction_FINAL_Code.ipynb
```

3. Run all cells from top to bottom.

The notebook is saved with executed outputs, including plots, model metrics, and evaluation results.

## Requirements

See `requirements.txt` for the required Python packages.
