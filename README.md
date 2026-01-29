# Heart Disease Logistic Regression Analysis

A machine learning project implementing logistic regression from scratch to predict heart disease presence using clinical data.

## Overview

This project performs a complete analysis pipeline for heart disease prediction:

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Custom train/test split and normalization
- Logistic regression from scratch (sigmoid, cost, gradient descent)
- **No high-level ML libraries** (scikit-learn, statsmodels, TensorFlow, PyTorch) are used

## Dataset

**File:** `Heart_Disease_Prediction.csv`

| Feature | Description |
|---------|-------------|
| Age | Patient age |
| Sex | Gender (1=male, 0=female) |
| Chest pain type | Type of chest pain (1-4) |
| BP | Blood Pressure (mmHg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar > 120 mg/dl |
| EKG results | Electrocardiogram results |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina |
| ST depression | ST depression induced by exercise |
| Slope of ST | Slope of peak exercise ST segment |
| Number of vessels fluro | Number of major vessels (0-3) |
| Thallium | Thallium stress test result |
| Heart Disease | Target: Presence/Absence |

## Notebook Contents

### 1. Data Loading & Target Binarization

- Load CSV into Pandas DataFrame
- Map target column: `Presence` → 1, `Absence` → 0

### 2. Exploratory Data Analysis (EDA)

- Summary statistics and data info
- Missing values check (none found)
- Outlier detection using IQR method
- Outlier handling via capping (winsorization)
- Class distribution visualization (bar chart & pie chart)

### 3. Data Preparation

- **Feature Selection (6 features):**
  - Age, BP, Cholesterol, Max HR, ST depression, Number of vessels fluro
- **Stratified Train/Test Split (70/30):**
  - Manual implementation preserving class proportions
  - Training: 189 samples | Test: 81 samples
- **Min-Max Normalization:**
  - Custom implementation scaling features to [0, 1]

### 4. Logistic Regression 

- Sigmoid activation, binary cross-entropy cost
- Gradient descent training with cost tracking
- Cost vs. iteration plot
- Predictions with threshold 0.5
- Metrics reported on train/test: accuracy, precision, recall, F1
- Coefficient table for basic interpretation

## Results Summary

| Metric | Value |
|--------|-------|
| Total Samples | 270 |
| Training Set | 189 (70%) |
| Test Set | 81 (30%) |
| Class 0 (Absence) | 150 (55.6%) |
| Class 1 (Presence) | 120 (44.4%) |
| Features Selected | 6 |

## Requirements

```bash
pandas
numpy
matplotlib
```

## Usage

1. Clone the repository
2. Ensure the dataset `Heart_Disease_Prediction.csv` is in the project root
3. Open and run `heart_disease_lr_analysis.ipynb`

```bash
jupyter notebook heart_disease_lr_analysis.ipynb
```

## Project Structure

```
Taller02/
├── README.md
├── Heart_Disease_Prediction.csv
└── heart_disease_lr_analysis.ipynb
```

## Author

AREP-2026-1

## License

This project is for educational purposes.
