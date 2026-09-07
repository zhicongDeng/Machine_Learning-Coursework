# Exoplanet Habitability Prediction

Predict whether an exoplanet lies within the habitable zone using machine learning. 

## Dataset

[NASA Exoplanet Archive Intelligence](https://www.kaggle.com/datasets/kanchana1990/nasa-exoplanet-archive-intelligence) from Kaggle.

## Approach

### Data Cleaning & EDA
- Handle missingness (median imputation), outlier analysis, class imbalance check
- Feature distributions, correlation analysis (Pearson, Spearman, Mutual Information), PCA projection

### Feature Engineering
- Log transformations for skewed features
- Derived features: planet density proxy, stellar flux proxy, relative orbital distance
- Categorical encoding (one-hot, ordinal, boolean)

### Models

Three model families, each with progressive improvements:

| Model | Variants |
|-------|----------|
| **Logistic Regression** | Baseline L2, L1/L2 grid search, threshold tuning, SMOTE |
| **SVM** | Baseline RBF, SMOTE, grid search, threshold tuning |
| **XGBoost** | Baseline, hyperparameter tuning, threshold tuning, SHAP analysis |

All models use 60/20/20 stratified splits. Threshold tuning is performed on the validation set and applied to the test set.

### Validation
- Stratification check, data leakage check, overfitting gap analysis
- 5-fold cross-validation stability
- Calibration reliability diagrams

## Tech Stack

Python, scikit-learn, XGBoost, SHAP, imbalanced-learn (SMOTE), matplotlib, seaborn

## Authors

Stephan Vermeulen, Guanwen Zhou, Callum Harris, Ziqi Deng, Zhicong Deng
