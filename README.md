# Bank Customer Churn Prediction
A complete end-to-end machine learning project that predicts which bank customers are likeky to churn, enabling proactive retention strategies

## Overview
Customer churn is one of the most costly problems in banking, acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a machine learning pipeline to identify at-risk customers before they leave, and uses SHAP to explain why they churn in terms the business can act on.

## Results
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 71% | 0.781 |
| Random Forest | 85% | 0.860 |
| XGBoost | 85% | 0.852 |

Best model: Random Forest with AUC-ROC of 0.860

## Key Findings
- Age is the strongest churn driver, customers aged 40+ churn at twice the rate of younger customers
- Inactive members are significantly more likely to churn regardless of other factors
- Germany has nearly double the churn rate of France and Spain (~32% vs ~16%)
- Customers with 3–4 products churn more than expected, possible over-selling issue
- Females churn at a higher rate than males in this dataset

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core Language |
| Pandas & NumPy | Data manipulation |
| matplotlib & Seaborn | Visualisation |
| Scikit-learn | Reprocessing and modelling |
| XGBoost | Gradient boosting model |
| SMOTE(imbalanced-learn) | Handling class imbalance |
| SHAP | Model explainability |

## Project Pipeline
1. Data Inspection = shape, types, missing values, duplicates
2. EDA = 5 visualisations uncovering churn drivers
3. Preprocessing = one-hot encoding, label encoding, StandardScaler
4. Class Imbalance = SMOTE applied to training set only
5. Modelling = Logistic Regression, Random Forest, XGBoost
6. Evaluation = AUC-ROC, classification report, confusion matrix, ROC curves
7. Explainability = SHAP summary and importance plots

## How to Run
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn jupyter

jupyter notebook churn_analysis.ipynb

## Dataset
- **Source:** Bank Customer Churn Prediction dataset (Kaggle)
- **Size:** 10,000 customers, 18 features
- **Target:** Exited (1 = churned, 0 = stayed)
- **Churn rate:** 20.4% imbalanced dataset handled with SMOTE

## Business Recommendations
Based on the model's findings
1. Target retention campaigns at customers aged 40+, highest churn risk segment
2. Trigger re-engagement for inactive members after 3 months of inactivity
3. Investigate the German market, churn rate is nearly double other regions
4. Review product bundling strategy, customers with 3–4 products churn more than expected
