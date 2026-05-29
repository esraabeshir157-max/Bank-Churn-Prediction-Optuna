ر# 📊 Bank Churn Prediction Platform (End-to-End Advanced ML)

This repository contains a production-ready Machine Learning pipeline designed to predict bank customer churn. The project evolved from a classic baseline approach to a highly optimized, interpretable, and business-driven AI solution.

---

## 📌 Project Overview
This project delivers a high-performance, deployment-ready Machine Learning solution to predict bank customer turnover (Churn). By leveraging advanced optimization techniques, the pipeline helps financial institutions proactively identify at-risk customers and implement retention strategies.

---

## 🛠️ Step-by-Step Technical Implementation

### 1. Data Preprocessing & Robust Pipeline
* **Data Splitting:** Applied a **Stratified 80/20 Train/Test Split** to ensure that the ratio of churned vs. retained customers remains perfectly balanced across training and testing sets.
* **Feature Transformation:** Built an automated `ColumnTransformer` pipeline:
  * **Numerical Features:** Scaled using `StandardScaler` to normalize distributions.
  * **Categorical Features:** Encoded using `OneHotEncoder(drop='first')` to handle geography and gender cleanly without creating dummy variable traps.

### 2. Advanced Feature Engineering (The Game Changer)
To capture hidden behavioral patterns, the following domain-specific features were engineered:
* `Balance_Salary_Ratio`: Financial health indicator mapping customer balance against estimated salary.
* `Is_Balance_Zero`: A binary flag capturing customers with zero balance (highly correlated with churn).
* `Tenure_Age_Ratio`: A loyalty metric evaluating time spent with the bank relative to the customer's age.
* `Products_Per_Active`: Behavioral interaction capturing the active engagement across multiple bank products.
* `Age_Group`: Segmented demographic profiling (`Young`, `Middle-Aged`, `Senior`, `Elderly`).

### 3. Iterative Model Exploration (The Journey)
We experimented with multiple world-class ensemble algorithms to find the best fit for our tabular dataset:
* **Random Forest Classifier (Baseline):** Delivered strong initial stability but lacked aggressive capture on the minority churn class.
* **XGBoost Classifier:** Significantly improved the model's sensitivity to catching churners.
* **LightGBM Classifier:** Provided exceptional training speed and smart tree-splitting logic.
* **Voting Ensemble:** Attempted a soft-voting combination (`RF + XGB + LGBM`) which achieved high overall stability.

### 4. Automated Hyperparameter Tuning (Optuna)
Instead of manual guessing or exhaustive GridSearch, we utilized **Optuna** (Bayesian Optimization) to run intelligent automated search trials. Optuna tuned the `learning_rate`, `max_depth`, `num_leaves`, and `min_child_samples` of LightGBM over 20 advanced iterations to squeeze out the maximum theoretical performance.

### 5. Explainable AI (XAI via SHAP)
To build corporate trust, we integrated **SHAP (Shapley Additive exPlanations)**. This allowed us to break down the "black box" model and visualize exactly which features drove the churn decisions, mapping `NumOfProducts` and `Age` as the leading global indicators.

---

## 📈 Final Model Performance Metrics (The Champion)

After complete optimization, the **Optimized LightGBM Classifier via Optuna** emerged as the absolute definitive winner, striking an exceptional balance for the bank's business needs:

```text
=================== PRODUCTION MODEL REPORT ===================
              precision    recall  f1-score   support

           0       0.93      0.82      0.87      1593
           1       0.52      0.76      0.62       407

    accuracy                           0.81      2000
   macro avg       0.72      0.79      0.74      2000
weighted avg       0.85      0.81      0.82      2000

Optimized Production ROC-AUC: 0.8724
===============================================================


🎯 Business Impact Analysis:
76% Churn Recall: The bank can successfully intercept and identify more than 3/4 of all churning customers before they leave.

0.8724 ROC-AUC: Demonstrates a world-class, highly stable ability to distinguish between churning and loyal bank members.

📂 Project Structure
Bank_Churn.csv - Raw historical customer dataset.

Bank-Churn-Prediction.ipynb - Core Jupyter Notebook containing the full iterative code, Optuna trials, and SHAP visualizations.

README.md - Comprehensive documentation of the project.

Developed as a high-fidelity Data Science project showcasing Advanced Feature Engineering, Hyperparameter Optimization, and Explainable AI.
