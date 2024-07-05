### README File for GitHub Repository

---

# Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent credit card transactions using various machine learning models. The workflow includes data preprocessing, model training, and evaluation with a focus on handling class imbalance through both under-sampling and over-sampling techniques.

## Table of Contents

1. [Libraries and Tools](#libraries-and-tools)
2. [Project Steps](#project-steps)
3. [Models Used](#models-used)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Conclusion](#conclusion)
8. [Contact](#contact)

## Libraries and Tools

- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Data preprocessing and model building
- **Imbalanced-learn**: Handling imbalanced data (SMOTE)
- **XGBoost**: Advanced gradient boosting

## Project Steps

1. **Reading and Exploring Data**:
    - Loaded the dataset and examined its structure.
    - Checked for missing values and analyzed feature distributions.

2. **Data Cleaning**:
    - Investigated and described the dataset.
    - Addressed class imbalance through under-sampling and over-sampling (SMOTE).

3. **Feature Engineering**:
    - Created a balanced dataset using under-sampling of legitimate transactions.
    - Split data into features and target.

4. **Data Splitting**:
    - Split the data into training and testing sets.

5. **Model Building**:
    - Trained various models including Logistic Regression, Random Forest, AdaBoost, and XGBoost.
    - Performed hyperparameter tuning using GridSearchCV where applicable.

6. **Model Evaluation**:
    - Evaluated models based on accuracy and provided insights into performance.

7. **Fraud Detection System**:
    - Built a simple system to predict if a transaction is fraudulent or legitimate using the trained models.

## Models Used

1. **Logistic Regression**:
    - Trained with extended iterations for convergence.

2. **RandomForestClassifier**:
    - Implemented with specific estimators and depth settings for optimal performance.

3. **AdaBoostClassifier**:
    - Used Decision Tree as the base estimator with tailored hyperparameters.

4. **XGBClassifier**:
    - Applied XGBoost with customized parameters for effective fraud detection.

## Results

- **Logistic Regression**:
    - Training Accuracy: 0.93
    - Testing Accuracy: 0.93

- **RandomForestClassifier**:
    - Training Accuracy: 0.99
    - Testing Accuracy: 0.94

- **AdaBoostClassifier**:
    - Training Accuracy: 1.0
    - Testing Accuracy: 0.92

- **XGBClassifier**:
    - Training Accuracy: 1.0
    - Testing Accuracy: 0.93

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/credit-card-fraud-detection.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to see the results.

## Usage

1. **Data Preparation**:
    - Ensure the dataset is in the correct path as specified in the code.

2. **Model Training**:
    - Run the script or Jupyter notebook to train and evaluate the models.

3. **Evaluation**:
    - Review the output metrics to assess model performance.

## Conclusion

This project demonstrates the use of machine learning to detect fraudulent credit card transactions effectively. The results highlight the significance of balanced data and the power of ensemble methods in fraud detection.

## Contact

Feel free to reach out for any queries or collaboration opportunities.

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Reading the data
data = pd.read_csv('path_to_data/creditcard.csv')

# Data exploration and cleaning
print(data.shape)
print(data.info())
print(data.isnull().sum())
print(data.describe())
print(data.Class.value_counts())

# Handling class imbalance through under-sampling
legit = data[data.Class == 0]
fraud = data[data.Class == 1]
legit_sample = legit.sample(n=492)
new_data = pd.concat([legit_sample, fraud], axis=0)

# Splitting data into features and target
X = new_data.drop(columns="Class", axis=1)
Y = new_data["Class"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Logistic Regression model
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(x_train, y_train)
train_prediction = log_reg.predict(x_train)
train_score = accuracy_score(y_train, train_prediction)
print(f"Logistic Regression Train Accuracy: {train_score}")
Test_prediction = log_reg.predict(x_test)
Test_score = accuracy_score(y_test, Test_prediction)
print(f"Logistic Regression Test Accuracy: {Test_score}")

# Random Forest Classifier model
model_RF = RandomForestClassifier(n_estimators=40, max_depth=10)
model_RF.fit(x_train, y_train)
train_prediction_RF = model_RF.predict(x_train)
train_score_RF = accuracy_score(y_train, train_prediction_RF)
print(f"Random Forest Train Accuracy: {train_score_RF}")
Test_prediction_RF = model_RF.predict(x_test)
Test_score_RF = accuracy_score(y_test, Test_prediction_RF)
print(f"Random Forest Test Accuracy: {Test_score_RF}")

# AdaBoost Classifier model
Adaboost_reg = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=100, min_samples_split=8, min_samples_leaf=4, random_state=42),
    n_estimators=5,
    learning_rate=1
)
Adaboost_reg.fit(x_train, y_train)
print(f"AdaBoost Train Accuracy: {Adaboost_reg.score(x_train, y_train)}")
print(f"AdaBoost Test Accuracy: {Adaboost_reg.score(x_test, y_test)}")

# XGBoost Classifier model
model_xgb = xgb.XGBClassifier(n_estimators=20, max_depth=10, learning_rate=1, min_child_weight=1, random_state=2, missing=60)
model_xgb.fit(x_train, y_train)
print(f"XGBoost Train Accuracy: {model_xgb.score(x_train, y_train)}")
print(f"XGBoost Test Accuracy: {model_xgb.score(x_test, y_test)}")

# Example prediction system
input_Data_Prediction = np.asarray(list(map(float, input().split(",")))).reshape(1, -1)
Prediction = model_xgb.predict(input_Data_Prediction)
print("Fraud" if Prediction == 1 else "Legitimate")

print(f"XGBoost - Train Accuracy: {model_xgb.score(x_train, y_train)}, Test Accuracy: {model_xgb.score(x_test, y_test)}")

# SMOTE for handling imbalanced data
X_2 = data.drop(columns="Class", axis=1)
Y_2 = data["Class"]
smote = SMOTE()
new_x1, new_y1 = smote.fit_resample(X_2, Y_2)
x_train1, x_test1, y_train1, y_test1 = train_test_split(new_x1, new_y1, train_size=0.7, random_state=42)
model_xgb_smote = xgb.XGBClassifier(n_estimators=100, max_depth=1000, learning_rate=0.1, min_child_weight=1, random_state=2)
model_xgb_smote.fit(x_train1, y_train1)
print(f"XGBoost with SMOTE - Train Accuracy: {model_xgb_smote.score(x_train1, y_train1)}, Test Accuracy: {model_xgb_smote.score(x_test1, y_test1)}")
