# Fraud Detection Project – Probabilistic Machine Learning

This repository contains a **probabilistic machine learning project** for detecting fraudulent transactions using Bayesian and frequentist approaches.

---

## 🚀 Project Goal

Fraud detection is a classic case of **imbalanced classification**. Our goal:

✅ Predict whether a transaction is fraudulent  
✅ Compare different probabilistic models  
✅ Quantify uncertainty and interpret model coefficients

We explore:
- Logistic Regression (Frequentist)
- Naive Bayes
- Bayesian Logistic Regression (PyMC)
- Model comparison and uncertainty evaluation

---

## 📂 Folder Structure

```plaintext
26-1CASKXX_fraud_detection/
│
├── data/
│    ├── data_description.md
│    ├── Fraud.csv
│    └── processed_fraud.csv
│
├── notebooks/
│    ├── 01_Data_Exploration_Preprocessing.ipynb
│    ├── 02_Frequentist_Logistic_Regression.ipynb
│    ├── 03_Naive_Bayes_Classification.ipynb
│    ├── 04_Logistic_Regression_Bayesian.ipynb
│    └── 05_Model_Comparison.ipynb
│
├── README.md
└── requirements.txt
```

## 🗂️ Dataset

We use a synthetic fraud detection dataset for training and evaluating fraud detection models.

⚠️ **Important:** The original fraud detection dataset used in this project is large (**~470 MB**) and **not included** in this repository to keep the repo size manageable.

To run this project, you must manually download the dataset and place it into the existing `data/` folder.

### How to Download

1. Visit the Kaggle dataset page:  
   👉 [Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/ashishkumarjayswal/froud-detection-dataset)

2. Download the file **`Fraud.csv`** (~470 MB).

3. Move the downloaded `Fraud.csv` file into the folder `data/` .

Each row in the dataset represents a single transaction, with the following columns:

- **step** → Represents a unit of time where **1 step = 1 hour**
- **type** → Type of online transaction (e.g. TRANSFER, CASH_OUT)
- **amount** → The amount of the transaction
- **nameOrig** → Customer initiating the transaction
- **oldbalanceOrg** → Customer's balance before the transaction
- **newbalanceOrig** → Customer's balance after the transaction
- **nameDest** → Recipient of the transaction
- **oldbalanceDest** → Initial balance of the recipient before the transaction
- **newbalanceDest** → New balance of the recipient after the transaction
- **isFraud** → Target variable:
    - `1` → Fraudulent transaction
    - `0` → Legitimate transaction

---
## 📓 Project Notebooks

### 01_Data_Exploration_Preprocessing.ipynb

- Loads and inspects the fraud dataset.
- Performs data cleaning:
  - Checks for missing values.
  - Converts categorical features.
- Visualizes data distribution:
  - Transaction types
  - Amount distributions
  - Class imbalance
- Creates a processed dataset (`processed_fraud.csv`) for modeling.

---

### 02_Frequentist_Logistic_Regression.ipynb

- Implements logistic regression using scikit-learn.
- Uses a single penalty (default or specified in the notebook).
- Applies `class_weight='balanced'` to handle class imbalance.
- Evaluates the model using:
  - ROC AUC
  - Precision-Recall AUC
  - Confusion matrix
- Plots ROC and Precision-Recall curves.

---

### 03_Naive_Bayes_Classification.ipynb

- Fits a Gaussian Naive Bayes model
- Prints metrics:
  - ROC AUC
  - Precision / Recall
- Plots ROC and PR curves

---

### 04_Logistic_Regression_Bayesian.ipynb

- Implements Bayesian logistic regression in PyMC
- Samples posterior distributions of:
  - Intercept
  - Coefficients
- Visualizes uncertainty using trace plots and summaries

---

### 05_Final_Model_Comparison.ipynb

- Compares:
  - Frequentist logistic regression
  - Naive Bayes
  - Bayesian logistic regression
- Prints ROC AUC and classification reports
- Plots curves for all models side by side

---
