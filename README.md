
# Vendor Invoice Intelligence System

**Freight Cost Prediction & Invoice Risk Flagging**

## 📌 Table of Contents

  * [Project Overview](https://www.google.com/search?q=%23project-overview)
  * [Business Objectives](https://www.google.com/search?q=%23business-objectives)
  * [Data Sources](https://www.google.com/search?q=%23data-sources)
  * [Exploratory Data Analysis](https://www.google.com/search?q=%23exploratory-data-analysis)
  * [Models Used](https://www.google.com/search?q=%23models-used)
  * [Evaluation Metrics](https://www.google.com/search?q=%23evaluation-metrics)
  * [Application](https://www.google.com/search?q=%23application)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [How to Run This Project](https://www.google.com/search?q=%23how-to-run-this-project)
  * [Author & Contact](https://www.google.com/search?q=%23author--contact)

-----

## 📌 Project Overview

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1.  **Predicting expected freight cost** for vendor invoices.
2.  **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

-----

## 🎯 Business Objectives

### 1\. Freight Cost Prediction (Regression)

**Objective:** Predict the expected freight cost for a vendor invoice using quantity, invoice value, and historical behavior.

**Why it matters:**

  * Freight is a non-trivial component of landed cost.
  * Poor freight estimation impacts margin analysis and budgeting.
  * Early prediction improves procurement planning and vendor negotiation.
    <img width="1891" height="827" alt="image" src="https://github.com/user-attachments/assets/d1e7463e-c6a3-4309-bcc8-0d12a4af10f7" />


### 2\. Invoice Risk Flagging (Classification)

**Objective:** Predict whether a vendor invoice should be flagged for manual approval due to abnormal cost, freight, or delivery patterns.

**Why it matters:**

  * Manual invoice review does not scale.
  * Financial leakage often occurs in large or complex invoices.
  * Early risk detection improves audit efficiency and operational control.
<img width="1913" height="818" alt="image" src="https://github.com/user-attachments/assets/fed77dbb-f908-4d32-b59d-61536bcf926e" />
<img width="1919" height="815" alt="image" src="https://github.com/user-attachments/assets/89b00350-f969-4cde-b96b-06d78b3bbb60" />


-----

## 📂 Data Sources

Data is stored in a relational SQLite database (`inventory.db`) with the following tables:

  * `vendor_invoice` – Invoice-level financial and timing data
  * `purchases` – Item-level purchase details
  * `purchase_prices` – Reference purchase prices
  * `begin_inventory`, `end_inventory` – Inventory snapshots

SQL aggregation is used to generate **invoice-level features**.

-----

## 📊 Exploratory Data Analysis (EDA)

EDA focuses on **business-driven questions**, such as:

  * Do flagged invoices have higher financial exposure?
  * Does freight scale linearly with quantity?
  * Does freight cost depend on quantity?

-----

## 🤖 Models Used

### Regression (Freight Prediction)

  * Linear Regression (baseline)
  * Decision Tree Regressor
  * Random Forest Regressor (**final model**)

### Classification (Invoice Flagging)

  * Logistic Regression (baseline)
  * Decision Tree Classifier
  * Random Forest Classifier (**final model with GridSearchCV**)

> Hyperparameter tuning is performed using **GridSearchCV** with **F1-score** to handle class imbalance.

-----

## 📈 Evaluation Metrics

### Freight Prediction

  * MAE
  * RMSE
  * $R^2$ Score

### Invoice Flagging

  * Accuracy
  * Precision, Recall, F1-score
  * Classification report
  * Feature importance analysis

-----

## 🖥️ End-to-End Application

A **Streamlit** application demonstrates the complete pipeline:

  * Input invoice details
  * Predict expected freight
  * Flag invoices in real time
  * Provide human-readable explanations

-----

## 📁 Project Structure

```text
inventory-invoice-analytics/
├── data/
│   └── inventory.db
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── models/
│   ├── predict_freight_model.pkl
│   ├── scaler.pkl
│   └── predict_flag_invoice.pkl
├── notebooks/
│   ├── Invoice Flagging.ipynb
│   └── Predict Freight Cost.ipynb
├── app.py
├── README.md
└── .gitignore
```

-----

## How to Run This Project

**1. Clone the repository:**

```bash
git clone https://github.com/nimal-dineshm/Vendor-Invoice-Intelligence-System.git
```

**2. Train and Save Best Fit Models:**

```bash
python freight_cost_prediction/train.py
python invoice_flagging/train.py
```

**3. Test Models:**

```bash
python inference/predict_freight.py
python inference/predict_invoice_flag.py
```

**4. Open Application:**

```bash
streamlit run app.py
```

-----

## Author & Contact

**Nimal Dinesh M** AI/ML Enthusiast
📧 Email: nimaldineshm128@gmail.com  
🔗 [LinkedIn]([https://www.google.com/search?q=%23](https://www.linkedin.com/in/nimal-dinesh-m-7508272bb))  

-----

Would you like me to help you refine any of the project descriptions or add a "Requirements" section with the necessary libraries (like `streamlit`, `scikit-learn`, etc.)?
