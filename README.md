# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using Random Forest with hyperparameter tuning and a custom decision threshold.

---

## 🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small percentage of total transactions.

This project builds a production-ready fraud detection pipeline including:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling class imbalance
- Hyperparameter tuning (RandomizedSearchCV)
- Custom threshold optimization
- Model serialization
- Streamlit deployment

---

## 📊 Dataset Information

- Source: Kaggle Credit Card Fraud Dataset
- Total Transactions: 284,807
- Fraud Cases: 492 (~0.17%)
- Features:
  - Time
  - Amount
  - V1–V28 (PCA-transformed features)
  - Target: `Class` (0 = Legitimate, 1 = Fraud)

This is an extremely imbalanced dataset.

---

## ⚙️ Model Development

### 🔹 Algorithm Used
Random Forest Classifier

### 🔹 Why Random Forest?
- Handles non-linearity well
- Robust to outliers
- Works well on imbalanced datasets with class_weight
- Feature importance extraction available

---

## 🔍 Hyperparameter Tuning

Used **RandomizedSearchCV** with:

- 3-fold cross validation
- ROC-AUC scoring
- 10 parameter combinations

### ✅ Best Parameters:
-n_estimators = 100
-max_depth = 10
-min_samples_split = 5
-class_weight = 'balanced'

---

## 🎯 Threshold Optimization

Instead of using sklearn’s default 0.5 threshold, a custom threshold of: 0.3

was selected based on precision-recall tradeoff to improve fraud recall while controlling false positives.

---

## 📈 Final Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.975 |
| Precision (Fraud) | 0.94 |
| Recall (Fraud) | 0.83 |
| F1-Score | 0.88 |

Confusion Matrix:
[[56859 5]
[ 17 81]]

---

## 🧠 Key ML Concepts Applied

- Handling severe class imbalance
- Class weighting
- Cross-validation
- ROC-AUC evaluation
- Precision-Recall analysis
- Custom business thresholding
- Model serialization with joblib
- Modular project structure
- Deployment with Streamlit

---

## 🏗 Project Structure
credit_card_fraud_detection/
│
├── data/
│ └── raw/
│ └── creditcard.csv
│
├── model/
│ └── fraud_detection_model.pkl
│
├── src/
│ ├── data_ingestion.py
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ └── utils.py
│
├── application.py # Training pipeline
├── app.py # Streamlit deployment
├── requirements.txt
└── README.md


---

## 🖥 Deployment

The model is deployed locally using Streamlit.

### Run Training Pipeline:

```bash
python application.py
Launch Streamlit App:
streamlit run app.py
🎯 Business Insight

Fraud detection systems must balance:

Catching maximum fraud (high recall)

Avoiding too many false alarms (precision control)

By using a custom probability threshold (0.3), this system improves fraud detection sensitivity without heavily increasing false positives.

💡 Future Improvements

SMOTE comparison

XGBoost benchmarking

SHAP explainability

Docker containerization

Cloud deployment (AWS/GCP)

👩‍💻 Author

Maitreyee
Data Analyst | Aspiring Machine Learning Engineer

⭐ If you found this project useful

Please give it a star ⭐ on GitHub!

