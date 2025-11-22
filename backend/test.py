import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs("/mnt/data/model_artifacts", exist_ok=True)

# 1) Synthetic dataset generation

def generate_synthetic_users(num_users=1000, seed=42):
    np.random.seed(seed)
    n = num_users
    ages = np.random.normal(38, 10, n).clip(18, 80)
    income = np.random.normal(55000, 20000, n).clip(15000, 200000)
    employment_years = np.random.exponential(5, n).clip(0, 40)
    credit_utilization = np.random.beta(2, 5, n)
    num_late_payments = np.random.poisson(0.8, n)
    num_credit_lines = np.random.randint(1, 12, n)
    transaction_count = np.random.poisson(40, n).clip(3, 350)
    avg_transaction_amount = np.random.gamma(2, 30, n)
    max_transaction_amount = avg_transaction_amount * np.random.uniform(2, 8, n)

    fraud_flags = (
        (max_transaction_amount > 1500).astype(int) |
        ((transaction_count > 150) & (avg_transaction_amount > 80)).astype(int) |
        ((avg_transaction_amount < 2) & (transaction_count > 200)).astype(int)
    )

    base_score = (
        0.003 * income
        + 15 * employment_years
        - 80 * credit_utilization
        - 50 * num_late_payments
        + np.random.normal(0, 20, n)
    )
    credit_score = 300 + (base_score - base_score.min()) / (base_score.max() - base_score.min()) * 550
    credit_score = credit_score.clip(300, 850)

    creditworthy_prob = (
        0.004 * (credit_score - 580)
        - 0.4 * fraud_flags
        + np.random.normal(0, 0.1, n)
    )

    labels = np.where(creditworthy_prob > 0.5, "creditworthy", "not_creditworthy")

    df = pd.DataFrame({
        "age": ages,
        "income": income,
        "employment_years": employment_years,
        "credit_utilization": credit_utilization,
        "num_late_payments": num_late_payments,
        "num_credit_lines": num_credit_lines,
        "transaction_count": transaction_count,
        "avg_transaction_amount": avg_transaction_amount,
        "max_transaction_amount": max_transaction_amount,
        "fraud_flag": fraud_flags,
        "credit_score": credit_score,
        "label": labels
    })
    return df

# Generate / save dataset
synthetic_df = generate_synthetic_users(1000)
dataset_path = "/mnt/data/Transaction_Dataset_2.csv"
synthetic_df.to_csv(dataset_path, index=False)
print("Saved synthetic dataset to", dataset_path)
print("Label distribution:\n", synthetic_df['label'].value_counts())

# 2) Feature engineering

df = synthetic_df.copy()
df["high_risk_utilization"] = (df.credit_utilization > 0.7).astype(int)
df["suspicious_spike"] = (df.max_transaction_amount > 1500).astype(int)
df["transaction_ratio"] = df.max_transaction_amount / (df.avg_transaction_amount + 1e-6)
df["income_per_txn"] = df["income"] / (df["transaction_count"] + 1e-6)
df["utilization_bucket"] = pd.cut(df.credit_utilization, bins=[-0.01,0.3,0.5,0.7,1.0], labels=[0,1,2,3]).astype(int)
df["combined_fraud_score"] = (0.6 * df["fraud_flag"] + 0.4 * df["suspicious_spike"]).clip(0, 1)

# Save engineered dataset
eng_path = "/mnt/data/Transaction_Dataset_2.csv"
df.to_csv(eng_path, index=False)
print("Saved engineered features to", eng_path)

# 3) Modeling pipeline

FEATURE_COLS = [
    "age", "income", "employment_years", "credit_utilization", "num_late_payments", "num_credit_lines",
    "transaction_count", "avg_transaction_amount", "max_transaction_amount",
    "high_risk_utilization", "suspicious_spike", "transaction_ratio", "income_per_txn",
    "utilization_bucket", "combined_fraud_score", "credit_score"
]

X = df[FEATURE_COLS].copy()
y = df["label"].map({"creditworthy": 1, "not_creditworthy": 0}).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE to balance training set
sm = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE, class counts:", np.bincount(y_train_res))


# 4) Retrain XGBoost with explicit feature names

dtrain = xgb.DMatrix(X_train_res[FEATURE_COLS], label=y_train_res, feature_names=FEATURE_COLS)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

booster = xgb.train(params, dtrain, num_boost_round=300)

# Save 
booster.save_model("xgb_model.json")
print("Saved updated xgb_model.json with correct feature names")

# test load it
test_model = xgb.Booster()
test_model.load_model("xgb_model.json")
print("Reload test passed")


# 5) Save full model as .pkl for backup

from xgboost import XGBClassifier
from joblib import dump

clf = XGBClassifier(**params, n_estimators=300, use_label_encoder=False)
clf.fit(X_train_res[FEATURE_COLS], y_train_res)
dump(clf, "credit_classifier.pkl")
print("Saved full classifier as credit_classifier.pkl")