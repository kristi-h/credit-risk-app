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
from joblib import dump
from xgboost import XGBClassifier

np.random.seed(42)

# ---------------------------
# 1) Synthetic dataset generation (1,000 users)
# ---------------------------
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

    # Spending patterns
    essentials_spend_annual = income * np.random.uniform(0.3, 0.6, n)
    discretionary_spend_annual = income * np.random.uniform(0.05, 0.4, n)
    risky_spend_annual = income * np.random.uniform(0.01, 0.5, n)

    # Risk labels based on signals
    fraud_flags = (
        (max_transaction_amount > 1500).astype(int) |
        ((transaction_count > 150) & (avg_transaction_amount > 80)).astype(int) |
        ((avg_transaction_amount < 2) & (transaction_count > 200)).astype(int)
    )

    # Heuristic labeling
    signal_score = (
        (income > 40000).astype(int)
        + (employment_years > 2).astype(int)
        + (credit_utilization < 0.5).astype(int)
        + (num_late_payments == 0).astype(int)
        + (num_credit_lines <= 5).astype(int)
        + (transaction_count < 250).astype(int)
        + (avg_transaction_amount <= 200).astype(int)
        + (essentials_spend_annual / (income + 1e-6) <= 0.45).astype(int)
        + (discretionary_spend_annual / (income + 1e-6) <= 0.25).astype(int)
        + (risky_spend_annual / (income + 1e-6) <= 0.2).astype(int)
    )

    labels = np.where((signal_score >= 7) & (fraud_flags == 0), "creditworthy", "not_creditworthy")

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
        "essentials_spend_annual": essentials_spend_annual,
        "discretionary_spend_annual": discretionary_spend_annual,
        "risky_spend_annual": risky_spend_annual,
        "fraud_flag": fraud_flags,
        "label": labels
    })
    return df

df = generate_synthetic_users(1000)
df.to_csv("Transaction_Dataset_2.csv", index=False)
print("Saved synthetic dataset to Transaction_Dataset_2.csv")
print("Label distribution:\n", df['label'].value_counts())

# ---------------------------
# 2) Engineered Features
# ---------------------------
df["too_many_credit_lines"] = (df["num_credit_lines"] > 5).astype(int)
df["too_many_transactions"] = (df["transaction_count"] > 250).astype(int)
df["big_avg_txn"] = (df["avg_transaction_amount"] > 200).astype(int)
df["essentials_ratio"] = df["essentials_spend_annual"] / (df["income"] + 1e-6)
df["discretionary_ratio"] = df["discretionary_spend_annual"] / (df["income"] + 1e-6)
df["risky_ratio"] = df["risky_spend_annual"] / (df["income"] + 1e-6)
df["high_risk_utilization"] = (df["credit_utilization"] > 0.7).astype(int)
df["suspicious_spike"] = (df["max_transaction_amount"] > 1500).astype(int)
df["transaction_ratio"] = df["max_transaction_amount"] / (df["avg_transaction_amount"] + 1e-6)
df["income_per_txn"] = df["income"] / (df["transaction_count"] + 1e-6)
df["utilization_bucket"] = pd.cut(df["credit_utilization"], bins=[-0.01, 0.3, 0.5, 0.7, 1.0], labels=[0,1,2,3]).astype(int)
df["combined_fraud_score"] = (0.6 * df["fraud_flag"] + 0.4 * df["suspicious_spike"]).clip(0,1)

df.to_csv("Transaction_Dataset_2.csv", index=False)
print("Saved engineered features to Transaction_Dataset_2.csv")

# ---------------------------
# 3) Modeling
# ---------------------------
FEATURE_COLS = [
    "age", "income", "employment_years", "credit_utilization", "num_late_payments", "num_credit_lines",
    "transaction_count", "avg_transaction_amount", "max_transaction_amount",
    "too_many_credit_lines", "too_many_transactions", "big_avg_txn",
    "essentials_ratio", "discretionary_ratio", "risky_ratio",
    "high_risk_utilization", "suspicious_spike", "transaction_ratio",
    "income_per_txn", "utilization_bucket", "combined_fraud_score"
]

X = df[FEATURE_COLS]
y = df["label"].map({"creditworthy": 1, "not_creditworthy": 0}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE
sm = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train XGBoost with feature names
dtrain = xgb.DMatrix(X_train_res, label=y_train_res, feature_names=FEATURE_COLS)
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
booster.save_model("xgb_model.json")
print("Saved model: xgb_model.json")

# Save .pkl version too
clf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
clf.fit(X_train_res, y_train_res)
dump(clf, "credit_classifier.pkl")
print("Saved credit_classifier.pkl")