from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app, origins=["https://credit-risk-app.vercel.app", "http://localhost:5173"])

FEATURE_COLS = [
    "age",
    "income",
    "employment_years",
    "credit_utilization",
    "num_late_payments",
    "num_credit_lines",
    "transaction_count",
    "avg_transaction_amount",
    "max_transaction_amount",
    "essentials_ratio",
    "discretionary_ratio",
    "risky_ratio",
    "too_many_credit_lines",
    "too_many_transactions",
    "big_avg_txn",
    "high_discretionary",
    "risk_seeking",
    "overburdened_essentials"
]

model = xgb.Booster()
model.load_model("xgb_model.json")

def build_feature_row(payload: dict) -> dict:
    eps = 1e-6

    age = float(payload.get("age", 0))
    income = float(payload.get("monthly_income", 0))  # Annual income
    employment_years = float(payload.get("employment_years", 0))
    num_late_payments = float(payload.get("num_late_payments", 0))
    num_credit_lines = float(payload.get("num_credit_lines", 0))
    transaction_count = float(payload.get("transaction_count", 0))
    avg_txn = float(payload.get("avg_transaction_amount", 0))
    max_txn = float(payload.get("max_transaction_amount", 0))

    essentials = float(payload.get("essentials_spend", 0))
    discretionary = float(payload.get("discretionary_spend", 0))
    risky = float(payload.get("risky_spend", 0))

    total_spend = essentials + discretionary + risky

    # Ratios
    credit_utilization = min(total_spend / (income + eps), 1.5)
    discretionary_ratio = discretionary / (income + eps)
    risky_ratio = risky / (income + eps)
    essentials_ratio = essentials / (income + eps)

    # Final engineered features (boolean signals)
    too_many_credit_lines = int(num_credit_lines > 5)
    too_many_transactions = int(transaction_count > 250)
    big_avg_txn = int(avg_txn > 200)

    high_discretionary = int(discretionary_ratio > 0.25)
    risk_seeking = int(risky_ratio > 0.2)
    overburdened_essentials = int(essentials_ratio > 0.45)

    return {
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "credit_utilization": credit_utilization,
        "num_late_payments": num_late_payments,
        "num_credit_lines": num_credit_lines,
        "transaction_count": transaction_count,
        "avg_transaction_amount": avg_txn,
        "max_transaction_amount": max_txn,
        "essentials_ratio": essentials_ratio,
        "discretionary_ratio": discretionary_ratio,
        "risky_ratio": risky_ratio,
        "too_many_credit_lines": too_many_credit_lines,
        "too_many_transactions": too_many_transactions,
        "big_avg_txn": big_avg_txn,
        "high_discretionary": high_discretionary,
        "risk_seeking": risk_seeking,
        "overburdened_essentials": overburdened_essentials,
    }
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json or {}

        row = build_feature_row(payload)
        df = pd.DataFrame([row], columns=FEATURE_COLS)
        dmatrix = xgb.DMatrix(df, feature_names=FEATURE_COLS)

        pred_score = float(model.predict(dmatrix)[0])
        label = "creditworthy" if pred_score >= 0.5 else "not_creditworthy"

        return jsonify({
            "prediction": label,
            "confidence": round(pred_score, 4),
            "features_used": row
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API running"})

if __name__ == "__main__":
    app.run(debug=True)