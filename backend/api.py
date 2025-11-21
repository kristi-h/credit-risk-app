from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

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
    "high_risk_utilization",
    "suspicious_spike",
    "transaction_ratio",
    "income_per_txn",
    "utilization_bucket",
    "combined_fraud_score",
    "credit_score",
]

model = xgb.Booster()
model.load_model("xgb_model.json")


def build_feature_row(payload: dict) -> dict:
    eps = 1e-6

    age = float(payload.get("age", 0) or 0)
    income = float(payload.get("monthly_income", 0) or 0)

    employment_years = float(payload.get("employment_years", 0) or 0)
    num_late_payments = float(payload.get("num_late_payments", 0) or 0)
    num_credit_lines = float(payload.get("num_credit_lines", 0) or 0)

    essentials = float(payload.get("essentials_spend", 0) or 0)
    discretionary = float(payload.get("discretionary_spend", 0) or 0)
    risky = float(payload.get("risky_spend", 0) or 0)

    transaction_count = float(payload.get("transaction_count", 0) or 0)
    avg_txn = float(payload.get("avg_transaction_amount", 0) or 0)
    max_txn = float(payload.get("max_transaction_amount", 0) or 0)

    total_spend = essentials + discretionary + risky

    # spend vs income
    credit_utilization = (
        min(total_spend / (income + eps), 1.5) if income > 0 else 0.0
    )

    # Risky spend share
    high_risk_utilization = risky / (total_spend + eps) if total_spend > 0 else 0.0

    suspicious_spike = 1.0 if (max_txn > 1500 or (transaction_count > 150 and avg_txn > 80)) else 0.0

    transaction_ratio = income / (total_spend + eps) if total_spend > 0 else 0.0
    income_per_txn = income / (transaction_count + eps) if transaction_count > 0 else 0.0

    if credit_utilization < 0.3:
        utilization_bucket = 1
    elif credit_utilization < 0.6:
        utilization_bucket = 2
    elif credit_utilization < 0.9:
        utilization_bucket = 3
    else:
        utilization_bucket = 4

    combined_fraud_score = (
        0.5 * high_risk_utilization
        + 0.3 * suspicious_spike
        + 0.2 * float(max_txn > 3000)
    )

    base = (
        0.003 * income
        + 5 * employment_years
        - 50 * credit_utilization
        - 40 * num_late_payments
    )
    credit_score = 580 + base / 20.0
    credit_score = float(max(300, min(850, credit_score)))

    row = {
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "credit_utilization": credit_utilization,
        "num_late_payments": num_late_payments,
        "num_credit_lines": num_credit_lines,
        "transaction_count": transaction_count,
        "avg_transaction_amount": avg_txn,
        "max_transaction_amount": max_txn,
        "high_risk_utilization": high_risk_utilization,
        "suspicious_spike": suspicious_spike,
        "transaction_ratio": transaction_ratio,
        "income_per_txn": income_per_txn,
        "utilization_bucket": utilization_bucket,
        "combined_fraud_score": combined_fraud_score,
        "credit_score": credit_score,
    }

    return row


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.json or {}

        row = build_feature_row(payload)

        df = pd.DataFrame([row], columns=FEATURE_COLS)
        dmatrix = xgb.DMatrix(df, feature_names=FEATURE_COLS)

        pred_score = float(model.predict(dmatrix)[0])
        label = "creditworthy" if pred_score >= 0.5 else "not_creditworthy"

        return jsonify(
            {
                "prediction": label,
                "confidence": pred_score,
                "features_used": row,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API running"})


if __name__ == "__main__":
    app.run(debug=True)