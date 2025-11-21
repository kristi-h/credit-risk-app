import { useState } from "react";

function App() {
  const [form, setForm] = useState({
    age: "",
    monthly_income: "",
    employment_years: "",
    num_late_payments: "",
    num_credit_lines: "",
    essentials_spend: "",
    discretionary_spend: "",
    risky_spend: "",
    transaction_count: "",
    avg_transaction_amount: "",
    max_transaction_amount: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const buildPayload = () => ({
    age: Number(form.age),
    monthly_income: Number(form.monthly_income),
    employment_years: Number(form.employment_years),
    num_late_payments: Number(form.num_late_payments),
    num_credit_lines: Number(form.num_credit_lines),
    essentials_spend: Number(form.essentials_spend),
    discretionary_spend: Number(form.discretionary_spend),
    risky_spend: Number(form.risky_spend),
    transaction_count: Number(form.transaction_count),
    avg_transaction_amount: Number(form.avg_transaction_amount),
    max_transaction_amount: Number(form.max_transaction_amount),
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload = buildPayload();

      const res = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const labelForPrediction = (prediction) => {
    if (!prediction) return "";
    if (prediction === "creditworthy") return "Creditworthy";
    if (prediction === "not_creditworthy") return "Not creditworthy";
    return String(prediction);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#050816",
        color: "#f9fafb",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        padding: "2rem",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 640,
          background: "rgba(15, 23, 42, 0.95)",
          borderRadius: "1.5rem",
          padding: "2rem",
          boxShadow: "0 20px 60px rgba(0,0,0,0.6)",
          border: "1px solid rgba(148, 163, 184, 0.25)",
        }}
      >
        <h1
          style={{
            fontSize: "1.75rem",
            fontWeight: 600,
            marginBottom: "0.5rem",
          }}
        >
          Cash-Flow Credit Risk Prototype
        </h1>
        <p
          style={{
            fontSize: "0.9rem",
            color: "#9ca3af",
            marginBottom: "1.5rem",
          }}
        >
          All inputs are safely bucketed. This model estimates creditworthiness
          from cash-flow, not traditional FICO.
        </p>

        <form
          onSubmit={handleSubmit}
          style={{ display: "grid", gap: "0.9rem", marginBottom: "1.5rem" }}
        >
          <Dropdown
            label="Age range"
            name="age"
            value={form.age}
            onChange={handleChange}
            options={[
              { value: "22", label: "18–24" },
              { value: "30", label: "25–34" },
              { value: "40", label: "35–44" },
              { value: "50", label: "45–54" },
              { value: "60", label: "55–64" },
              { value: "70", label: "65–80" },
            ]}
          />

          <Dropdown
            label="Monthly income"
            name="monthly_income"
            value={form.monthly_income}
            onChange={handleChange}
            options={[
              { value: "30000", label: "< $30k" },
              { value: "45000", label: "$30k – $60k" },
              { value: "73000", label: "$61k – $85k" },
              { value: "103000", label: "$86k – $120k" },
              { value: "138000", label: "$121k – $155k" },
              { value: "176000", label: "$156k - $199k" },
              { value: "240000", label: "$200k - $279k" },
              { value: "328000", label: "$280k - $375k" },
              { value: "463000", label: "$376k - $549k" },
              { value: "775000", label: "$550k - $999k" },
              { value: "1000000", label: "$1m+" },
            ]}
          />

          <Dropdown
            label="Employment stability"
            name="employment_years"
            value={form.employment_years}
            onChange={handleChange}
            options={[
              { value: "0.5", label: "< 1 year" },
              { value: "2", label: "1–3 years" },
              { value: "5", label: "3–7 years" },
              { value: "10", label: "7–15 years" },
              { value: "20", label: "15+ years" },
            ]}
          />

          <Dropdown
            label="Late payments"
            name="num_late_payments"
            value={form.num_late_payments}
            onChange={handleChange}
            options={[
              { value: "0", label: "None" },
              { value: "1", label: "1–2" },
              { value: "3", label: "3–4" },
              { value: "6", label: "5+" },
            ]}
          />

          <Dropdown
            label="Active credit lines"
            name="num_credit_lines"
            value={form.num_credit_lines}
            onChange={handleChange}
            options={[
              { value: "2", label: "1–3" },
              { value: "5", label: "4–6" },
              { value: "8", label: "7–10" },
              { value: "12", label: "10+" },
            ]}
          />

          <Dropdown
            label="Essentials spend"
            name="essentials_spend"
            value={form.essentials_spend}
            onChange={handleChange}
            options={[
              { value: "800", label: "Low (≤ $1k)" },
              { value: "1600", label: "Medium ($1k–$2k)" },
              { value: "2600", label: "High ($2k–$3k)" },
              { value: "3500", label: "Very high ($3k+)" },
            ]}
          />

          <Dropdown
            label="Discretionary spend"
            name="discretionary_spend"
            value={form.discretionary_spend}
            onChange={handleChange}
            options={[
              { value: "200", label: "Low (≤ $300)" },
              { value: "600", label: "Medium ($300–$900)" },
              { value: "1300", label: "High ($900–$2k)" },
            ]}
          />

          <Dropdown
            label="Risky spend"
            name="risky_spend"
            value={form.risky_spend}
            onChange={handleChange}
            options={[
              { value: "0", label: "None" },
              { value: "100", label: "Occasional (≤ $200)" },
              { value: "350", label: "Regular ($200–$600)" },
              { value: "800", label: "Heavy ($600+)" },
            ]}
          />

          <Dropdown
            label="Transactions (30 days)"
            name="transaction_count"
            value={form.transaction_count}
            onChange={handleChange}
            options={[
              { value: "20", label: "≤ 30" },
              { value: "60", label: "30–100" },
              { value: "130", label: "100–200" },
              { value: "220", label: "200+" },
            ]}
          />

          <Dropdown
            label="Average transaction amount"
            name="avg_transaction_amount"
            value={form.avg_transaction_amount}
            onChange={handleChange}
            options={[
              { value: "15", label: "$0–$25" },
              { value: "40", label: "$25–$60" },
              { value: "90", label: "$60–$150" },
              { value: "180", label: "$150+" },
            ]}
          />

          <Dropdown
            label="Max transaction amount"
            name="max_transaction_amount"
            value={form.max_transaction_amount}
            onChange={handleChange}
            options={[
              { value: "150", label: "≤ $200" },
              { value: "400", label: "$200–$600" },
              { value: "1100", label: "$600–$1500" },
              { value: "2200", label: "$1500+" },
            ]}
          />

          <button
            type="submit"
            disabled={loading}
            style={{
              marginTop: "0.5rem",
              borderRadius: "999px",
              padding: "0.75rem 1.25rem",
              border: "none",
              background: loading
                ? "rgba(59,130,246,0.5)"
                : "linear-gradient(135deg,#4f46e5,#06b6d4)",
              color: "#f9fafb",
              fontWeight: 600,
              fontSize: "0.95rem",
              cursor: loading ? "default" : "pointer",
              transition: "transform 0.12s ease, box-shadow 0.12s ease",
              boxShadow: loading ? "none" : "0 12px 30px rgba(37,99,235,0.45)",
            }}
          >
            {loading ? "Scoring..." : "Score credit risk"}
          </button>
        </form>

        {error && (
          <div
            style={{
              marginBottom: "1rem",
              padding: "0.75rem 1rem",
              borderRadius: "0.75rem",
              background: "rgba(239,68,68,0.15)",
              border: "1px solid rgba(239,68,68,0.4)",
              fontSize: "0.85rem",
            }}
          >
            Error: {error}
          </div>
        )}

        {result && (
          <div
            style={{
              padding: "1rem 1.25rem",
              borderRadius: "1rem",
              background: "rgba(15,118,110,0.2)",
              border: "1px solid rgba(45,212,191,0.4)",
              display: "grid",
              gap: "0.3rem",
            }}
          >
            <div style={{ fontSize: "0.85rem", color: "#a5b4fc" }}>
              Model output
            </div>
            <div style={{ fontSize: "1.2rem", fontWeight: 600 }}>
              {labelForPrediction(result.prediction)}
            </div>
            {typeof result.confidence === "number" && (
              <div style={{ fontSize: "0.9rem", color: "#e5e7eb" }}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Dropdown({ label, name, value, onChange, options }) {
  return (
    <div style={{ display: "grid", gap: "0.4rem" }}>
      <label style={{ fontSize: "0.85rem" }}>{label}</label>
      <select
        name={name}
        value={value}
        onChange={onChange}
        required
        style={{
          borderRadius: "0.75rem",
          padding: "0.7rem 0.9rem",
          border: "1px solid rgba(148,163,184,0.6)",
          background: "rgba(15,23,42,0.85)",
          color: "#e5e7eb",
          fontSize: "0.9rem",
        }}
      >
        <option value="">Select...</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default App;
