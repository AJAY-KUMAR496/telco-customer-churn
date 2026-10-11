import os
import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "churn_model.joblib")

DEFAULT_CATS = {
    "gender": ["Female", "Male"], "Partner": ["No", "Yes"], "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"], "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"], "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"], "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"], "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"], "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
}
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


@st.cache_resource(show_spinner="Loading model...")
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        return None, "Model file 'churn_model.joblib' not found. Run train.py first and keep it next to app.py."
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:  # version mismatch, corrupt file, etc.
        return None, f"Could not load the model ({type(e).__name__}: {e}). Re-run train.py in the same environment as this app."


def predict_proba(artifact, frame):
    return artifact["pipeline"].predict_proba(frame)[:, 1]


def risk_label(p, thr):
    if p >= max(thr, 0.5) + 0.15 or p >= 0.75:
        return "🔴 High risk"
    if p >= thr:
        return "🟠 Elevated risk"
    return "🟢 Low risk"


artifact, load_error = load_artifact()
st.title("📉 Customer Churn Predictor")
st.caption("Telecom customer churn prediction")

if load_error:
    st.error(load_error)
    st.stop()

cats = {k: (artifact.get("categories", {}).get(k) or v) for k, v in DEFAULT_CATS.items()}
thr = float(artifact.get("threshold", 0.5))
tenure_max = max(72, int(artifact.get("tenure_max", 72)))
monthly_max = max(150.0, float(artifact.get("monthly_max", 120.0)))

with st.sidebar:
    st.subheader("Model info")
    st.write(f"**Model:** {artifact.get('model_name', 'unknown')}")
    st.write(f"**Decision threshold:** {thr:.2f}")
    m = artifact.get("test_metrics", {})
    if m:
        st.write(f"**Test ROC-AUC:** {m.get('ROC_AUC', float('nan')):.3f}")
        st.write(f"**Test Recall:** {m.get('Recall', float('nan')):.3f}")
        st.write(f"**Test Precision:** {m.get('Precision', float('nan')):.3f}")

tab_single, tab_batch = st.tabs(["Single customer", "Batch (CSV)"])

with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Customer**")
        gender = st.selectbox("Gender", cats["gender"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", cats["Partner"])
        dependents = st.selectbox("Dependents", cats["Dependents"])
        tenure = st.slider("Tenure (months)", 0, tenure_max, 12)
    with c2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", cats["PhoneService"])
        if phone == "No":
            lines = "No phone service"
            st.selectbox("Multiple lines", [lines], disabled=True)
        else:
            lines = st.selectbox("Multiple lines", [o for o in cats["MultipleLines"] if o != "No phone service"] or ["No", "Yes"])
        internet = st.selectbox("Internet service", cats["InternetService"])
        addon_values = {}
        for a in ADDONS:
            if internet == "No":
                addon_values[a] = "No internet service"
            else:
                opts = [o for o in cats[a] if o != "No internet service"] or ["No", "Yes"]
                addon_values[a] = st.selectbox(a, opts, key=f"addon_{a}")
        if internet == "No":
            st.info("No internet: add-on services set to 'No internet service'.")
    with c3:
        st.markdown("**Billing**")
        contract = st.selectbox("Contract", cats["Contract"])
        paperless = st.selectbox("Paperless billing", cats["PaperlessBilling"])
        payment = st.selectbox("Payment method", cats["PaymentMethod"])
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=float(monthly_max), value=70.0, step=1.0)
        total = st.number_input("Total charges", min_value=0.0, value=float(round(monthly * tenure, 2)), step=10.0,
                                help="Defaults to monthly charges x tenure; edit if you know the exact value.")

    if st.button("Predict churn", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, **addon_values, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        try:
            p = float(predict_proba(artifact, row)[0])
            st.divider()
            a, b = st.columns([1, 2])
            a.metric("Churn probability", f"{p:.1%}")
            b.markdown(f"### {risk_label(p, thr)}")
            b.progress(min(max(p, 0.0), 1.0))
            if p >= thr:
                st.warning("This customer is likely to churn. Consider a retention offer, e.g. a longer contract or discount.")
            else:
                st.success("This customer is unlikely to churn.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab_batch:
    st.write("Upload a CSV with the same columns as the Telco dataset (`customerID` and `Churn` are optional).")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
            data.columns = [c.strip() for c in data.columns]
            if data.empty:
                st.error("The file has no rows.")
            else:
                from features import RAW_COLUMNS
                absent = [c for c in RAW_COLUMNS if c not in data.columns]
                if absent:
                    st.error(f"Missing columns: {', '.join(absent)}")
                else:
                    probs = predict_proba(artifact, data[RAW_COLUMNS])
                    out = data.copy()
                    out["churn_probability"] = probs.round(4)
                    out["churn_prediction"] = (probs >= thr).map({True: "Yes", False: "No"})
                    st.success(f"Scored {len(out)} customers. {int((probs >= thr).sum())} flagged as likely to churn.")
                    st.dataframe(out.head(500), use_container_width=True)
                    st.download_button("Download results", out.to_csv(index=False).encode("utf-8"),
                                       "churn_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")
