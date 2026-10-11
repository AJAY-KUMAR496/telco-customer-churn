"""Shared feature engineering. Imported by train.py AND app.py so the saved model loads cleanly."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_RAW = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
NUMERIC_ENGINEERED = ["tenure_monthly_charge_ratio", "high_spend_customer", "long_term_customer"]
NUMERIC_ALL = NUMERIC_RAW + NUMERIC_ENGINEERED

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
RAW_COLUMNS = NUMERIC_RAW + CATEGORICAL


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Cleans raw Telco rows and adds the engineered features from the notebook.

    Unlike the notebook, the 'high spend' median is learned in fit() (training data only),
    so there is no leakage and the same threshold is reused at prediction time.
    """

    def _clean(self, X):
        X = pd.DataFrame(X).copy()
        for col in RAW_COLUMNS:
            if col not in X.columns:
                X[col] = np.nan
        X = X[RAW_COLUMNS]

        # SeniorCitizen may arrive as 0/1 or Yes/No
        if X["SeniorCitizen"].dtype == object:
            X["SeniorCitizen"] = X["SeniorCitizen"].astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
            )
        for col in NUMERIC_RAW:
            X[col] = pd.to_numeric(X[col], errors="coerce")  # TotalCharges has blank strings in Telco data

        # New customers (tenure 0) have no charges yet
        X.loc[X["TotalCharges"].isna() & (X["tenure"] == 0), "TotalCharges"] = 0.0

        for col in CATEGORICAL:
            s = X[col].astype(object)
            s = s.where(s.notna(), np.nan)
            X[col] = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        return X

    def fit(self, X, y=None):
        X = self._clean(X)
        med = X["MonthlyCharges"].median()
        self.charge_median_ = float(med) if pd.notna(med) else 0.0
        return self

    def transform(self, X):
        X = self._clean(X)
        X["tenure_monthly_charge_ratio"] = X["tenure"] / (X["MonthlyCharges"] + 1e-6)
        X["high_spend_customer"] = (X["MonthlyCharges"] > self.charge_median_).astype(int)
        X["long_term_customer"] = (X["tenure"] >= 12).astype(int)
        return X[NUMERIC_ALL + CATEGORICAL]
