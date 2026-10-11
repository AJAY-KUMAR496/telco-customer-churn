"""Trains, compares, tunes and saves the best churn model -> churn_model.joblib"""
import argparse
import glob
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import CATEGORICAL, NUMERIC_ALL, RAW_COLUMNS, ChurnFeatureBuilder

warnings.filterwarnings("ignore")
SEED = 42
MODEL_PATH = "churn_model.joblib"
DEFAULT_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def find_csv(cli_path=None):
    candidates = [cli_path, DEFAULT_CSV,
                  "/mnt/c/Users/AJAY/Music/deployment/TELCO CUSTOMER CHURN/" + DEFAULT_CSV,
                  r"C:\Users\AJAY\Music\deployment\TELCO CUSTOMER CHURN\\" + DEFAULT_CSV]
    candidates += glob.glob("*elco*.csv") + glob.glob("*hurn*.csv")
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    sys.exit(f"Dataset not found. Put '{DEFAULT_CSV}' next to this notebook or pass --data <path>.")


def make_preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC_ALL), ("cat", cat, CATEGORICAL)])


def make_pipeline(model):
    return Pipeline([("features", ChurnFeatureBuilder()), ("prep", make_preprocessor()), ("model", model)])


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.05, 0.1, 0.5, 1, 5]},
    "Decision Tree": {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [10, 20, 40]},
    "Random Forest": {"model__max_depth": [6, 8, 12, None], "model__min_samples_leaf": [2, 5, 10], "model__max_features": ["sqrt", 0.3]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3, 4], "model__subsample": [0.8, 1.0]},
    "Extra Trees": {"model__max_depth": [8, 12, None], "model__min_samples_leaf": [2, 5, 10], "model__max_features": ["sqrt", 0.3]},
    "Hist Gradient Boosting": {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [3, 5, None], "model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0.0, 1.0]},
}


def best_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.nanargmax(f1))])


def metrics_at(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0), "F1": f1_score(y_true, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, proba), "Threshold": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--fast", action="store_true", help="skip hyper-parameter search")
    args = ap.parse_args()

    csv = find_csv(args.data)
    print("Loading:", csv)
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in RAW_COLUMNS + ["Churn"] if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing columns: {missing}")

    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["Churn"]).drop_duplicates()
    y = df["Churn"].astype(int)
    X = df[RAW_COLUMNS].copy()
    print(f"Rows: {len(X)} | churn rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Compare all models (whole pipeline inside CV -> no leakage)
    rows = []
    for name, model in candidate_models().items():
        pipe = make_pipeline(model)
        auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        rows.append({"Model": name, "CV_ROC_AUC": auc.mean(), "CV_ROC_AUC_std": auc.std(), "CV_F1": f1.mean()})
        print(f"{name:24s} CV ROC-AUC {auc.mean():.4f} (+/-{auc.std():.4f}) | CV F1 {f1.mean():.4f}")
    cv_df = pd.DataFrame(rows).sort_values("CV_ROC_AUC", ascending=False).reset_index(drop=True)
    winner = cv_df.loc[0, "Model"]
    print("\nBest model by CV ROC-AUC:", winner)

    # 2) Light hyper-parameter tuning of the winner
    pipe = make_pipeline(candidate_models()[winner])
    if not args.fast:
        search = RandomizedSearchCV(pipe, PARAM_GRIDS[winner], n_iter=12, scoring="roc_auc", cv=cv,
                                    random_state=SEED, n_jobs=1, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print("Tuned params:", search.best_params_, "| CV ROC-AUC:", round(search.best_score_, 4))
    else:
        pipe.fit(X_train, y_train)

    # 3) Pick decision threshold from out-of-fold predictions on TRAIN only
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thr = best_threshold(y_train, oof)

    # 4) Honest evaluation on untouched test set
    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at(y_test, test_proba, thr)
    default_metrics = metrics_at(y_test, test_proba, 0.5)
    print("\nTEST @ tuned threshold:", {k: round(v, 4) for k, v in test_metrics.items()})
    print("TEST @ 0.50 threshold :", {k: round(v, 4) for k, v in default_metrics.items()})

    # 5) Refit on ALL data for deployment, then save
    final = make_pipeline(candidate_models()[winner]).set_params(
        **{k: v for k, v in pipe.get_params().items() if k.startswith("model__")})
    final.fit(X, y)

    categories = {c: sorted(X[c].dropna().astype(str).str.strip().unique().tolist()) for c in CATEGORICAL}
    artifact = {
        "pipeline": final, "threshold": thr, "model_name": winner, "test_metrics": test_metrics,
        "categories": categories, "sklearn_version": sklearn.__version__,
        "tenure_max": int(X["tenure"].max()), "monthly_max": float(X["MonthlyCharges"].max()),
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    cv_df.to_csv("model_comparison.csv", index=False)
    with open("model_metrics.json", "w") as f:
        json.dump({"model": winner, "threshold": thr, "test": test_metrics}, f, indent=2)
    print(f"\nSaved {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e6:.1f} MB), model_comparison.csv, model_metrics.json")


if __name__ == "__main__":
    main()
