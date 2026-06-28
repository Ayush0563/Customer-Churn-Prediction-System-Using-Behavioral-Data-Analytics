import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import optuna
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report,
    confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

print("Loading dataset...")

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df.drop("customerID", axis=1, inplace=True)

services = [
    "PhoneService","MultipleLines","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies"
]

df["Service_Count"] = 0
for col in services:
    df["Service_Count"] += (df[col] == "Yes").astype(int)

df["Spend_Velocity"] = df["TotalCharges"] / (df["tenure"] + 1)
df["Monthly_to_Total_Ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
df["Charge_Trend"] = df["MonthlyCharges"] / (df["Spend_Velocity"] + 1)
df["Service_Density"] = df["Service_Count"] / (df["tenure"] + 1)
df["Charges_Per_Service"] = df["MonthlyCharges"] / (df["Service_Count"] + 1)
df["Tenure_x_Monthly"] = df["tenure"] * df["MonthlyCharges"]
df["High_Value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
df["HighRiskContract"] = (df["Contract"] == "Month-to-month").astype(int)
df["Senior_Alone"] = (
    (df["SeniorCitizen"] == 1) &
    (df["Partner"] == "No") &
    (df["Dependents"] == "No")
).astype(int)

encoders = {}
for col in df.select_dtypes(include="object").columns:
    if col != "Churn":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop("Churn", axis=1)
y = df["Churn"]

selected_features = X.columns.tolist()

X_tune = X.sample(min(4000, len(X)), random_state=42)
y_tune = y.loc[X_tune.index]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "verbose": -1
    }

    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = []
    for tr_idx, va_idx in cv.split(X_tune, y_tune):
        Xtr, Xva = X_tune.iloc[tr_idx], X_tune.iloc[va_idx]
        ytr, yva = y_tune.iloc[tr_idx], y_tune.iloc[va_idx]

        model.fit(Xtr, ytr)
        pred = model.predict_proba(Xva)[:, 1]
        scores.append(roc_auc_score(yva, pred))

    return np.mean(scores)

print("Running Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_params = study.best_params

xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

lgb = LGBMClassifier(**best_params, random_state=42, verbose=-1)

cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    verbose=0,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

et = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

stack_model = StackingClassifier(
    estimators=[
        ("xgb", xgb),
        ("lgb", lgb),
        ("cat", cat),
        ("rf", rf),
        ("et", et)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

print("Generating OOF predictions...")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_proba = cross_val_predict(
    stack_model,
    X,
    y,
    cv=skf,
    method="predict_proba",
    n_jobs=1
)[:, 1]

best_threshold = 0.5
best_score = 0

for threshold in np.arange(0.3, 0.4, 0.6):
    pred = (oof_proba >= threshold).astype(int)

    score = (
        0.7 * precision_score(y, pred, zero_division=0)
        + 0.3 * recall_score(y, pred, zero_division=0)
    )

    if score > best_score:
        best_score = score
        best_threshold = threshold

final_preds = (oof_proba >= best_threshold).astype(int)

auc = roc_auc_score(y, oof_proba)
precision = precision_score(y, final_preds)
recall = recall_score(y, final_preds)
f1 = f1_score(y, final_preds)

print("ROC AUC:", auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print(confusion_matrix(y, final_preds))
print(classification_report(y, final_preds))

print("Training final model...")
stack_model.fit(X, y)

print("Generating SHAP plot...")
lgb_shap = LGBMClassifier(**best_params, random_state=42, verbose=-1)
lgb_shap.fit(X, y)

explainer = shap.TreeExplainer(lgb_shap)
shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")

package = {
    "model": stack_model,
    "encoders": encoders,
    "features": selected_features,
    "threshold": best_threshold,
    "roc_auc": auc,
    "f1": f1
}

joblib.dump(package, "customer_churn_production.pkl")

print("Done.")
