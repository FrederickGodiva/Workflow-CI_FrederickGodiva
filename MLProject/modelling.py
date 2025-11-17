import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import mlflow
import joblib


def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    os.makedirs("artifacts", exist_ok=True)
    path = os.path.join("artifacts", f"{filename}.png")

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()

    return path


mlflow.set_experiment("BreastCancer_Experiment")

df = pd.read_csv("processed_data.csv")

features = df.drop(columns="diagnosis", axis=1)
target = df["diagnosis"]

le = LabelEncoder()
target_encoded = le.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(
    features, target_encoded, test_size=0.3, random_state=42
)

os.makedirs("artifacts", exist_ok=True)

model_results = {}


with mlflow.start_run(run_name="XGBoost_Baseline", nested=True) as xgb_run:
    mlflow.xgboost.autolog()

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_xgb_labels = le.inverse_transform(y_pred_xgb)
    y_test_labels = le.inverse_transform(y_test)

    cm_path = save_confusion_matrix(
        y_test_labels, y_pred_xgb_labels, labels=le.classes_,
        title="XGBoost Confusion Matrix", filename="xgb_confusion_matrix"
    )
    mlflow.log_artifact(cm_path)

    report = classification_report(y_test_labels, y_pred_xgb_labels)
    report_path = "artifacts/xgb_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    xgb_acc = accuracy_score(y_test_labels, y_pred_xgb_labels)
    xgb_f1 = f1_score(y_test_labels, y_pred_xgb_labels, average="weighted")

    mlflow.log_metrics({
        "test_accuracy": xgb_acc,
        "test_f1_score": xgb_f1
    })

    print("Accuracy XGBoost:", xgb_acc)
    print("F1 XGBoost:", xgb_f1)

    model_results["XGBoost"] = {
        "model": xgb_model,
        "f1": xgb_f1,
        "run_id": xgb_run.info.run_id
    }


with mlflow.start_run(run_name="AdaBoost_Baseline", nested=True) as ada_run:
    mlflow.sklearn.autolog()

    ada_model = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.1,
        random_state=42
    )
    ada_model.fit(X_train, y_train)

    y_pred_ada = ada_model.predict(X_test)
    y_pred_ada_labels = le.inverse_transform(y_pred_ada)

    cm_path = save_confusion_matrix(
        y_test_labels, y_pred_ada_labels, labels=le.classes_,
        title="AdaBoost Confusion Matrix", filename="adaboost_confusion_matrix"
    )
    mlflow.log_artifact(cm_path)

    report = classification_report(y_test_labels, y_pred_ada_labels)
    report_path = "artifacts/adaboost_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    ada_acc = accuracy_score(y_test_labels, y_pred_ada_labels)
    ada_f1 = f1_score(y_test_labels, y_pred_ada_labels, average="weighted")

    mlflow.log_metrics({
        "test_accuracy": ada_acc,
        "test_f1_score": ada_f1
    })

    print("Accuracy AdaBoost:", ada_acc)
    print("F1 AdaBoost:", ada_f1)

    model_results["AdaBoost"] = {
        "model": ada_model,
        "f1": ada_f1,
        "run_id": ada_run.info.run_id
    }


best_model_name = max(model_results, key=lambda m: model_results[m]["f1"])
best_model = model_results[best_model_name]["model"]
best_run_id = model_results[best_model_name]["run_id"]


joblib.dump(best_model, "artifacts/best_model.pkl")

with open("artifacts/best_run_id.txt", "w") as f:
    f.write(best_run_id)

with open("artifacts/best_model_name.txt", "w") as f:
    f.write(best_model_name)
