"""
modelling.py — MLflow Project entry point (Kriteria 3)
Melatih model klasifikasi pekerjaan dengan parameter CLI.
Tracking disimpan ke DagsHub via environment variables.

Usage:
    python modelling.py
    python modelling.py --dataset jobs_preprocessing.csv --max_features 5000 --C 1.0 --max_iter 1000
"""

import argparse
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

dagshub.init(
    repo_owner="adinplb",
    repo_name="Eksperimen_SML_Muhammad-Adin-Palimbani",
    mlflow=True,
)

LABEL_NAMES = [
    "Education", "Engineering", "Finance & Accounting", "Healthcare",
    "Human Resources", "IT & Software", "Marketing & Sales",
    "Operations & Supply Chain",
]


def main(dataset_path, max_features, C, max_iter):
    df = pd.read_csv(dataset_path)
    X = df["text_clean"].fillna("")
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("Indonesian Job Classification - CI")

    with mlflow.start_run(run_name="ci_training"):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, sublinear_tf=True, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)

        mlflow.sklearn.log_model(pipeline, "model")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
            mlflow.log_artifact(report_path)

        run_id = mlflow.active_run().info.run_id
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Training selesai. Run ID: {run_id}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="jobs_preprocessing.csv")
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()
    main(args.dataset, args.max_features, args.C, args.max_iter)
