from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import build_feature_config, build_preprocessor, make_customer_feature_frame
from src.target_engineering import build_customer_target


def build_training_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    features = make_customer_feature_frame(raw_df)
    target, _, _, _ = build_customer_target(raw_df)
    out = features.merge(target, on="CustomerId", how="inner")
    if out["is_high_risk"].isna().any():
        raise ValueError("Missing is_high_risk after merge.")
    return out


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main(data_path: str, experiment_name: str, random_state: int):
    raw = pd.read_csv(data_path)
    train_df = build_training_table(raw)

    y = train_df["is_high_risk"].astype(int)
    X = train_df.drop(columns=["CustomerId", "is_high_risk"])

    config = build_feature_config()
    preprocessor = build_preprocessor(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    mlflow.set_experiment(experiment_name)

    models = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    ]

    best_auc = -1.0
    best_run = None

    for name, estimator, grid in models:
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=5, n_jobs=-1)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model_type", name)
            mlflow.log_param("random_state", random_state)

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            for k, v in search.best_params_.items():
                mlflow.log_param(k, v)

            mlflow.sklearn.log_model(best_model, artifact_path="model")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run = run.info.run_id

    print("Best ROC-AUC:", best_auc)
    print("Best run id:", best_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/raw/data.csv")
    p.add_argument("--experiment_name", default="credit-risk-task5")
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    main(args.data_path, args.experiment_name, args.random_state)
