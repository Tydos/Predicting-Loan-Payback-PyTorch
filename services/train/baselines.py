"""
Standalone sklearn baseline evaluation.

Run:
    PYTHONPATH=. python services/train/baselines.py
"""

import logging
import os

import dagshub
import mlflow
import numpy as np
from core.metrics import ValidationMetrics, score, xy
from pipeline.prepare import load_and_prepare_data
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


def run_all_baselines(
    train_frame,
    val_frame,
    target: str,
) -> list[ValidationMetrics]:
    x_train, y_train = xy(train_frame, target)
    x_val, y_val = xy(val_frame, target)

    logging.info("Fitting majority-class baseline")
    majority_prob = np.full(y_val.shape, float(np.mean(y_train)))
    majority = score(
        "majority_class", y_val, majority_prob,
        f"Always predicts paid_back={majority_prob[0]:.3f}",
    )
    logging.info("majority_class — AUC=%.4f, F1=%.4f", majority.auc, majority.f1)

    logging.info("Fitting LogisticRegression baseline")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=11)
    lr.fit(x_train, y_train)
    lr_result = score("logistic_regression", y_val, lr.predict_proba(x_val)[:, 1])
    logging.info("logistic_regression — AUC=%.4f, F1=%.4f", lr_result.auc, lr_result.f1)

    return [majority, lr_result]


def log_baselines_to_mlflow(results: list[ValidationMetrics], mlflow_module) -> None:
    for result in results:
        logging.info("Logging baseline '%s' to MLflow", result.name)
        with mlflow_module.start_run(run_name=result.name):
            mlflow_module.log_param("model_type", "baseline")
            mlflow_module.log_param("baseline", result.name)
            mlflow_module.log_param("split", result.split)
            mlflow_module.log_metric("val_auc", result.auc)
            mlflow_module.log_metric("val_f1", result.f1)
            mlflow_module.log_metric("val_accuracy", result.accuracy)
            mlflow_module.log_metric("val_precision", result.precision)
            mlflow_module.log_metric("val_recall", result.recall)
            if result.notes:
                mlflow_module.set_tag("notes", result.notes)
    logging.info("Logged %d baseline runs to MLflow", len(results))


def main() -> None:
    try:
        data = load_and_prepare_data()

        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "pjawale")
        repo_name = os.getenv("DAGSHUB_REPO_NAME", "credit-scorer")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(data.config.mlflow.experiment_name)

        target = data.config.dataset.target_column
        logging.info("Train paid_back rate: %.3f", data.trainset[target].mean())

        results = run_all_baselines(data.trainset, data.valset, target)
        log_baselines_to_mlflow(results, mlflow)
        logging.info("Baseline evaluation complete")
    except Exception as exc:
        logging.error("Baseline evaluation failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
