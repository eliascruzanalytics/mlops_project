import json
import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> tf.keras.Model:
    """Load trained Keras model."""

    model_path = "models/model.keras"

    logger.info(f"Loading model from {model_path}")

    model = tf.keras.models.load_model(model_path)

    return model


def load_encoder() -> OneHotEncoder:
    """Load fitted OneHotEncoder."""

    encoder_path = "artifacts/[target]_one_hot_encoder.joblib"

    logger.info(f"Loading encoder from {encoder_path}")

    encoder = joblib.load(encoder_path)

    return encoder


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load processed test dataset."""

    data_path = "data/processed/test_processed.csv"

    logger.info(f"Loading test data from {data_path}")

    data = pd.read_csv(data_path)

    X = data.drop("target", axis=1)

    y = data["target"]

    return X, y


def evaluate_model(
    model: tf.keras.Model,
    encoder: OneHotEncoder,
    X: pd.DataFrame,
    y_true: pd.Series,
) -> None:
    """
    Evaluate trained model and log metrics to MLflow.
    """

    # Set experiment
    mlflow.set_experiment("ml_classification")

    experiment = mlflow.get_experiment_by_name(
        "ml_classification"
    )

    if experiment is None:
        raise Exception(
            "Experiment 'ml_classification' not found."
        )

    # Get latest run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if runs.empty:
        raise Exception(
            "No MLflow runs found for experiment."
        )

    run_id = runs.iloc[0].run_id

    logger.info(f"Using MLflow run_id: {run_id}")

    with mlflow.start_run(run_id=run_id):

        # Generate predictions
        logger.info("Generating predictions...")

        y_pred_proba = model.predict(X)

        y_pred = np.argmax(y_pred_proba, axis=1)

        # Encode true labels
        y_true_encoded = encoder.transform(
            y_true.values.reshape(-1, 1)
        )

        y_true_classes = np.argmax(
            y_true_encoded,
            axis=1
        )

        # Generate evaluation metrics
        logger.info("Calculating evaluation metrics...")

        report = classification_report(
            y_true_classes,
            y_pred,
            output_dict=True,
        )

        cm = confusion_matrix(
            y_true_classes,
            y_pred,
        ).tolist()

        evaluation = {
            "classification_report": report,
            "confusion_matrix": cm,
        }

        # Create metrics directory
        os.makedirs("metrics", exist_ok=True)

        # Save metrics locally
        evaluation_path = "metrics/evaluation.json"

        logger.info(
            f"Saving evaluation metrics to {evaluation_path}"
        )

        with open(evaluation_path, "w") as f:
            json.dump(evaluation, f, indent=2)

        # Log metrics to MLflow
        logger.info("Logging metrics to MLflow...")

        mlflow.log_metrics(
            {
                "test_accuracy": report["accuracy"],
                "test_precision_weighted": report[
                    "weighted avg"
                ]["precision"],
                "test_recall_weighted": report[
                    "weighted avg"
                ]["recall"],
                "test_f1_weighted": report[
                    "weighted avg"
                ]["f1-score"],
            }
        )

        # Log evaluation artifact
        mlflow.log_artifact(evaluation_path)

        logger.info("Model evaluation completed successfully")


def main() -> None:
    """Main evaluation pipeline."""

    model = load_model()

    encoder = load_encoder()

    X, y = load_test_data()

    evaluate_model(
        model=model,
        encoder=encoder,
        X=X,
        y_true=y,
    )


if __name__ == "__main__":
    main()