import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def evaluate_model(
    model_path,
    x_test_seq,
    y_test_seq,
    threshold=0.5
):
    """
    Evaluate the saved LSTM model on test data.
    """

    try:
        # Load saved model
        model = tf.keras.models.load_model(
            model_path
        )

        # Generate prediction probabilities
        y_pred_prob = model.predict(
            x_test_seq
        ).ravel()

        # Convert probabilities to class labels
        y_pred = (
            y_pred_prob >= threshold
        ).astype(int)

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_test_seq,
                y_pred,
                target_names=[
                    "No Irrigation",
                    "Irrigation"
                ]
            )
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_test_seq,
            y_pred
        )

        print("\nConfusion Matrix:")
        print(cm)

        # ROC-AUC
        roc_auc = roc_auc_score(
            y_test_seq,
            y_pred_prob
        )

        print(
            "\nROC-AUC:",
            round(roc_auc, 4)
        )

        # PR-AUC
        pr_auc = average_precision_score(
            y_test_seq,
            y_pred_prob
        )

        print(
            "PR-AUC:",
            round(pr_auc, 4)
        )

        return {
            "classification_report": classification_report(
                y_test_seq,
                y_pred,
                target_names=[
                    "No Irrigation",
                    "Irrigation"
                ],
                output_dict=True
            ),
            "confusion_matrix": cm,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }

    except Exception as e:
        print(
            "Error occurred during model evaluation:",
            e
        )
        raise