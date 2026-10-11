"""Evaluation utilities for the Phase 7 held-out test-set comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.train_models import LABELS, MODEL_NAMES


def evaluate_single_model(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Calculate aggregate and per-class metrics for one model."""
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    per_class = pd.DataFrame(
        [
            {
                "model": model_name,
                "class": label,
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        ]
    )
    return summary, per_class


def evaluate_all_models(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all model prediction columns using one fixed class ordering."""
    if "actual_outcome" not in predictions:
        raise ValueError("Predictions must contain an actual_outcome column.")

    y_true = predictions["actual_outcome"]
    summaries = []
    per_class_metrics = []
    for model_name in MODEL_NAMES:
        prediction_column = f"{model_name}_prediction"
        if prediction_column not in predictions:
            raise ValueError(f"Missing prediction column: {prediction_column}")
        summary, per_class = evaluate_single_model(
            y_true,
            predictions[prediction_column],
            model_name,
        )
        summaries.append(summary)
        per_class_metrics.append(per_class)

    comparison = pd.DataFrame(summaries).sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)
    return comparison, pd.concat(per_class_metrics, ignore_index=True)


def confusion_matrix_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Return a labelled confusion matrix as a serializable table."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis(
        "actual"
    ).reset_index().melt(
        id_vars="actual",
        var_name="predicted",
        value_name="count",
    ).assign(model=model_name)[
        ["model", "actual", "predicted", "count"]
    ]


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    model_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot side-by-side confusion matrices for selected models."""
    if len(model_names) != 2:
        raise ValueError("Exactly two models are required for the side-by-side plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = predictions["actual_outcome"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, model_name in zip(axes, model_names):
        prediction_column = f"{model_name}_prediction"
        if prediction_column not in predictions:
            raise ValueError(f"Missing prediction column: {prediction_column}")
        matrix = confusion_matrix(
            y_true,
            predictions[prediction_column],
            labels=LABELS,
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=axis,
        )
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_xlabel("Predicted outcome")
        axis.set_ylabel("Actual outcome")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_outputs(
    predictions: pd.DataFrame,
    tables_directory: str | Path,
    figures_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Evaluate predictions and persist all Phase 7 tables and figures."""
    tables_directory = Path(tables_directory)
    figures_directory = Path(figures_directory)
    tables_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    comparison, per_class_metrics = evaluate_all_models(predictions)
    comparison.to_csv(tables_directory / "model_comparison.csv", index=False)
    per_class_metrics.to_csv(
        tables_directory / "per_class_metrics.csv",
        index=False,
    )

    best_model = str(comparison.iloc[0]["model"])
    confusion_models = ["majority_baseline", best_model]
    for model_name in confusion_models:
        matrix = confusion_matrix_table(
            predictions["actual_outcome"],
            predictions[f"{model_name}_prediction"],
            model_name,
        )
        matrix.to_csv(
            tables_directory / f"confusion_matrix_{model_name}.csv",
            index=False,
        )

    confusion_figure = figures_directory / "evaluation_confusion_matrices.png"
    plot_confusion_matrices(predictions, confusion_models, confusion_figure)
    return comparison, per_class_metrics, best_model