"""Train/test preparation utilities for the modelling pipeline.

Phase 5 deliberately stops before fitting estimators.  The functions here
define one chronological, leakage-safe data contract for every later model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "match_outcome"
TEST_SEASON = "2015/2016"
LABELS = ["Home Win", "Draw", "Away Win"]

NUMERIC_FEATURE_COLUMNS = [
    "Home_Form_Last_5",
    "Away_Form_Last_5",
    "Home_Avg_Goals_Last_5",
    "Away_Avg_Goals_Last_5",
    "Home_Avg_Conceded_Last_5",
    "Away_Avg_Conceded_Last_5",
    "Home_Win_Rate",
    "Away_Win_Rate",
    "Form_Diff",
    "Abs_Form_Diff",
]
CATEGORICAL_FEATURE_COLUMNS = ["league_id"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
IDENTIFIER_COLUMNS = ["id", "date", "season"]


@dataclass(frozen=True)
class PreparedData:
    """Chronologically split data and metadata shared by all models."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_rows: pd.DataFrame
    test_rows: pd.DataFrame
    imputation_values: dict[str, float]


def load_features(path: str | Path) -> pd.DataFrame:
    """Load and validate the engineered feature dataset."""
    features = pd.read_csv(path, parse_dates=["date"])
    required = set(FEATURE_COLUMNS + IDENTIFIER_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")
    if features.empty:
        raise ValueError("Feature dataset is empty.")
    return features


def summarize_class_distribution(
    values: pd.Series,
    split_name: str,
) -> pd.DataFrame:
    """Return counts and percentages for all outcome labels."""
    counts = values.value_counts().reindex(LABELS, fill_value=0)
    total = int(counts.sum())
    if total == 0:
        raise ValueError(f"{split_name} contains no target values.")
    return pd.DataFrame(
        {
            "split": split_name,
            "class": LABELS,
            "count": counts.to_numpy(),
            "percentage": (counts.to_numpy() / total * 100).round(4),
        }
    )


def chronological_split(
    features: pd.DataFrame,
    test_season: str = TEST_SEASON,
) -> PreparedData:
    """Split by season and impute historical features using training rows only."""
    required = set(FEATURE_COLUMNS + IDENTIFIER_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {missing}")

    ordered = features.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered = ordered.sort_values(["date", "id"]).reset_index(drop=True)

    test_rows = ordered.loc[ordered["season"] == test_season].copy()
    train_rows = ordered.loc[ordered["season"] != test_season].copy()
    if train_rows.empty or test_rows.empty:
        raise ValueError("Both chronological train and test sets must be non-empty.")
    if train_rows["date"].max() >= test_rows["date"].min():
        raise ValueError("Chronological split overlaps or reverses the time order.")

    imputation_values = (
        train_rows[NUMERIC_FEATURE_COLUMNS].median(skipna=True).to_dict()
    )
    if any(pd.isna(value) for value in imputation_values.values()):
        raise ValueError("Training imputation values contain missing medians.")

    train_rows.loc[:, NUMERIC_FEATURE_COLUMNS] = train_rows[
        NUMERIC_FEATURE_COLUMNS
    ].fillna(imputation_values)
    test_rows.loc[:, NUMERIC_FEATURE_COLUMNS] = test_rows[
        NUMERIC_FEATURE_COLUMNS
    ].fillna(imputation_values)

    X_train = train_rows[FEATURE_COLUMNS].copy()
    X_test = test_rows[FEATURE_COLUMNS].copy()
    y_train = train_rows[TARGET_COLUMN].copy()
    y_test = test_rows[TARGET_COLUMN].copy()

    if set(train_rows["id"]).intersection(test_rows["id"]):
        raise ValueError("A match ID appears in both train and test sets.")
    if not set(y_train.unique()).issubset(LABELS) or not set(y_test.unique()).issubset(
        LABELS
    ):
        raise ValueError("Unexpected target label found.")

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_rows=train_rows,
        test_rows=test_rows,
        imputation_values=imputation_values,
    )


def build_split_summary(prepared: PreparedData) -> pd.DataFrame:
    """Return row counts and temporal boundaries for the split."""
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(prepared.train_rows),
                "season_start": prepared.train_rows["season"].min(),
                "season_end": prepared.train_rows["season"].max(),
                "date_start": prepared.train_rows["date"].min().date().isoformat(),
                "date_end": prepared.train_rows["date"].max().date().isoformat(),
            },
            {
                "split": "test",
                "rows": len(prepared.test_rows),
                "season_start": prepared.test_rows["season"].min(),
                "season_end": prepared.test_rows["season"].max(),
                "date_start": prepared.test_rows["date"].min().date().isoformat(),
                "date_end": prepared.test_rows["date"].max().date().isoformat(),
            },
        ]
    )