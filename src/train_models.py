"""Train/test preparation utilities for the modelling pipeline.

Phase 5 deliberately stops before fitting estimators.  The functions here
define one chronological, leakage-safe data contract for every later model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "match_outcome"
TEST_SEASON = "2015/2016"
VALIDATION_SEASON = "2014/2015"
TRAINING_SEASONS = [
    "2009/2010",
    "2010/2011",
    "2011/2012",
    "2012/2013",
    "2013/2014",
]
FULL_TRAINING_SEASONS = TRAINING_SEASONS + [VALIDATION_SEASON]
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
RANDOM_STATE = 42
MODEL_NAMES = [
    "majority_baseline",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "knn",
]


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


@dataclass(frozen=True)
class TrainedModels:
    """Fitted estimators and their test predictions."""

    estimators: dict[str, object]
    predictions: pd.DataFrame


@dataclass(frozen=True)
class TuningResult:
    """Validation-season tuning results and selected model configurations."""

    selected_configs: dict[str, dict[str, object]]
    tried_configs: pd.DataFrame


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
    train_seasons: list[str] | None = None,
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
    if train_seasons is None:
        train_rows = ordered.loc[ordered["season"] != test_season].copy()
    else:
        train_rows = ordered.loc[ordered["season"].isin(train_seasons)].copy()
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


def _build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the same feature transformation contract for each estimator."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURE_COLUMNS),
            (
                "league",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(
    configs: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Create the baseline and four classifiers from selected configurations."""
    configs = configs or {}
    logistic_config = configs.get("logistic_regression", {})
    tree_config = configs.get("decision_tree", {})
    forest_config = configs.get("random_forest", {})
    knn_config = configs.get("knn", {})
    return {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent",
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", _build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        class_weight=logistic_config.get("class_weight", "balanced"),
                        C=logistic_config.get("C", 1.0),
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocessor", _build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=tree_config.get("max_depth", 8),
                        min_samples_leaf=tree_config.get("min_samples_leaf", 10),
                        class_weight=tree_config.get("class_weight", "balanced"),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=forest_config.get("n_estimators", 300),
                        max_depth=forest_config.get("max_depth", 10),
                        min_samples_leaf=forest_config.get("min_samples_leaf", 5),
                        class_weight=forest_config.get("class_weight", "balanced"),
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("preprocessor", _build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=knn_config.get("n_neighbors", 11),
                    ),
                ),
            ]
        ),
    }


def fit_models(
    prepared: PreparedData,
    configs: dict[str, dict[str, object]] | None = None,
) -> TrainedModels:
    """Fit every model on the same chronological training matrix."""
    estimators = build_models(configs)
    predictions = prepared.test_rows[["id", "date", "season"]].copy()
    predictions["actual_outcome"] = prepared.y_test.to_numpy()

    for name in MODEL_NAMES:
        estimator = estimators[name]
        estimator.fit(prepared.X_train, prepared.y_train)
        predictions[f"{name}_prediction"] = estimator.predict(prepared.X_test)

    return TrainedModels(estimators=estimators, predictions=predictions)


def tune_models(
    features: pd.DataFrame,
    validation_season: str = VALIDATION_SEASON,
) -> TuningResult:
    """Select hyperparameters on 2014/15 using macro F1 only."""
    tuning_train = chronological_split(
        features,
        test_season=validation_season,
        train_seasons=TRAINING_SEASONS,
    )
    candidates = {
        "logistic_regression": [
            {"C": value, "class_weight": "balanced"}
            for value in [0.1, 1.0, 10.0]
        ],
        "decision_tree": [
            {
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": "balanced",
            }
            for depth in [5, 8, 10]
            for leaf in [5, 10]
        ],
        "random_forest": [
            {
                "n_estimators": trees,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": "balanced",
            }
            for trees in [200, 300]
            for depth in [8, 10]
            for leaf in [1, 5]
        ],
        "knn": [{"n_neighbors": value} for value in [5, 11, 21]],
    }
    tried = []
    selected: dict[str, dict[str, object]] = {}

    for model_name, model_candidates in candidates.items():
        best_score = -1.0
        best_config: dict[str, object] | None = None
        best_row_index: int | None = None
        for config in model_candidates:
            estimator = build_models({model_name: config})[model_name]
            estimator.fit(tuning_train.X_train, tuning_train.y_train)
            prediction = estimator.predict(tuning_train.X_test)
            score = float(
                f1_score(
                    tuning_train.y_test,
                    prediction,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
            tried.append(
                {
                    "model": model_name,
                    "validation_season": validation_season,
                    "macro_f1": score,
                    "selected": False,
                    **config,
                }
            )
            if score > best_score:
                best_score = score
                best_config = config
                best_row_index = len(tried) - 1
        if best_config is None:
            raise ValueError(f"No tuning candidates found for {model_name}.")
        selected[model_name] = best_config
        if best_row_index is None:
            raise ValueError(f"No selected tuning row found for {model_name}.")
        tried[best_row_index]["selected"] = True

    return TuningResult(
        selected_configs=selected,
        tried_configs=pd.DataFrame(tried),
    )


def build_training_summary(
    prepared: PreparedData,
    configs: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Return reproducibility metadata for the fitted model set."""
    rows = []
    for name in MODEL_NAMES:
        rows.append(
            {
                "model": name,
                "training_rows": len(prepared.X_train),
                "test_rows": len(prepared.X_test),
                "feature_count_before_encoding": len(FEATURE_COLUMNS),
                "random_state": RANDOM_STATE if name != "majority_baseline" else "",
                "selected_config": str((configs or {}).get(name, {})),
            }
        )
    return pd.DataFrame(rows)


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> None:
    """Persist test predictions with traceability columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)