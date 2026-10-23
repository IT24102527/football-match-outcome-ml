"""Every file the dashboard reads, loaded once, checked and cached. This is the contract the pages import.

Each loader checks that the expected columns are present and that nothing is missing, so a renamed column or a
stale export fails with a clear message instead of a broken chart. Numbers are always read from these files and never
typed into page code, so the dashboard cannot disagree with the report.
"""

from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.model_io import CLASS_ORDER, MODEL_NAME, load_model  # noqa: E402

TABLES = REPO_ROOT / "outputs" / "tables"
DASH = REPO_ROOT / "outputs" / "dashboard"

FEATURED_MODEL = MODEL_NAME                     # Logistic Regression, the selected model (D-029, D-034)
MODELS = ["logistic_regression", "random_forest", "decision_tree", "knn", "majority_baseline"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "majority_baseline": "Majority baseline",
}
FEATURES = [
    "Home_Form_Last_5", "Away_Form_Last_5", "Home_Avg_Goals_Last_5", "Away_Avg_Goals_Last_5",
    "Home_Avg_Conceded_Last_5", "Away_Avg_Conceded_Last_5", "Home_Win_Rate", "Away_Win_Rate",
    "Form_Diff", "Abs_Form_Diff",
]

# file -> required columns
SCHEMA = {
    DASH / "predictions.csv": ["id", "date", "season", "league_id", "home_team_api_id", "away_team_api_id", "actual"]
                              + [f"pred_{m}" for m in MODELS] + ["p_home", "p_draw", "p_away"],
    DASH / "team_lookup.csv": ["team_api_id", "team_long_name", "team_short_name", "display_name", "league_id", "league_name"],
    REPO_ROOT / "data" / "processed" / "features.csv": ["id", "date", "season", "league_id", "home_team_api_id",
                                                        "away_team_api_id", "match_outcome"] + FEATURES,
    TABLES / "model_comparison.csv": ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
    TABLES / "per_class_metrics.csv": ["model", "class", "precision", "recall", "f1", "support"],
    TABLES / "class_distribution.csv": ["split", "class", "count", "percentage"],
    TABLES / "model_training_summary.csv": ["model", "training_rows", "test_rows", "selected_config"],
    TABLES / "hyperparameter_tuning.csv": ["model", "validation_season", "macro_f1", "selected"],
    TABLES / "permutation_importance.csv": ["model", "feature", "macro_f1_drop_mean", "macro_f1_drop_std"],
    TABLES / "logistic_regression_coefficients.csv": ["Home Win", "Draw", "Away Win"],
    TABLES / "confusion_matrix_logistic_regression.csv": ["model", "actual", "predicted", "count"],
    TABLES / "accuracy_by_form_gap.csv": ["Abs_Form_Diff", "matches", "share_of_matches", "logistic_regression",
                                          "random_forest", "majority_baseline"],
    TABLES / "draws_by_form_gap.csv": ["Abs_Form_Diff", "actual draw rate", "logistic_regression predicts Draw",
                                       "random_forest predicts Draw"],
}
FORM_GAP_ORDER = ["0–1 (even)", "2–3", "4–6", "7+ (mismatch)"]   # bands from notebook 07, in order
# Columns that are legitimately empty in some rows (e.g. hyperparameters that do not apply to a model)
OPTIONAL_GAPS = {TABLES / "hyperparameter_tuning.csv", TABLES / "model_training_summary.csv"}


class DataContractError(ValueError):
    """A dashboard input file is missing, or its columns or values are not what the pages expect."""


def _cache(func):
    """st.cache_data when running inside Streamlit, a plain in-process cache otherwise (scripts and tests)."""
    try:
        from streamlit.runtime import exists
        if exists():
            import streamlit as st
            return st.cache_data(show_spinner=False)(func)
    except ImportError:
        pass
    cached = lru_cache(maxsize=None)(func)

    def wrapper(*args, **kwargs):
        result = cached(*args, **kwargs)
        return result.copy() if hasattr(result, "copy") else result   # callers can't modify the cached frame
    wrapper.cache_clear = cached.cache_clear
    return wrapper


def _read(path, **kwargs):
    if not path.exists():
        hint = " Run `python scripts/export_dashboard_data.py` first." if path.parent == DASH else ""
        raise DataContractError(f"Missing file: {path.relative_to(REPO_ROOT)}.{hint}")
    df = pd.read_csv(path, **kwargs)
    missing = [c for c in SCHEMA.get(path, []) if c not in df.columns]
    if missing:
        raise DataContractError(f"{path.name} is missing columns: {missing}")
    if path not in OPTIONAL_GAPS and df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise DataContractError(f"{path.name} has missing values in: {bad}")
    return df


# ---------------- one loader per file ----------------
@_cache
def load_predictions():
    df = _read(DASH / "predictions.csv", parse_dates=["date"])
    if not ((df[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs() < 1e-6).all():
        raise DataContractError("predictions.csv: probabilities do not sum to 1")
    return df


@_cache
def load_team_lookup():
    return _read(DASH / "team_lookup.csv")


@_cache
def load_features():
    return _read(REPO_ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])


@_cache
def load_model_comparison():
    df = _read(TABLES / "model_comparison.csv")
    df["label"] = df["model"].map(MODEL_LABELS)
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


@_cache
def load_per_class_metrics():
    df = _read(TABLES / "per_class_metrics.csv")
    df["label"] = df["model"].map(MODEL_LABELS)
    return df


@_cache
def load_class_distribution():
    return _read(TABLES / "class_distribution.csv")


@_cache
def load_training_summary():
    return _read(TABLES / "model_training_summary.csv")


@_cache
def load_hyperparameter_tuning():
    return _read(TABLES / "hyperparameter_tuning.csv")


@_cache
def load_permutation_importance():
    return _read(TABLES / "permutation_importance.csv")


def _form_gap_table(name):
    df = _read(TABLES / name)
    unexpected = set(df["Abs_Form_Diff"]) - set(FORM_GAP_ORDER)
    if unexpected:
        raise DataContractError(f"{name} has unexpected form-gap bands: {sorted(unexpected)}")
    df["Abs_Form_Diff"] = pd.Categorical(df["Abs_Form_Diff"], categories=FORM_GAP_ORDER, ordered=True)
    return df.sort_values("Abs_Form_Diff").reset_index(drop=True)


@_cache
def load_accuracy_by_form_gap():
    """Test-season accuracy per pre-match form-gap band (notebook 07): rates from 0 to 1, bands in order."""
    return _form_gap_table("accuracy_by_form_gap.csv")


@_cache
def load_draws_by_form_gap():
    """Actual draw rate and how often each model predicts a draw, per form-gap band (notebook 07)."""
    return _form_gap_table("draws_by_form_gap.csv")


@_cache
def load_lr_coefficients():
    df = _read(TABLES / "logistic_regression_coefficients.csv", index_col=0)
    df.index.name = "feature"
    return df


# ---------------- derived views used by several pages ----------------
@_cache
def load_test_matches():
    """One row per 2015/16 test match: teams and league by name, the 10 features, every prediction, the
    featured model's probabilities, and whether it was right."""
    pred = load_predictions()
    teams = load_team_lookup().set_index("team_api_id")
    feats = load_features().set_index("id")[FEATURES]
    df = pred.join(feats, on="id")
    if df[FEATURES].isna().any().any():
        raise DataContractError("Some test matches have no features in features.csv")
    df["home_team"] = df["home_team_api_id"].map(teams["display_name"])
    df["away_team"] = df["away_team_api_id"].map(teams["display_name"])
    league_names = teams.drop_duplicates("league_id").set_index("league_id")["league_name"]
    df["league_name"] = df["league_id"].map(league_names)
    if df[["home_team", "away_team", "league_name"]].isna().any().any():
        raise DataContractError("Some test matches have a team or league missing from team_lookup.csv")
    df["fixture"] = df["home_team"] + " vs " + df["away_team"]
    df["correct"] = df[f"pred_{FEATURED_MODEL}"] == df["actual"]
    return df.sort_values(["date", "id"]).reset_index(drop=True)


def confusion_matrix(model):
    """Counts of actual (rows) vs predicted (columns) outcomes on the test season, in CLASS_ORDER."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    pred = load_predictions()
    table = pd.crosstab(pred["actual"], pred[f"pred_{model}"]).reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
    table.index.name, table.columns.name = "Actual", "Predicted"
    return table


def get_model():
    """The featured model bundle, loaded or refitted once per session (see model_io)."""
    try:
        from streamlit.runtime import exists
        if exists():
            import streamlit as st
            return st.cache_resource(show_spinner="Loading the model…")(load_model)()
    except ImportError:
        pass
    return _model_once()


@lru_cache(maxsize=1)
def _model_once():
    return load_model()
