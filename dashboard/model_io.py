"""Load the selected model (Logistic Regression, D-029) and turn its output into named class probabilities.

The saved model (`models/final_logistic_regression.joblib`, written by notebook 05) is git-ignored and tied to one
scikit-learn version. When it is missing or was saved with a different version, the model is refitted from the
committed `features.csv` with the configuration selected on the 2014/15 validation season. Logistic Regression is
deterministic, so the refit reproduces the official predictions exactly on any machine.
"""

import ast
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.train_models import FEATURE_COLUMNS, TEST_SEASON, build_models, chronological_split, load_features

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "final_logistic_regression.joblib"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.csv"
TRAINING_SUMMARY_PATH = REPO_ROOT / "outputs" / "tables" / "model_training_summary.csv"

MODEL_NAME = "logistic_regression"
CLASS_ORDER = ["Home Win", "Draw", "Away Win"]   # order used everywhere in the report and dashboard


def selected_config(model_name=MODEL_NAME):
    """The configuration chosen on the 2014/15 validation season (notebook 05)."""
    summary = pd.read_csv(TRAINING_SUMMARY_PATH)
    row = summary.loc[summary["model"] == model_name]
    if row.empty:
        raise ValueError(f"No selected configuration for {model_name!r} in {TRAINING_SUMMARY_PATH.name}")
    return ast.literal_eval(row["selected_config"].iloc[0])


def refit_model():
    """Refit the selected Logistic Regression on 2009/10–2014/15, exactly as notebook 05 does."""
    prepared = chronological_split(load_features(FEATURES_PATH), test_season=TEST_SEASON)
    config = selected_config()
    model = build_models({MODEL_NAME: config})[MODEL_NAME]
    model.fit(prepared.X_train, prepared.y_train)
    return {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "imputation_values": prepared.imputation_values,
        "config": config,
        "trained_on_seasons": sorted(prepared.train_rows["season"].unique()),
        "sklearn_version": sklearn.__version__,
        "source": "refit",
    }


def load_model():
    """The saved model if it exists and matches this scikit-learn version; otherwise a refit."""
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if bundle.get("sklearn_version") == sklearn.__version__:
            bundle["source"] = "saved"
            return bundle
    return refit_model()


def predict_proba(bundle, X):
    """Class probabilities as a DataFrame with columns in CLASS_ORDER.

    `predict_proba` returns columns in `model.classes_` order, which is alphabetical (Away Win, Draw, Home Win),
    so the columns are mapped by name, never by position.
    """
    model = bundle["model"]
    X = X[bundle["feature_columns"]]
    raw = pd.DataFrame(model.predict_proba(X), columns=list(model.classes_), index=X.index)
    missing = set(CLASS_ORDER) - set(raw.columns)
    if missing:
        raise ValueError(f"Model is missing classes: {sorted(missing)}")
    proba = raw[CLASS_ORDER]
    if not ((proba.sum(axis=1) - 1).abs() < 1e-9).all():
        raise ValueError("Predicted probabilities do not sum to 1.")
    return proba


def explain(bundle, X, outcome, top=3):
    """The inputs that pushed one match's prediction most towards (positive) or away from (negative) `outcome`.

    For Logistic Regression each input's contribution is its coefficient for that outcome times its standardised
    value. League indicators are left out, so the drivers are the football features.
    """
    model = bundle["model"]
    pre, clf = model.named_steps["preprocessor"], model.named_steps["classifier"]
    row = X[bundle["feature_columns"]].iloc[[0]]
    z = pd.Series(pre.transform(row)[0], index=pre.get_feature_names_out())
    coef = pd.Series(clf.coef_[list(clf.classes_).index(outcome)], index=z.index)
    contrib = (coef * z).drop([c for c in z.index if c.startswith("league_id")])
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)
