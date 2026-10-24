"""Tests for the dashboard's data and model layer (Member A). Run from the repo root: `pytest tests`."""

import numpy as np
import pandas as pd
import pytest

from dashboard import data_access as da
from dashboard.model_io import CLASS_ORDER, MODEL_PATH, explain, load_model, predict_proba, refit_model, typical_values
from src.train_models import TEST_SEASON, chronological_split


@pytest.fixture(scope="module")
def bundle():
    return load_model()


@pytest.fixture(scope="module")
def matches():
    return da.load_test_matches()


# ---------------- loaders ----------------
@pytest.mark.parametrize("loader", [
    "load_predictions", "load_team_lookup", "load_features", "load_model_comparison", "load_per_class_metrics",
    "load_class_distribution", "load_training_summary", "load_hyperparameter_tuning", "load_permutation_importance",
    "load_lr_coefficients", "load_accuracy_by_form_gap", "load_draws_by_form_gap", "load_test_matches",
])
def test_every_loader_returns_rows(loader):
    assert len(getattr(da, loader)()) > 0


def test_missing_file_gives_clear_error():
    with pytest.raises(da.DataContractError, match="export_dashboard_data.py"):
        da._read(da.DASH / "does_not_exist.csv")


def test_form_gap_bands_come_back_in_order():
    for table in (da.load_accuracy_by_form_gap(), da.load_draws_by_form_gap()):
        assert list(table["Abs_Form_Diff"]) == da.FORM_GAP_ORDER


def test_test_matches_cover_the_whole_test_season(matches):
    assert len(matches) == 3326 and matches["season"].eq(TEST_SEASON).all()
    assert matches[["home_team", "away_team", "league_name"]].notna().all().all()


# ---------------- predictions and probabilities ----------------
def test_probabilities_sum_to_one(matches):
    assert np.allclose(matches[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)


def test_exported_probabilities_agree_with_official_predictions(matches):
    top = np.array(CLASS_ORDER)[matches[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)]
    assert (top == matches[f"pred_{da.FEATURED_MODEL}"].to_numpy()).all()


def test_live_model_reproduces_exported_probabilities(bundle, matches):
    live = predict_proba(bundle, matches[bundle["feature_columns"]].astype(float))
    assert np.allclose(live.to_numpy(), matches[["p_home", "p_draw", "p_away"]].to_numpy(), atol=1e-9)


def test_predict_proba_maps_classes_by_name(bundle, matches):
    proba = predict_proba(bundle, matches[bundle["feature_columns"]].head(5).astype(float))
    assert list(proba.columns) == CLASS_ORDER          # not scikit-learn's alphabetical order


def test_known_match_prediction(bundle, matches):
    row = matches[(matches["fixture"] == "Chelsea vs Liverpool") & (matches["date"] == "2015-10-31")]
    assert len(row) == 1
    proba = predict_proba(bundle, row[bundle["feature_columns"]].astype(float)).iloc[0]
    assert proba.idxmax() == row["pred_logistic_regression"].iloc[0] == "Home Win"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="saved model not present (fresh clone)")
def test_refit_matches_saved_model(bundle, matches):
    X = matches[bundle["feature_columns"]].astype(float)
    assert np.allclose(predict_proba(refit_model(), X).to_numpy(), predict_proba(bundle, X).to_numpy(), atol=1e-12)


def test_confusion_matrix_matches_notebook_06():
    saved = pd.read_csv(da.TABLES / "confusion_matrix_logistic_regression.csv")
    saved = saved.pivot(index="actual", columns="predicted", values="count").loc[CLASS_ORDER, CLASS_ORDER]
    assert (da.confusion_matrix("logistic_regression").to_numpy() == saved.to_numpy()).all()


# ---------------- explanation ----------------
def test_explain_contributions_rebuild_the_model_score(bundle, matches):
    """Contributions of every input plus the intercept must equal the model's score (logit) for each outcome."""
    model = bundle["model"]
    pre, clf = model.named_steps["preprocessor"], model.named_steps["classifier"]
    sample = matches.sample(20, random_state=0)
    for _, row in sample.iterrows():
        X = pd.DataFrame([row[bundle["feature_columns"]]]).astype(float)
        scores = clf.decision_function(pre.transform(X))[0]
        for k, outcome in enumerate(clf.classes_):
            total = explain(bundle, X, outcome, top=None, include_league=True).sum() + clf.intercept_[k]
            assert total == pytest.approx(scores[k], abs=1e-9)


def test_explain_rejects_more_than_one_match(bundle, matches):
    X = matches[bundle["feature_columns"]].head(2).astype(float)
    with pytest.raises(ValueError, match="exactly one match"):
        explain(bundle, X, "Home Win")


def test_typical_values_are_the_training_averages(bundle):
    prepared = chronological_split(da.load_features(), test_season=TEST_SEASON)
    typical = typical_values(bundle)
    assert np.allclose(typical.to_numpy(), prepared.X_train[typical.index].mean().to_numpy())
