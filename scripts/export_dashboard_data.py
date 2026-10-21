"""Export the files the dashboard reads, so it runs without the 313 MB database.

Run once from the repo root, after notebooks 05 and 06:

    python scripts/export_dashboard_data.py

Writes to outputs/dashboard/ (the pipeline's own outputs are never overwritten):

- predictions.csv   one row per 2015/16 test match: teams, league, actual result, every model's prediction (copied from
                    the official run, D-031) and the selected model's Home/Draw/Away probabilities
- team_lookup.csv   team ID, name, a unique display name and league (needs database.sqlite once)

Feature importance and the comparison tables are read directly from outputs/tables/, so they are not exported again.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dashboard.model_io import CLASS_ORDER, FEATURES_PATH, MODEL_NAME, load_model, predict_proba  # noqa: E402
from src.db import load_matches, load_teams  # noqa: E402
from src.train_models import TEST_SEASON, chronological_split, load_features  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "dashboard"
OFFICIAL_PREDICTIONS = REPO_ROOT / "outputs" / "predictions" / "test_predictions.csv"
MODELS = ["majority_baseline", "logistic_regression", "decision_tree", "random_forest", "knn"]


def export_predictions():
    official = pd.read_csv(OFFICIAL_PREDICTIONS)
    features = load_features(FEATURES_PATH)
    prepared = chronological_split(features, test_season=TEST_SEASON)

    bundle = load_model()
    proba = predict_proba(bundle, prepared.X_test)
    proba.index = prepared.test_rows["id"].to_numpy()

    meta = features.set_index("id").loc[official["id"], ["league_id", "home_team_api_id", "away_team_api_id"]]
    out = pd.DataFrame({
        "id": official["id"].to_numpy(),
        "date": official["date"].to_numpy(),
        "season": official["season"].to_numpy(),
        "league_id": meta["league_id"].to_numpy(),
        "home_team_api_id": meta["home_team_api_id"].to_numpy(),
        "away_team_api_id": meta["away_team_api_id"].to_numpy(),
        "actual": official["actual_outcome"].to_numpy(),
    })
    for model in MODELS:
        out[f"pred_{model}"] = official[f"{model}_prediction"].to_numpy()
    p = proba.loc[out["id"]]
    out["p_home"], out["p_draw"], out["p_away"] = (p[c].to_numpy() for c in CLASS_ORDER)

    # Checks: probabilities are valid and agree with the official predictions of the selected model
    sums = out[["p_home", "p_draw", "p_away"]].sum(axis=1)
    assert np.allclose(sums, 1.0), "probabilities do not sum to 1"
    argmax = pd.Series(np.array(CLASS_ORDER)[out[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)])
    agree = (argmax.to_numpy() == out[f"pred_{MODEL_NAME}"].to_numpy()).mean()
    assert agree == 1.0, f"selected-model probabilities disagree with the official predictions ({agree:.1%} match)"
    assert out["id"].is_unique and len(out) == len(official) and not out.isna().any().any()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "predictions.csv", index=False)
    print(f"predictions.csv: {len(out):,} test matches · model source: {bundle['source']} · "
          f"probabilities match official {MODEL_NAME} predictions: {agree:.0%}")


def export_team_lookup():
    matches = load_matches()            # needs data/raw/database.sqlite
    teams = load_teams().set_index("team_api_id")

    sides = pd.concat([
        matches[["home_team_api_id", "league_id", "league_name", "date"]].rename(columns={"home_team_api_id": "team_api_id"}),
        matches[["away_team_api_id", "league_id", "league_name", "date"]].rename(columns={"away_team_api_id": "team_api_id"}),
    ])
    leagues_per_team = sides.groupby("team_api_id")["league_id"].nunique()
    assert (leagues_per_team == 1).all(), "a team appears in more than one league"
    league = sides.drop_duplicates("team_api_id").set_index("team_api_id")[["league_id", "league_name"]]

    lookup = teams[["team_long_name", "team_short_name"]].join(league, how="inner").reset_index()
    # Two names each cover two different clubs (D-017), so the display name adds the ID where a name repeats
    repeated = lookup["team_long_name"].duplicated(keep=False)
    lookup["display_name"] = np.where(
        repeated, lookup["team_long_name"] + " (" + lookup["team_api_id"].astype(str) + ")", lookup["team_long_name"])
    assert lookup["team_api_id"].is_unique and lookup["display_name"].is_unique

    lookup = lookup[["team_api_id", "team_long_name", "team_short_name", "display_name", "league_id", "league_name"]]
    lookup = lookup.sort_values(["league_name", "display_name"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(OUT_DIR / "team_lookup.csv", index=False)
    print(f"team_lookup.csv: {len(lookup)} teams in {lookup['league_id'].nunique()} leagues · "
          f"{int(repeated.sum())} display names disambiguated")


if __name__ == "__main__":
    export_predictions()
    export_team_lookup()
