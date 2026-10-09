import numpy as np
import pandas as pd

from src.db import load_matches
from src.preprocessing import (
    add_match_outcome,
    remove_excluded_columns,
    remove_warmup_season,
    sort_matches_chronologically,
)


def build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level data into two team-level rows per match.

    Each match produces:
    - one row for the home team
    - one row for the away team
    """

    df = df.copy()

    columns = [
        "id",
        "date",
        "season",
        "league_id",
        "home_team_api_id",
        "away_team_api_id",
        "home_team_goal",
        "away_team_goal",
    ]

    df = df[columns]

    # Home-team perspective
    home = df[
        [
            "id",
            "date",
            "season",
            "league_id",
            "home_team_api_id",
            "home_team_goal",
            "away_team_goal",
        ]
    ].rename(
        columns={
            "home_team_api_id": "team_api_id",
            "home_team_goal": "goals_for",
            "away_team_goal": "goals_against",
        }
    )

    # Away-team perspective
    away = df[
        [
            "id",
            "date",
            "season",
            "league_id",
            "away_team_api_id",
            "away_team_goal",
            "home_team_goal",
        ]
    ].rename(
        columns={
            "away_team_api_id": "team_api_id",
            "away_team_goal": "goals_for",
            "home_team_goal": "goals_against",
        }
    )

    home["venue"] = "home"
    away["venue"] = "away"

    # Combine home and away records
    team_matches = pd.concat(
        [home, away],
        ignore_index=True,
    )

    # Sort chronologically for each team
    team_matches = team_matches.sort_values(
        ["team_api_id", "date", "id"]
    ).reset_index(drop=True)

    # Points:
    # Win = 3
    # Draw = 1
    # Loss = 0
    team_matches["points"] = np.select(
        [
            team_matches["goals_for"] > team_matches["goals_against"],
            team_matches["goals_for"] == team_matches["goals_against"],
        ],
        [3, 1],
        default=0,
    )

    # Win indicator
    team_matches["win"] = (
        team_matches["goals_for"]
        > team_matches["goals_against"]
    ).astype(int)

    return team_matches


def add_historical_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical team features using only matches
    played before the current match.

    Features:
    - Points from previous 5 matches
    - Average goals scored in previous 5 matches
    - Average goals conceded in previous 5 matches
    - Home/away-specific historical win rate

    shift(1) is used to exclude the current match
    and prevent target leakage.
    """

    team_matches = team_matches.copy()

    # Group by team for form and goal-based features.
    grouped = team_matches.groupby("team_api_id")

    # Points from previous 5 matches
    team_matches["form_last_5"] = grouped["points"].transform(
        lambda s: s.shift(1).rolling(5).sum()
    )

    # Average goals scored in previous 5 matches
    team_matches["avg_goals_last_5"] = grouped["goals_for"].transform(
        lambda s: s.shift(1).rolling(5).mean()
    )

    # Average goals conceded in previous 5 matches
    team_matches["avg_conceded_last_5"] = grouped["goals_against"].transform(
        lambda s: s.shift(1).rolling(5).mean()
    )

    # Home/away-specific historical win rate
    venue_grouped = team_matches.groupby(
        ["team_api_id", "venue"]
    )

    team_matches["win_rate"] = venue_grouped["win"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    return team_matches


def to_match_level(team_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level historical features back into
    one row per match.

    Creates separate Home and Away historical features.
    """

    feature_columns = [
        "id",
        "form_last_5",
        "avg_goals_last_5",
        "avg_conceded_last_5",
        "win_rate",
    ]

    # Home-team historical features
    home = (
        team_matches[
            team_matches["venue"] == "home"
        ][feature_columns]
        .rename(
            columns={
                "form_last_5": "Home_Form_Last_5",
                "avg_goals_last_5": "Home_Avg_Goals_Last_5",
                "avg_conceded_last_5": "Home_Avg_Conceded_Last_5",
                "win_rate": "Home_Win_Rate",
            }
        )
    )

    # Away-team historical features
    away = (
        team_matches[
            team_matches["venue"] == "away"
        ][feature_columns]
        .rename(
            columns={
                "form_last_5": "Away_Form_Last_5",
                "avg_goals_last_5": "Away_Avg_Goals_Last_5",
                "avg_conceded_last_5": "Away_Avg_Conceded_Last_5",
                "win_rate": "Away_Win_Rate",
            }
        )
    )

    # Combine home and away features
    match_features = home.merge(
        away,
        on="id",
        how="inner",
    )

    # Difference between home and away recent form
    match_features["Form_Diff"] = (
        match_features["Home_Form_Last_5"]
        - match_features["Away_Form_Last_5"]
    )

    # Absolute form difference
    match_features["Abs_Form_Diff"] = (
        match_features["Form_Diff"].abs()
    )

    return match_features


def build_match_level_feature_dataset() -> pd.DataFrame:
    """Build the final match-level feature dataset without imputing missing values."""
    matches = load_matches()
    matches = sort_matches_chronologically(matches)
    matches = add_match_outcome(matches)
    matches = remove_excluded_columns(matches)

    team_matches = build_team_long(matches)
    team_matches = add_historical_features(team_matches)
    match_features = to_match_level(team_matches)

    matches = matches.merge(match_features, on="id", how="inner")
    matches = remove_warmup_season(matches)

    final_columns = [
        "id",
        "date",
        "season",
        "league_id",
        "home_team_api_id",
        "away_team_api_id",
        "match_outcome",
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
    return matches.loc[:, final_columns]


def impute_historical_features(
    df: pd.DataFrame,
    fill_values: dict,
) -> pd.DataFrame:
    """
    Fill missing historical feature values using supplied
    training-derived fill values.

    The fill values should be calculated from the training data
    only to avoid temporal leakage.
    """

    df = df.copy()

    feature_columns = [
        "Home_Form_Last_5",
        "Away_Form_Last_5",
        "Home_Avg_Goals_Last_5",
        "Away_Avg_Goals_Last_5",
        "Home_Avg_Conceded_Last_5",
        "Away_Avg_Conceded_Last_5",
        "Home_Win_Rate",
        "Away_Win_Rate",
    ]

    for column in feature_columns:
        if column in fill_values:
            df[column] = df[column].fillna(fill_values[column])

    # Recalculate derived features after imputation.
    df["Form_Diff"] = (
        df["Home_Form_Last_5"]
        - df["Away_Form_Last_5"]
    )

    df["Abs_Form_Diff"] = (
        df["Form_Diff"].abs()
    )

    return df