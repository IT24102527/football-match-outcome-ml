import numpy as np
import pandas as pd


def add_match_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the match outcome target:
    Home Win / Draw / Away Win.
    """
    df = df.copy()

    conditions = [
        df["home_team_goal"] > df["away_team_goal"],
        df["home_team_goal"] == df["away_team_goal"],
        df["home_team_goal"] < df["away_team_goal"],
    ]

    choices = ["Home Win", "Draw", "Away Win"]

    df["match_outcome"] = np.select(
        conditions,
        choices,
        default="Unknown",
    )

    assert (df["match_outcome"] != "Unknown").all(), \
        "Unexpected match outcome found."

    return df


def sort_matches_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort matches by date and match ID in chronological order.
    """
    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    )

    return df.sort_values(
        ["date", "id"]
    ).reset_index(drop=True)