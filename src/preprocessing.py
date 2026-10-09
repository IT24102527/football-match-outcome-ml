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


def remove_excluded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns excluded from modelling:
    - In-match event/XML data
    - Player lineup and X/Y position data
    - Betting odds
    """
    df = df.copy()

    # In-match event/XML columns
    event_columns = [
        "goal",
        "shoton",
        "shotoff",
        "foulcommit",
        "card",
        "cross",
        "corner",
        "possession",
    ]

    # Player lineup, X/Y position and player ID columns
    lineup_columns = [
        column
        for column in df.columns
        if column.startswith(
            (
                "home_player_X",
                "away_player_X",
                "home_player_Y",
                "away_player_Y",
                "home_player_",
                "away_player_",
            )
        )
    ]

    # Betting odds columns
    odds_columns = [
        "B365H", "B365D", "B365A",
        "BWH", "BWD", "BWA",
        "IWH", "IWD", "IWA",
        "LBH", "LBD", "LBA",
        "PSH", "PSD", "PSA",
        "WHH", "WHD", "WHA",
        "SJH", "SJD", "SJA",
        "VCH", "VCD", "VCA",
        "GBH", "GBD", "GBA",
        "BSH", "BSD", "BSA",
    ]

    columns_to_drop = (
        event_columns
        + lineup_columns
        + odds_columns
    )

    df = df.drop(
        columns=columns_to_drop,
        errors="raise"
    )

    return df