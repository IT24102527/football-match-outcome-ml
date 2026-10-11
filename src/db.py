"""Shared access to the European Soccer Database (data/raw/database.sqlite).

Every notebook and script loads data through these functions so all members
query the database the same way.

Usage from a notebook in notebooks/:

    import sys; sys.path.append("..")
    from src.db import load_matches, load_teams, load_team_attributes

    matches = load_matches()
"""

from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "raw" / "database.sqlite"


def get_connection(db_path=DB_PATH):
    """Open a read-only connection so the raw database can never be modified."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "See README.md > Dataset for download instructions."
        )
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def query(sql, params=None, db_path=DB_PATH):
    """Run a SQL query and return the result as a DataFrame."""
    with closing(get_connection(db_path)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def list_tables(db_path=DB_PATH):
    """Return every table in the database with its row and column count."""
    with closing(get_connection(db_path)) as conn:
        names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn
        )["name"]
        rows = []
        for name in names:
            n_rows = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            n_cols = len(conn.execute(f'PRAGMA table_info("{name}")').fetchall())
            rows.append({"table": name, "rows": n_rows, "columns": n_cols})
    return pd.DataFrame(rows)


def load_table(name, db_path=DB_PATH):
    """Load a whole table by name (validated against the database's table list)."""
    tables = set(list_tables(db_path)["table"])
    if name not in tables:
        raise ValueError(f"Unknown table {name!r}. Available: {sorted(tables)}")
    return query(f'SELECT * FROM "{name}"', db_path=db_path)


def load_matches(with_names=True, parse_dates=True, db_path=DB_PATH):
    """Load the Match table (one row per league fixture).

    with_names:  add country_name, league_name, home_team_name and away_team_name
                 so matches are human-readable during EDA.
    parse_dates: convert the text `date` column to datetime.
    """
    if with_names:
        sql = """
            SELECT m.*,
                   c.name            AS country_name,
                   l.name            AS league_name,
                   ht.team_long_name AS home_team_name,
                   at.team_long_name AS away_team_name
            FROM Match m
            LEFT JOIN Country c ON c.id = m.country_id
            LEFT JOIN League  l ON l.id = m.league_id
            LEFT JOIN Team   ht ON ht.team_api_id = m.home_team_api_id
            LEFT JOIN Team   at ON at.team_api_id = m.away_team_api_id
        """
    else:
        sql = "SELECT * FROM Match"
    matches = query(sql, db_path=db_path)
    if parse_dates:
        matches["date"] = pd.to_datetime(matches["date"])
    return matches


def load_teams(db_path=DB_PATH):
    """Load the Team table (team_api_id, team_long_name, team_short_name, ...)."""
    return query("SELECT * FROM Team", db_path=db_path)


def load_team_attributes(parse_dates=True, db_path=DB_PATH):
    """Load Team_Attributes (dated team style ratings; see decision D-012)."""
    attrs = query("SELECT * FROM Team_Attributes", db_path=db_path)
    if parse_dates:
        attrs["date"] = pd.to_datetime(attrs["date"])
    return attrs
