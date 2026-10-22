"""Shared look and data-derived helpers for dashboard pages."""

import ast
import re

import pandas as pd
import streamlit as st

from dashboard import data_access as da

OUTCOMES = ["Home Win", "Draw", "Away Win"]
OUTCOME_COLORS = {"Home Win": "#0072B2", "Draw": "#999999", "Away Win": "#E69F00"}


@st.cache_data
def read_table(name):
    """Read one of the small output tables without bypassing the data contract."""
    return pd.read_csv(da.TABLES / f"{name}.csv")


@st.cache_data
def league_names():
    """Return one display name for each league identifier."""
    return da.load_team_lookup()[["league_id", "league_name"]].drop_duplicates("league_id")


@st.cache_data
def readme_fingerprint():
    """Read the dataset fingerprint displayed on the overview page."""
    text = (da.REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rows = {}
    for key in ["File", "Size", "SHA-256", "Kaggle version date", "Contents"]:
        match = re.search(rf"^\|\s*{re.escape(key)}\s*\|\s*(.+?)\s*\|\s*$", text, flags=re.M)
        if match:
            rows[key] = match.group(1).strip("`")
    return rows


def model_note():
    """Describe the selected model using its exported training configuration."""
    summary = da.load_training_summary().set_index("model")
    config = ast.literal_eval(summary.loc[da.FEATURED_MODEL, "selected_config"])
    parts = [
        "balanced class weights" if key == "class_weight" and value == "balanced"
        else f"{key} = {value}"
        for key, value in config.items()
    ]
    return f"{da.MODEL_LABELS[da.FEATURED_MODEL]}, " + ", ".join(parts)


def style(fig, title, x_title=None, y_title=None, height=380):
    """Apply the dashboard's common Plotly layout."""
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"title": None, "orientation": "h", "y": -0.2},
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return fig


def show(fig):
    """Render a Plotly figure at full width."""
    st.plotly_chart(fig, width="stretch")


def caution(text):
    st.warning(text, icon=":material/info:")


def footer(sources, model_note=None):
    """Render a consistent source and model footer."""
    st.divider()
    parts = [f"Data: {', '.join(sources)}", "Test season: 2015/16"]
    if model_note:
        parts.append(model_note)
    st.caption(" · ".join(parts))
