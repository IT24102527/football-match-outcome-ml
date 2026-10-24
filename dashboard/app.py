"""Entry point: run with `streamlit run dashboard/app.py` from the repository root."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Football match outcome predictor", page_icon=":soccer:", layout="wide")

PAGES = [
    ("pages/1_Overview.py", "Overview", True),
    ("pages/2_Data_and_Home_Advantage.py", "Data and home advantage", False),
    ("pages/model_comparison.py", "Model comparison", False),
    ("pages/match_predictor.py", "Match predictor", False),
    ("pages/5_Insights_and_Limits.py", "Insights and limits", False),
]

pages = [
    st.Page(path, title=title, default=default)
    for path, title, default in PAGES
    if (HERE / path).exists()
]
st.navigation(pages).run()
