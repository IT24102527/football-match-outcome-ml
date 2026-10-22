"""Page 1: what this is, and can I trust it?"""

import plotly.express as px
import streamlit as st

from dashboard import data_access as da, theme

st.title("Football match outcome predictor")
st.write(
    "A club analyst's view of a model that predicts whether a European league match ends in a "
    "**Home Win, Draw or Away Win**, using only each team's form and home/away record before kickoff."
)

comparison = da.load_model_comparison().set_index("model")
split = da.load_class_distribution()
best = comparison["macro_f1"].idxmax()
baseline = "majority_baseline"
total = int(split.loc[split["split"] == "overall", "count"].sum())
test_n = int(split.loc[split["split"] == "test", "count"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches in the data", f"{total:,}")
c2.metric("Test matches (2015/16)", f"{test_n:,}")
c3.metric(f"Best model macro F1 ({best.replace('_', ' ')})", f"{comparison.loc[best, 'macro_f1']:.3f}")
c4.metric("Baseline macro F1 (always Home Win)", f"{comparison.loc[baseline, 'macro_f1']:.3f}")

theme.caution(
    f"The model is modest: accuracy is {comparison.loc[best, 'accuracy']:.1%} against "
    f"{comparison.loc[baseline, 'accuracy']:.1%} for always guessing Home Win. Its value is that it also "
    "predicts Draws and Away Wins, which the baseline never does. Draws remain hard to predict."
)

dist = split.copy()
dist["class"] = dist["class"].astype(str)
fig = px.bar(
    dist,
    x="split",
    y="percentage",
    color="class",
    barmode="group",
    text=dist["percentage"].round(1),
    category_orders={"class": theme.OUTCOMES, "split": ["overall", "train", "test"]},
    color_discrete_map=theme.OUTCOME_COLORS,
)
fig.update_traces(texttemplate="%{text}%", textposition="outside")
home_share = dist.loc[
    (dist["split"] == "test") & (dist["class"] == "Home Win"), "percentage"
].iloc[0]
theme.show(theme.style(fig, f"Home teams win {home_share:.0f}% of 2015/16 test matches",
                       "Data split", "Share of matches (%)"))

with st.expander("Dataset fingerprint"):
    for key, value in theme.readme_fingerprint().items():
        st.write(f"**{key}:** `{value}`")

st.markdown(
    "**Use the sidebar to move between pages:** data and home advantage, model comparison, "
    "a match predictor, and insights and limitations."
)
theme.footer(["model_comparison.csv", "class_distribution.csv", "README.md"], theme.model_note())
