"""Page 5: what should an analyst do, and what can't the model tell us?"""

import plotly.express as px
import streamlit as st

from dashboard import data_access as da, theme

st.title("Insights and limitations")

importance = da.load_permutation_importance()
importance = importance[importance["model"] == da.FEATURED_MODEL].sort_values("macro_f1_drop_mean")
fig = px.bar(
    importance,
    x="macro_f1_drop_mean",
    y="feature",
    orientation="h",
    error_x="macro_f1_drop_std",
)
fig.update_traces(marker_color=theme.OUTCOME_COLORS["Home Win"])
top2 = (
    importance.sort_values("macro_f1_drop_mean", ascending=False)
    .head(2)["feature"]
    .tolist()
)
theme.show(theme.style(
    fig,
    f"Win rates matter most: {top2[0]} and {top2[1]} (D-030)",
    "Drop in macro F1 when the feature is shuffled",
    None,
    height=460,
))
st.caption(
    "A value near zero or below means shuffling that feature did not hurt the model. "
    "That says it adds little here, not that it is wrong."
)

coefficients = da.load_lr_coefficients()
fig = px.imshow(
    coefficients.round(2),
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
)
theme.show(theme.style(
    fig,
    "Logistic Regression coefficients by outcome (standardised features)",
    "Outcome",
    None,
    height=460,
))

st.subheader("Recommendations")
st.markdown(
    "1. **Watch each team's home and away win record first.** Shuffling it costs the model more than any form measure.\n"
    "2. **Treat a predicted Draw as a weak signal.** The model finds only about a quarter of real draws.\n"
    "3. **Use the probabilities, not only the label.** A 40% Home Win is far from a sure thing."
)

st.subheader("Limitations")
st.markdown(
    "- No injuries, suspensions, tactics or transfers: the data only holds past results.\n"
    "- Betting odds were excluded on purpose (D-011), so the model is not compared with bookmakers.\n"
    "- One test season (2015/16) and 11 leagues: results may differ in other seasons.\n"
    "- Draws are the hardest outcome to predict.\n"
    "- KNN has no class weighting in scikit-learn, so its weaker Draw results are partly a method limit (D-022)."
)

st.subheader("AI-use declaration")
st.write(
    "AI tools assisted with the dashboard plan and parts of the dashboard code and text. "
    "The group reviewed, tested and is responsible for all content. Update this wording to match "
    "the group's final declaration before submission."
)

theme.footer(["permutation_importance.csv", "logistic_regression_coefficients.csv"],
             theme.model_note())
