"""Page 2: how often do home teams win, and where?"""

import plotly.express as px
import streamlit as st

from dashboard import data_access as da, theme

st.title("Data and home advantage")

features = da.load_features().merge(theme.league_names(), on="league_id", how="left")
features["home_win"] = (features["match_outcome"] == "Home Win").astype(int)

all_leagues = sorted(features["league_name"].dropna().unique())
all_seasons = sorted(features["season"].unique())
leagues = st.sidebar.multiselect("Leagues", all_leagues, default=all_leagues)
seasons = st.sidebar.multiselect("Seasons", all_seasons, default=all_seasons)
view = features[features["league_name"].isin(leagues) & features["season"].isin(seasons)]

if view.empty:
    st.info("No matches for this selection. Pick at least one league and one season.")
    st.stop()

st.metric("Matches in selection", f"{len(view):,}")

split = (
    view["match_outcome"].value_counts(normalize=True)
    .mul(100)
    .reindex(theme.OUTCOMES)
    .reset_index()
)
split.columns = ["outcome", "percent"]
fig = px.bar(
    split,
    x="outcome",
    y="percent",
    color="outcome",
    text=split["percent"].round(1),
    color_discrete_map=theme.OUTCOME_COLORS,
)
fig.update_traces(texttemplate="%{text}%", textposition="outside")
fig.update_layout(showlegend=False)
theme.show(theme.style(fig, f"Home teams win {split.loc[0, 'percent']:.0f}% of the selected matches",
                       "Outcome", "Share of matches (%)"))

col1, col2 = st.columns(2)
by_league = (
    features[features["season"].isin(seasons)]
    .groupby("league_name")["home_win"]
    .mean()
    .mul(100)
    .sort_values()
    .reset_index()
)
fig = px.bar(by_league, x="home_win", y="league_name", orientation="h",
             text=by_league["home_win"].round(1))
fig.update_traces(texttemplate="%{text}%", marker_color=theme.OUTCOME_COLORS["Home Win"])
top = by_league.iloc[-1]
with col1:
    theme.show(theme.style(fig, f"{top['league_name']} has the highest home-win rate ({top['home_win']:.0f}%)",
                           "Home-win rate (%)", None, height=460))
    st.caption("This comparison intentionally ignores the league filter; season selection keeps leagues comparable.")

by_season = view.groupby("season")["home_win"].mean().mul(100).reset_index()
fig = px.line(by_season, x="season", y="home_win", markers=True)
fig.update_traces(line_color=theme.OUTCOME_COLORS["Home Win"])
with col2:
    theme.show(theme.style(fig, "Home-win rate by season in the selection",
                           "Season", "Home-win rate (%)", height=460))

goals = (
    view.groupby("league_name")[["Home_Avg_Goals_Last_5", "Away_Avg_Goals_Last_5"]]
    .mean()
    .reset_index()
    .melt("league_name", var_name="side", value_name="goals")
)
goals["side"] = goals["side"].map({
    "Home_Avg_Goals_Last_5": "Home team",
    "Away_Avg_Goals_Last_5": "Away team",
})
fig = px.bar(
    goals,
    x="league_name",
    y="goals",
    color="side",
    barmode="group",
    color_discrete_map={
        "Home team": theme.OUTCOME_COLORS["Home Win"],
        "Away team": theme.OUTCOME_COLORS["Away Win"],
    },
)
st.caption("Averages of each team's goals in its previous five matches, not the goals of the match itself.")
theme.show(theme.style(fig, "Home teams enter matches scoring slightly more than away teams",
                       "League", "Average goals in previous 5 matches"))

st.subheader("Does the form gap change the result? (2015/16 test season)")
draws = theme.read_table("draws_by_form_gap")
accuracy = theme.read_table("accuracy_by_form_gap")
c1, c2 = st.columns(2)
fig = px.bar(draws, x="Abs_Form_Diff", y="actual draw rate",
             text=(draws["actual draw rate"] * 100).round(0))
fig.update_traces(texttemplate="%{text}%", marker_color=theme.OUTCOME_COLORS["Draw"])
with c1:
    theme.show(theme.style(fig, "Draws are rarer when the teams' form is far apart",
                           "Form gap (points)", "Actual draw rate"))
fig = px.bar(accuracy, x="Abs_Form_Diff", y="logistic_regression",
             text=(accuracy["logistic_regression"] * 100).round(0))
fig.update_traces(texttemplate="%{text}%", marker_color=theme.OUTCOME_COLORS["Home Win"])
with c2:
    theme.show(theme.style(fig, "Model accuracy is highest in mismatched games",
                           "Form gap (points)", "Logistic Regression accuracy"))

theme.footer(["features.csv", "team_lookup.csv", "draws_by_form_gap.csv",
              "accuracy_by_form_gap.csv"], theme.model_note())
