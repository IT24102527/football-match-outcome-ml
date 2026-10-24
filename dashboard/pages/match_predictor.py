"""Page 4 · Match predictor: what does the model say about a match?

Owner: Member A (data and model layer). Replay any 2015/16 test match, or set the inputs yourself in what-if mode.
Probabilities come from the selected model (Logistic Regression, D-029) through dashboard.model_io.
Run on its own with `streamlit run dashboard/pages/match_predictor.py`, or through dashboard/app.py.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import data_access as da  # noqa: E402
from dashboard.model_io import CLASS_ORDER, explain, predict_proba, typical_values  # noqa: E402

try:                                    # Member B's shared colours, once dashboard/theme.py exists
    from dashboard.theme import OUTCOME_COLORS  # noqa: E402
except ImportError:
    OUTCOME_COLORS = {"Home Win": "#2a78d6", "Draw": "#eb6834", "Away Win": "#1baf7a"}

FEATURE_LABELS = {
    "Home_Form_Last_5": "Home team: points in last 5 matches",
    "Away_Form_Last_5": "Away team: points in last 5 matches",
    "Home_Avg_Goals_Last_5": "Home team: goals scored per match (last 5)",
    "Away_Avg_Goals_Last_5": "Away team: goals scored per match (last 5)",
    "Home_Avg_Conceded_Last_5": "Home team: goals conceded per match (last 5)",
    "Away_Avg_Conceded_Last_5": "Away team: goals conceded per match (last 5)",
    "Home_Win_Rate": "Home team: win rate in earlier home matches",
    "Away_Win_Rate": "Away team: win rate in earlier away matches",
    "Form_Diff": "Form difference (home − away points)",
    "Abs_Form_Diff": "Closeness of form (absolute difference)",
}
# slider key -> (feature, min, max, step)
SLIDERS = {
    "wi_home_form": ("Home_Form_Last_5", 0, 15, 1), "wi_away_form": ("Away_Form_Last_5", 0, 15, 1),
    "wi_home_goals": ("Home_Avg_Goals_Last_5", 0.0, 6.0, 0.2), "wi_away_goals": ("Away_Avg_Goals_Last_5", 0.0, 6.0, 0.2),
    "wi_home_conc": ("Home_Avg_Conceded_Last_5", 0.0, 6.0, 0.2), "wi_away_conc": ("Away_Avg_Conceded_Last_5", 0.0, 6.0, 0.2),
    "wi_home_wr": ("Home_Win_Rate", 0.0, 1.0, 0.01), "wi_away_wr": ("Away_Win_Rate", 0.0, 1.0, 0.01),
}

bundle = da.get_model()
matches = da.load_test_matches()
teams = da.load_team_lookup()
leagues = teams.drop_duplicates("league_id").set_index("league_name")["league_id"].sort_index()
medians = bundle["imputation_values"]


# ---------------- helpers ----------------
def probability_bar(proba):
    """One horizontal bar split into the three outcome probabilities, labelled in text as well as colour."""
    fig = go.Figure()
    for outcome in CLASS_ORDER:
        p = float(proba[outcome])
        fig.add_bar(x=[p], y=[""], orientation="h", name=outcome, marker_color=OUTCOME_COLORS[outcome],
                    text=[f"{outcome} {p:.0%}"], textposition="inside", insidetextanchor="middle",
                    hovertemplate=f"{outcome}: {p:.1%}<extra></extra>")
    fig.update_layout(barmode="stack", height=110, showlegend=False, margin=dict(l=0, r=0, t=4, b=4),
                      xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"), yaxis=dict(visible=False))
    return fig


def confidence_note(proba):
    ranked = proba.sort_values(ascending=False)
    lead = ranked.iloc[0] - ranked.iloc[1]
    if lead < 0.10:
        return f"Close call: {ranked.index[0]} leads {ranked.index[1]} by only {lead:.0%}. Treat this match as open."
    return f"{ranked.index[0]} is the most likely outcome, {lead:.0%} ahead of {ranked.index[1]}."


def drivers_table(X, outcome):
    contrib = explain(bundle, X, outcome)
    typical = typical_values(bundle)
    return pd.DataFrame({
        "Input": [FEATURE_LABELS[f] for f in contrib.index],
        "This match": [X.iloc[0][f] for f in contrib.index],
        "Typical match": [typical[f] for f in contrib.index],
        "Effect": ["pushes towards " + outcome if v > 0 else "pushes away from " + outcome for v in contrib],
    })


def show_prediction(proba, X):
    predicted = proba.idxmax()
    st.plotly_chart(probability_bar(proba), width="stretch", config={"displayModeBar": False})
    st.markdown(f"**Most likely: {predicted}.** {confidence_note(proba)}")
    st.markdown("**What pushed this prediction most, compared with a typical match**")
    st.dataframe(drivers_table(X, predicted), hide_index=True, width="stretch",
                 column_config={"This match": st.column_config.NumberColumn(format="%.2f"),
                                "Typical match": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Each input is compared with its average over the training seasons. An input pushes the prediction "
               "only as far as it differs from that typical value.")


def open_in_what_if(row):
    """Copy a replayed match's real inputs onto the what-if sliders and switch mode."""
    for key, (feature, *_rest) in SLIDERS.items():
        value = row[feature]
        st.session_state[key] = int(round(value)) if feature.endswith("Form_Last_5") else float(value)
    st.session_state["wi_league"] = row["league_name"]
    st.session_state["p4_mode"] = "What if"


def reset_what_if():
    for key, (feature, *_rest) in SLIDERS.items():
        value = medians[feature]
        st.session_state[key] = int(round(value)) if feature.endswith("Form_Last_5") else round(float(value), 2)


# ---------------- page ----------------
st.title("Match predictor")
st.markdown(
    f"See what the selected model ({da.MODEL_LABELS[da.FEATURED_MODEL]}) expects before kickoff. It is right on "
    f"**{matches['correct'].mean():.1%}** of 2015/16 test matches, so read the probabilities, not just the top outcome."
)
mode = st.radio("Mode", ["Replay a 2015/16 match", "What if"], horizontal=True, key="p4_mode",
                label_visibility="collapsed")

if mode == "Replay a 2015/16 match":
    league = st.selectbox("League", sorted(matches["league_name"].unique()), key="rp_league")
    in_league = matches[matches["league_name"] == league]
    options = in_league.index.tolist()
    pick = st.selectbox("Match", options, key="rp_match",
                        format_func=lambda i: f"{in_league.at[i, 'date']:%d %b %Y} · {in_league.at[i, 'fixture']}")
    row = matches.loc[pick]
    X = pd.DataFrame([row[bundle["feature_columns"]]]).astype(float)
    proba = row[["p_home", "p_draw", "p_away"]].set_axis(CLASS_ORDER).astype(float)

    hit = row["correct"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted", row[f"pred_{da.FEATURED_MODEL}"])
    c2.metric("Actual result", row["actual"])
    c3.metric("Model was", "Right ✓" if hit else "Wrong ✗")
    show_prediction(proba, X)

    st.markdown("**The inputs the model saw** (all from matches before this one)")
    side = pd.DataFrame({
        "": ["Points in last 5", "Goals scored per match (last 5)", "Goals conceded per match (last 5)",
             "Win rate at this venue (earlier matches)"],
        row["home_team"] + " (home)": [row["Home_Form_Last_5"], row["Home_Avg_Goals_Last_5"],
                                       row["Home_Avg_Conceded_Last_5"], row["Home_Win_Rate"]],
        row["away_team"] + " (away)": [row["Away_Form_Last_5"], row["Away_Avg_Goals_Last_5"],
                                       row["Away_Avg_Conceded_Last_5"], row["Away_Win_Rate"]],
    })
    st.dataframe(side, hide_index=True, width="stretch")

    st.markdown("**What every model predicted for this match**")
    st.dataframe(pd.DataFrame({"Model": [da.MODEL_LABELS[m] for m in da.MODELS],
                               "Prediction": [row[f"pred_{m}"] for m in da.MODELS],
                               "Right?": ["✓" if row[f"pred_{m}"] == row["actual"] else "✗" for m in da.MODELS]}),
                 hide_index=True, width="stretch")
    st.button("Open this match in what-if mode", on_click=open_in_what_if, args=(row,), key="rp_to_wi")

else:
    if "wi_home_form" not in st.session_state:
        reset_what_if()
    st.caption("Set each team's recent record. Sliders start at the training-season medians, a typical match.")
    league = st.selectbox("League", leagues.index.tolist(), key="wi_league")
    left, right = st.columns(2)
    with left:
        st.markdown("**Home team**")
        for key in ["wi_home_form", "wi_home_goals", "wi_home_conc", "wi_home_wr"]:
            feature, lo, hi, step = SLIDERS[key]
            st.slider(FEATURE_LABELS[feature].replace("Home team: ", "").capitalize(), lo, hi, step=step, key=key)
    with right:
        st.markdown("**Away team**")
        for key in ["wi_away_form", "wi_away_goals", "wi_away_conc", "wi_away_wr"]:
            feature, lo, hi, step = SLIDERS[key]
            st.slider(FEATURE_LABELS[feature].replace("Away team: ", "").capitalize(), lo, hi, step=step, key=key)
    st.button("Reset to a typical match", on_click=reset_what_if, key="wi_reset")

    values = {feature: float(st.session_state[key]) for key, (feature, *_rest) in SLIDERS.items()}
    values["Form_Diff"] = values["Home_Form_Last_5"] - values["Away_Form_Last_5"]
    values["Abs_Form_Diff"] = abs(values["Form_Diff"])
    values["league_id"] = int(leagues[league])
    X = pd.DataFrame([values])[bundle["feature_columns"]]
    proba = predict_proba(bundle, X).iloc[0]
    show_prediction(proba, X)
    # Bands and accuracies come from notebook 07 (accuracy_by_form_gap.csv), so they cannot drift from the report
    gaps = da.load_accuracy_by_form_gap().set_index("Abs_Form_Diff")
    clear_band, close_bands = da.FORM_GAP_ORDER[-1], da.FORM_GAP_ORDER[:2]
    close = gaps.loc[close_bands]
    weighted = lambda col: (close[col] * close["matches"]).sum() / close["matches"].sum()
    st.caption(
        f"Form difference {values['Form_Diff']:+.0f} points. On the 2015/16 test season the model was right "
        f"{gaps.loc[clear_band, da.FEATURED_MODEL]:.0%} of the time in the {clear_band} form-gap band, but only "
        f"{weighted(da.FEATURED_MODEL):.0%} in the {close_bands[0]} and {close_bands[1]} bands, where always "
        f"picking the home team scored {weighted('majority_baseline'):.0%}.")

st.divider()
st.warning("Decision support, not a guarantee: the model has no injury, line-up or tactical data, draws are hard to "
           "predict, and its probabilities are not calibrated.", icon="⚠️")
st.caption(f"Source: outputs/dashboard/predictions.csv (official run, D-031) and data/processed/features.csv. "
           f"Model: {da.MODEL_LABELS[da.FEATURED_MODEL]} "
           f"({'saved model file' if bundle['source'] == 'saved' else 'refitted from features.csv'}, D-034). "
           f"Test season 2015/16.")
