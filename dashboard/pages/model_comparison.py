"""Page 3 · Model comparison: which model is best, and why macro F1?

Owner: Member A (data and model layer). Every number comes from dashboard.data_access; nothing is typed in.
Run on its own with `streamlit run dashboard/pages/model_comparison.py`, or through dashboard/app.py.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard import data_access as da  # noqa: E402
from dashboard.model_io import CLASS_ORDER  # noqa: E402

try:                                    # Member B's shared colours, once dashboard/theme.py exists
    from dashboard.theme import OUTCOME_COLORS  # noqa: E402
except ImportError:
    OUTCOME_COLORS = {"Home Win": "#2a78d6", "Draw": "#eb6834", "Away Win": "#1baf7a"}
FEATURED_COLOR = OUTCOME_COLORS["Home Win"]
OTHER_COLOR = "#9aa5b1"

comparison = da.load_model_comparison()
per_class = da.load_per_class_metrics()
training = da.load_training_summary().set_index("model")

featured = da.FEATURED_MODEL
best = comparison.iloc[0]
runner_up = comparison.iloc[1]
baseline = comparison.set_index("model").loc["majority_baseline"]
feat_row = comparison.set_index("model").loc[featured]

# ---------------- header ----------------
st.title("Model comparison")
trained = comparison[comparison["model"] != "majority_baseline"]
gap = best["macro_f1"] - runner_up["macro_f1"]
versus = (f"effectively tied with {runner_up['label']} ({runner_up['macro_f1']:.3f})" if gap < 0.005
          else f"ahead of {runner_up['label']} ({runner_up['macro_f1']:.3f})")
doubled = trained[trained["macro_f1"] > 2 * baseline["macro_f1"]]["label"].tolist()
if len(doubled) == len(trained):
    vs_baseline = "every trained model more than doubles"
elif doubled:
    vs_baseline = f"{', '.join(doubled[:-1]) + ' and ' + doubled[-1] if len(doubled) > 1 else doubled[0]} more than double"
else:
    vs_baseline = "every trained model beats"
st.markdown(
    f"**{best['label']} has the best macro F1 ({best['macro_f1']:.3f}), {versus}; {vs_baseline} the majority "
    f"baseline's {baseline['macro_f1']:.3f}.** All results are on the 2015/16 season, which no model saw during "
    f"training or tuning."
)

c1, c2, c3 = st.columns(3)
c1.metric(f"{da.MODEL_LABELS[featured]} macro F1", f"{feat_row['macro_f1']:.3f}",
          f"+{feat_row['macro_f1'] - baseline['macro_f1']:.3f} vs baseline")
c2.metric("Baseline accuracy", f"{baseline['accuracy']:.1%}", help="Always predicts Home Win")
c3.metric("Baseline macro F1", f"{baseline['macro_f1']:.3f}", help="Never predicts a draw or an away win")
st.info(
    f"**Why macro F1, not accuracy?** The baseline reaches {baseline['accuracy']:.1%} accuracy by always predicting "
    f"a home win, close to the trained models, yet it never predicts a draw or an away win. Macro F1 averages the F1 "
    f"score of the three outcomes equally, so a model only scores well if it handles all of them (D-008).",
    icon="ℹ️",
)

# ---------------- macro F1 chart ----------------
chart = comparison.sort_values("macro_f1")
chart["highlight"] = chart["model"].map(lambda m: da.MODEL_LABELS[m] + (" (selected)" if m == featured else ""))
fig = px.bar(
    chart, x="macro_f1", y="highlight", orientation="h", text=chart["macro_f1"].map("{:.3f}".format),
    color=chart["model"].eq(featured).map({True: "Selected model", False: "Other models"}),
    color_discrete_map={"Selected model": FEATURED_COLOR, "Other models": OTHER_COLOR},
    labels={"macro_f1": "Macro F1 on the 2015/16 test season (0–1)", "highlight": ""},
    title=f"{best['label']} leads on macro F1; the baseline scores {baseline['macro_f1']:.2f}",
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(xaxis_range=[0, max(0.5, chart["macro_f1"].max() * 1.15)], legend_title_text="",
                  height=340, margin=dict(l=10, r=30, t=60, b=40))
st.plotly_chart(fig, width="stretch")

# ---------------- full metrics table (sortable) ----------------
st.subheader("All metrics")
draw = per_class[per_class["class"] == "Draw"].set_index("model")
pc = per_class.pivot(index="model", columns="class", values="f1")
table = comparison.set_index("model")[["label", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]]
table = table.join(pc[CLASS_ORDER].add_suffix(" F1")).join(draw["recall"].rename("Draw recall"))
table = table.rename(columns={"label": "Model", "accuracy": "Accuracy", "macro_precision": "Macro precision",
                              "macro_recall": "Macro recall", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"})
st.dataframe(
    table.reset_index(drop=True),
    hide_index=True, width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c != "Model"}
    | {"Accuracy": st.column_config.NumberColumn(format="percent"),
       "Draw recall": st.column_config.NumberColumn(format="percent")},
)
st.caption("Click a column header to sort. Precision, recall and F1 range from 0 to 1. "
           "Draw is the hardest class for every model; KNN has no class-weighting option, so it finds the fewest draws.")

# ---------------- confusion matrix picker ----------------
st.subheader("Confusion matrix")
labels = [da.MODEL_LABELS[m] for m in da.MODELS]
choice = st.selectbox("Model", labels, index=labels.index(da.MODEL_LABELS[featured]), key="cm_model")
model = da.MODELS[labels.index(choice)]
view = st.radio("Show", ["Match counts", "Share of each actual outcome"], horizontal=True, key="cm_view")

cm = da.confusion_matrix(model)
shown = cm if view == "Match counts" else cm.div(cm.sum(axis=1), axis=0)
correct = int(cm.values.diagonal().sum())
total = int(cm.values.sum())
fig_cm = px.imshow(
    shown, text_auto=".0f" if view == "Match counts" else ".0%", color_continuous_scale="Blues", aspect="auto",
    labels=dict(x="Predicted outcome", y="Actual outcome", color="Matches" if view == "Match counts" else "Share"),
    title=f"{choice} gets {correct:,} of {total:,} test matches right ({correct / total:.1%})",
)
fig_cm.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=40))
st.plotly_chart(fig_cm, width="stretch")
st.caption("Rows are what actually happened; columns are what the model predicted; the diagonal is correct. "
           "In the share view each row adds up to 100%, so the diagonal is each outcome's recall.")

st.markdown(f"**Per-class results: {choice}**")
rows = per_class[per_class["model"] == model].set_index("class").loc[CLASS_ORDER, ["precision", "recall", "f1", "support"]]
st.dataframe(
    rows.rename(columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Test matches"}),
    width="stretch",
    column_config={"Precision": st.column_config.NumberColumn(format="%.3f"),
                   "Recall": st.column_config.NumberColumn(format="%.3f"),
                   "F1": st.column_config.NumberColumn(format="%.3f")},
)
if model == "majority_baseline":
    st.warning("The baseline predicts Home Win for every match, so Draw and Away Win have no correct predictions.")
elif model == "random_forest":
    st.warning("Random Forest results differ slightly between Windows and macOS even with fixed seeds (D-031). "
               "These numbers come from the single official run.")

# ---------------- footer ----------------
import ast
cfg = ast.literal_eval(training.loc[featured, "selected_config"])
cfg_text = ", ".join(("balanced class weights" if k == "class_weight" and v == "balanced" else f"{k} = {v}")
                     for k, v in cfg.items())
st.divider()
st.caption(
    f"Source: outputs/tables/model_comparison.csv, per_class_metrics.csv and outputs/dashboard/predictions.csv "
    f"(official run, D-031). Test season 2015/16, {int(training.loc[featured, 'test_rows']):,} matches; trained on "
    f"{int(training.loc[featured, 'training_rows']):,} matches from 2009/10–2014/15. "
    f"Selected model: {da.MODEL_LABELS[featured]} ({cfg_text}, D-029)."
)
