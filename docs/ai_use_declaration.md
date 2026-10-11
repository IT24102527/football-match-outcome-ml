# AI-Use Declaration — Group AI-39

The module requires an honest declaration of any AI-tool use (Assignment Descriptor §5, "Reproducibility,
documentation, and AI-use transparency"). Each member writes **their own section, in their own words**: which tools,
what they were used for, and how the output was checked. Reported numbers come from the project's code and data, never
from an AI tool directly. The report's AI-use section is assembled from this file.

---

## Ahamed M.A.U (IT24101779): Data Understanding & EDA

**Tool:** Claude (Anthropic), used through Claude Code in VS Code, 2026-09-25 to 2026-10-11.

**What it was used for:**
- **Planning:** summarising the assignment descriptor, Initial Submission and technical guide into a task list for my
  role.
- **Code:** drafting `src/db.py` and the code in `notebooks/01_data_understanding.ipynb`, `02_eda.ipynb`,
  `04a_feature_validation.ipynb` and `07_eda_to_recommendation.ipynb`; fixing `requirements.txt` for macOS/Linux.
- **Write-ups:** first drafts of the notebook commentary, `docs/data_dictionary.md`, `logs/eda_insight_log.md`
  (EDA-01–EDA-39), decision log entries D-001–D-018 (D-001–D-012 transcribed from the Initial Submission),
  `docs/eda_handoff.md` and `docs/eda_to_recommendation.md`.
- **Review support:** re-running teammates' notebooks to check that their outputs reproduce, and drafting PR review
  comments and the Phase 8 brief for Jassim.

**How the output was checked:**
- Every number in my documents is produced by code in the notebooks. All notebooks were re-run from a clean state to
  confirm that the outputs match.
- AI-drafted statements were checked against the actual outputs, and several were corrected. Examples: the
  duplicate team names were first described as different clubs with similar names, but the data showed they were
  mislabelled; a leakage spot-check value and two SQLite column types were also wrong in the first draft.
- The engineered features were validated with an independent rebuild and leakage tests (EDA-34–EDA-37), not taken on
  trust.

**What I decided and did myself:**
- Chose the project set-up decisions (working branch, shared `src/db.py` helper, adding D-001–D-012 to the log) and
  directed each step of the work.
- Reviewed every part before committing it myself, merged teammates' PRs after review, and coordinated hand-offs with
  the group.

---

## Jassim M.N.M (IT24103137): Business & Project Understanding, Phase 8

## Jassim M.N.M (IT24103137): Business & Project Understanding, Phase 8

**Tool:** OpenAI , used through the Codex desktop app, 2026-10-10 to 2026-10-11.

**What it was used for:**
- **Project understanding:** explaining the assignment materials, project structure, my assigned work, and the GitHub pull-request review and merge process.
- **Phase 8:** helping structure `notebooks/08_model_selection.ipynb`, analyse feature importance, and draft the model-selection explanation and stakeholder recommendations.
- **Documentation:** drafting updates to `docs/eda_to_recommendation.md` and decision-log entries D-029 and D-030.

**How the output was checked:**
- The Phase 8 notebook’s six code cells were executed. The Logistic Regression and Random Forest predictions were compared with the project’s saved test predictions; both matched 100%.
- The reported metrics and feature-importance values came from the project’s data and executed code, not directly from the AI tool.
- I reviewed the generated documentation and analysis against the notebook outputs and project files before submission.

**What I decided and did myself:**
- I reviewed the model comparison and selected Logistic Regression for its effectively tied Macro F1 score, slightly higher accuracy, and easier-to-explain coefficients.
- I reviewed the recommendations and limitations and used GitHub to submit my work for team review.

---

## Umair M.W.M (IT24102527): Preprocessing & Feature Engineering

*[To be completed by Umair: tools used (or "none"), what for, and how the output was checked.]*

---

## Yusuf S.A (IT24103319): Modelling & Evaluation

*[To be completed by Yusuf: tools used (or "none"), what for, and how the output was checked.]*
