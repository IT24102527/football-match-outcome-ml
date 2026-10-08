# Decision Log — Group AI-39

Every non-trivial project decision is recorded here **in the week it is made**, using a continuous ID
sequence.

- **D-001 to D-012** are the initial decisions from the Initial Submission (Week 1). The Initial
  Submission recorded the decision and its reason. The *Options Considered* column was added when the
  decisions were copied into this log.
- **D-013 onwards** are recorded as the project runs, with the date and owner.

Fields: Decision ID · Date · Decision Area · Decision Made · Options Considered · Reason Chosen · Owner

---

## Initial decisions (Initial Submission, Week 1)

| ID | Date | Decision Area | Decision Made | Options Considered | Reason Chosen | Owner |
|---|---|---|---|---|---|---|
| D-001 | Week 1 | Dataset | European Football Database (Kaggle, SQLite) | European Football Database; Football-Data.co.uk season CSVs | Contains a large amount of historical match, team, and player data | All members |
| D-002 | Week 1 | Primary Lens | Match Outcome Prediction | Match outcome analysis; team performance patterns; home advantage/form analysis | Provides a clear and measurable classification problem | All members |
| D-003 | Week 1 | Supporting Lens | Team Form & Home Advantage | No secondary lens; Team Form & Home Advantage; team performance patterns | Provides useful features for the primary prediction problem | All members |
| D-004 | Week 1 | Target | Home Win / Draw / Away Win | 3-class H/D/A; binary Home Win vs not; goal-difference regression | Natural representation of football match outcomes | All members |
| D-005 | Week 1 | Feature Engineering | Historical form and performance | Historical form features; in-match statistics; betting odds | Uses information that would realistically be known before kickoff | All members |
| D-006 | Week 1 | Baseline | Majority-class classifier | Majority-class; stratified random guess; plain Logistic Regression | Provides a simple benchmark for model comparison | All members |
| D-007 | Week 1 | Algorithms | Logistic Regression, Decision Tree, Random Forest, KNN | Linear, tree-based, ensemble, instance-based, boosting, neural network models | Provides several different classification approaches | All members |
| D-008 | Week 1 | Main Evaluation | Macro F1 plus supporting metrics | Accuracy only; weighted F1; macro F1 | Classes are not equally distributed | All members |
| D-009 | Week 1 | Validation | Chronological split | Random split; k-fold CV; chronological split | Better represents predicting future matches | All members |
| D-010 | Week 1 | Leakage Control | Remove current-match information | Keep all columns; remove current-match information | Prevents unrealistic model performance | All members |
| D-011 | Week 1 | Betting Odds | Exclude from the first main model | Use odds as features; exclude odds | Avoid relying heavily on external bookmaker intelligence | All members |
| D-012 | Week 1 | Team Attributes | Use only when historical date is valid | Ignore Team_Attributes; join latest snapshot regardless of date; join only snapshots dated on/before the match | Prevents future information from entering earlier matches | All members |

---

## Decisions during the project

| ID | Date | Decision Area | Decision Made | Options Considered | Reason Chosen | Owner |
|---|---|---|---|---|---|---|
| D-013 | 2026-10-08 | Dataset access & fingerprint | Keep `database.sqlite` out of Git. Members download it from Kaggle (or the group Drive mirror) into `data/raw/` and check it against the recorded SHA-256 (`4df8569777d59fdd690754b1cc8ca1f7989baf65f2eaddd0f1368285f11139a9`, 313,090,048 bytes). | Commit the file to Git; Git LFS; Kaggle download / Drive mirror plus a SHA-256 check | At 313 MB the file is over GitHub's 100 MB file limit. The hash proves every member and the marker run the notebooks on the identical data. | Ahamed M.A.U (IT24101779) |
| D-014 | 2026-10-08 | Environment | `requirements.txt` stored as UTF-8, with the Windows-only `pywinpty` limited by `sys_platform == "win32"` | Keep the Windows `pip freeze` as-is; hand-written minimal list; keep the pinned freeze but fix the platform-specific package | The original UTF-16 Windows freeze failed to install on macOS because of `pywinpty`. Keeping the exact pins keeps every member's library versions identical. | Ahamed M.A.U (IT24101779) |
