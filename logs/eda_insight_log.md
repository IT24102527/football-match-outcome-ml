# EDA Insight Log — Group AI-39

Owner: Ahamed M.A.U (IT24101779) · Phase 1: Data Understanding & EDA
Sources: `notebooks/01_data_understanding.ipynb` (NB01), `notebooks/02_eda.ipynb` (NB02), figures in `outputs/figures/`

One entry per non-trivial finding. Every entry states **why it matters for modelling** and what action follows.
Numbers are computed from the full database (25,979 matches); re-running the notebooks reproduces them.

Owners: **Umair** (preprocessing & features) · **Yusuf** (modelling & evaluation) · **Jassim** (business narrative) · **All**

---

## 1. Dataset structure and coverage

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-01 | 2026-10-08 | One `Match` row = one league fixture. `id` and `match_api_id` are unique, and no team plays itself. | NB01 §4.1 | Defines the unit of analysis. Team-level history needs a long format with 2 rows per match, one per team. | Build features in team-match long format (guide §5.3). **Umair** |
| EDA-02 | 2026-10-08 | 25,979 matches · 11 leagues · 8 seasons (2008/09–2015/16) · 299 teams. Seasons run July → May/June. | NB01 §5, `eda_01` | The season boundaries suit a chronological split by season (D-009). | Split by season; record the cut-off as a decision. **Yusuf** |
| EDA-03 | 2026-10-08 | **Belgium 2013/14 has only 12 of ~240 matches.** Italy 2011/12 is 22 short, Switzerland 2011/12 18 short. | NB01 §5, `eda_01` | Belgian teams have a near-full-season hole in their history, so rolling form early in 2014/15 reaches back over a year. | Note as a limitation. Optionally flag Belgian matches early in 2014/15. **Umair** |
| EDA-04 | 2026-10-08 | Scotland and Switzerland teams meet 3–4 times a season (228 / 180 matches). Repeated (season, home, away) fixtures are real matches. | NB01 §5, NB02 §4 | De-duplicating on (season, home, away) would wrongly delete 1,470 real matches. | **Never de-duplicate on that key.** **Umair** |
| EDA-05 | 2026-10-08 | Player and Player_Attributes tables are player-level (184k rating rows). | NB01 §2 | Outside the team-level lens (D-003); using them would need line-up reconstruction and time alignment. | Out of scope. **All** |

## 2. Target variable

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-06 | 2026-10-08 | Class split reproduces the Initial Submission **exactly**: Home Win 11,917 (45.87%), Draw 6,596 (25.39%), Away Win 7,466 (28.74%). | NB02 §2 (assert), `eda_02` | Confirms the target rule (D-004) and that everyone uses the same dataset version (D-013). | None. |
| EDA-07 | 2026-10-08 | Majority-class baseline ("always Home Win") = **45.9% accuracy but 0.21 macro F1**. | NB02 §2.1 | Accuracy alone would make a useless model look decent. Direct evidence for macro F1 as the main metric (D-008). | Report macro F1 first; recompute the baseline on the test set. **Yusuf** |
| EDA-08 | 2026-10-08 | **Draw is the minority class (25.4%).** | NB02 §2, `eda_02` | Unweighted models will under-predict draws. | Try `class_weight="balanced"`; report Draw recall explicitly. **Yusuf** |
| EDA-09 | 2026-10-08 | Home-win rate **falls over time**: 47.1% (2008/09) → 43.9% (2015/16). Away wins rise 27.9% → 30.4%. | NB02 §2.2, `eda_04` | With a chronological split, the test seasons have fewer home wins than training. The test baseline will be below 45.9%, and a model leaning on "home wins" will drop slightly. | Report train vs test class balance (guide §5.5). **Yusuf** |
| EDA-10 | 2026-10-08 | Outcome mix differs by league: home win 41.7% (Scotland) to 48.8% (Spain); draws 23.2% (Spain) to 28.3% (France). | NB02 §2.2, `eda_03` | The differences are moderate, so one model across leagues is reasonable. League is a candidate context feature. | Consider `league_id` as a categorical feature. **Umair** |

## 3. Data quality: missing values, duplicates, validity

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-11 | 2026-10-08 | The 11 core columns (IDs, season, stage, date, goals) are **100% complete**. 104 of 115 columns have some missing values. | NB02 §3 | Target and all history features can be computed for every match. **No raw-value imputation needed** once excluded groups are dropped. | Drop the excluded groups; impute only engineered features. **Umair** |
| EDA-12 | 2026-10-08 | Missingness is **structural**: Bet365-type odds are **100% missing for Poland and Switzerland**; Pinnacle odds are 100% missing before 2012/13. | NB02 §3, `eda_05` | Imputing would fabricate odds for two entire leagues. Confirms the D-011 exclusion. | Exclude odds from features. **Umair** |
| EDA-13 | 2026-10-08 | The 8 in-match event columns (`goal`, `shoton`, `card`, `possession`, …) are XML, **45.3% missing**, and complete only for England, Germany, Italy and Spain. | NB01 §4.2, NB02 §3 | Known only after the match, so **leakage** (D-010). Using them would also silently limit the model to 4 leagues. | Drop all 8. **Umair** (leakage sign-off) |
| EDA-14 | 2026-10-08 | The 66 line-up / X-Y position columns are 5–7% missing (26% in 2008/09; Poland 62% for positions). | NB01 §4.2, NB02 §3 | Player-level, outside the team lens. | Drop. **Umair** |
| EDA-15 | 2026-10-08 | **No duplicates**: no identical rows, no repeated `match_api_id`, no fixture twice on one date. **No team plays twice on the same day.** | NB02 §4 | The date has no kickoff time, but ordering by date alone is enough to define "strictly earlier matches" without ambiguity. | Sort by `date` (then `id`) before every rolling feature. **Umair** |
| EDA-16 | 2026-10-08 | Validity checks all pass: no null or negative goals, no self-matches, all dates inside their season, all team IDs resolve. | NB02 §5 | No rows need removal. | None. |
| EDA-17 | 2026-10-08 | Extreme scores (PSV 10–0 Feyenoord, Real Madrid 10–2 Rayo, Deportivo 2–8 Real Madrid) are **genuine results**. | NB02 §5 | Removing real thrashings would distort rolling goal averages. Their effect on a 5-match window is temporary. | **Keep**; no outlier removal on goals. **Umair** |
| EDA-18 | 2026-10-08 | **Team names are unreliable**: "Polonia Bytom" and "Widzew Łódź" are each used for two different clubs (the pairs played each other). "Royal Excel Mouscron" covers two non-overlapping IDs. | NB01 §6 | Grouping by name would merge two clubs' histories and corrupt their form features. | **Always group by `team_api_id`.** **All** |

## 4. Team_Attributes alignment (D-012)

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-19 | 2026-10-08 | Six snapshot dates (Feb 2010/11/12, Sep 2013/14/15). The first is **2010-02-22**, so 5,469 matches (21%) come before any snapshot. | NB01 §7 | Under D-012, no attributes exist for all of 2008/09 and half of 2009/10. | Use `merge_asof(direction="backward")`. **Umair** |
| EDA-20 | 2026-10-08 | Only **74.5%** of matches have a valid prior snapshot for **both** teams (0% in 2008/09, 34% in 2009/10, ~94% after). 11 teams have no attributes at all. | NB01 §7 | Team_Attributes cannot be a core feature group. | **Optional** extra features only, with a missing indicator. Core features come from match history. **Umair** |
| EDA-21 | 2026-10-08 | `buildUpPlayDribbling` is **66.5%** missing. Every `…Class` column is a binned copy of its numeric rating. | NB01 §7, data dictionary §3 | The dribbling column is unusable, and class/number pairs are redundant. | Drop `buildUpPlayDribbling`; use numeric ratings, not their `Class` copies. **Umair** |

## 5. Home advantage and team form (supporting lens)

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-22 | 2026-10-09 | Home teams score **1.54** goals per match, away teams **1.16** (+0.38). Away teams fail to score in 33% of matches; home teams in 23%. | NB02 §7, `eda_06` | Venue shifts the goal distribution, so goal features should be read together with venue. | Keep home and away features separate. **Umair** |
| EDA-23 | 2026-10-09 | Home advantage exists in **every league**, from **+7.8 pts (Scotland)** to **+20.9 pts (Spain)** (home-win % minus away-win %). | NB02 §8, `eda_07` | Home advantage is real but not uniform. A single global home bonus would mis-fit leagues. | Team-level venue win rates capture it. Use in the narrative. **Umair, Jassim** |
| EDA-24 | 2026-10-09 | **97%** of teams (≥ 60 matches) win more at home; typical team 43% home vs 27% away wins. A team's home and away win rates correlate at **r = 0.83**. | NB02 §9, `eda_08` | Supports `Home_Win_Rate` and `Away_Win_Rate`, which encode team strength and the venue effect. | Build them as **expanding means over strictly earlier matches at that venue**. **Umair** |
| EDA-25 | 2026-10-09 | **Form carries a strong, monotonic signal.** When the home team's last-5 points exceed the away team's by 7+, home wins **68.8%**. When 6+ below, home wins **29.5%** and away wins 45.2%. | NB02 §10, `eda_09` | The strongest evidence for `Home_Form_Last_5` / `Away_Form_Last_5`. A form-difference feature may help linear models. | Build both; consider `Form_Diff`. **Umair** |
| EDA-26 | 2026-10-09 | Form difference vs final goal difference: **r = 0.27** (weak). | NB02 §10 | Pre-match form explains only part of the outcome. Expect modest macro F1; a solid gain over the baseline is a success. | Set expectations in the evaluation and limitations. **Yusuf, Jassim** |
| EDA-27 | 2026-10-09 | Leak-free spot-check: Chelsea's `form5` for match 6 = 3+3+1+3+1 = 11, the points of matches 1–5 only. | NB02 §10 | Confirms that `shift(1)` before `rolling(5)` excludes the current match. | Repeat on 3–5 matches for the final features (guide §5.4). **Umair** |

## 6. Draws, cold start, rest days and feature redundancy

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-28 | 2026-10-09 | The draw rate barely changes with form: **27.5%** when form is within 1 point vs **22.3%** for big mismatches. 1-1 is 45.7% of draws, 0-0 30.0%. | NB02 §11 | No pre-match signal clearly separates draws, so Draw will have the lowest recall. Absolute form difference is the best "closeness" signal. | `class_weight="balanced"`; report Draw recall; consider `abs(Form_Diff)`. **Yusuf, Umair** |
| EDA-29 | 2026-10-09 | Teams play ~34–38 matches a season (median 36). | NB02 §12 | A 5-match window is ~13% of a season: recent, but not a single-game reaction. Justifies the `_Last_5` window. | Keep window = 5. **Umair** |
| EDA-30 | 2026-10-09 | **Cold start:** 1,018 matches (3.9%) have a team with < 5 earlier matches, 474 of them in 2008/09. 111 of 299 teams first appear after 2008/09 (promoted; their lower-division history is not in the data). | NB02 §12 | Last-5 features are `NaN` for these rows. The handling must be decided and logged. | Choose: drop 2008/09 as a warm-up season / neutral imputation + `has_history` flag / smaller `min_periods`. Log it. **Umair** |
| EDA-31 | 2026-10-09 | Resetting form each season leaves **14.3%** of matches without form; rolling across seasons only **3.9%**. | NB02 §12 | A season reset would throw away ~10% more training signal. | **Roll across seasons.** **Umair** |
| EDA-32 | 2026-10-09 | Rest days: median 7, but 2.5% of gaps are > 60 days (summer breaks), up to 1,877 days (relegated teams returning). | NB02 §12 | A raw rest-days feature would be dominated by extreme gaps. | If used: cap (e.g. 30 days) or use a "first match after a break" flag. **Umair** |
| EDA-33 | 2026-10-09 | Same-team candidate features correlate moderately: form vs goals scored **r = 0.72**, form vs conceded **−0.65**, form vs venue win rate **0.51**. Home-team vs away-team features: **\|r\| ≤ 0.06**. | NB02 §13, `eda_10` | No redundant feature; home and away sides add independent information. Moderate collinearity makes LR coefficients harder to read one by one. | Keep all six planned features and add **goals conceded**. Use RF importance for interpretation. **Umair, Yusuf** |

## 7. Validation of the engineered features (Week 4)

Source: `notebooks/04a_feature_validation.ipynb` (NB04a), checking `data/processed/features.csv` from Phase 3.

| ID | Date | Finding | Evidence | Why it matters for modelling | Action / owner |
|---|---|---|---|---|---|
| EDA-34 | 2026-10-09 | An **independent rebuild** of all 8 base features (cumulative sums, separate code) **matches `features.csv` exactly** on all 22,653 matches (max difference ≈ 1e-16). | NB04a §2–3 | Two different algorithms agree, so the features are computed exactly as defined. | None. Features approved for modelling. **Yusuf** |
| EDA-35 | 2026-10-09 | All imputed cold-start values (276–279 rows per form/goal feature, 111 per win rate) equal the **training-season medians**. Including the test season would change the win-rate medians. | NB04a §3 | The imputation used no test-season information (D-019, D-022). | None. **Umair, Yusuf** |
| EDA-36 | 2026-10-09 | SQL spot-checks of 3 random matches from 3 leagues: form, goals scored, goals conceded and home win rate recomputed by hand from raw rows all match. One check shows the window correctly rolling across the summer break. | NB04a §4 | A human-readable trace of the feature definitions (guide §5.4). | Can be cited in the report's leakage section. **Umair** |
| EDA-37 | 2026-10-09 | **Truncation test:** deleting every match from 2013-01-01 onwards leaves the features of all 14,693 earlier matches unchanged. **Perturbation test:** changing a match's score leaves its own features unchanged and changes the team's next match. | NB04a §5 | Direct proof that the feature code uses **no future matches** and **not the match's own result**. | Cite as the leakage evidence (D-010, D-020). **Umair, Yusuf** |
| EDA-38 | 2026-10-09 | The final features reproduce the EDA form signal: home win **29.8% → 68.8%** across form-difference bins. | NB04a §6 | Imputation and the 2008/09 drop did not weaken the main signal. | None. |
| EDA-39 | 2026-10-09 | 444 rows per side (2.0%) have a win rate from only **1–4 earlier venue matches**, mostly promoted teams. More than half are exactly 0 or 1 (248 home, 295 away). | NB04a §7 | A rate of 0/2 looks as certain as 0/20. This is noise for LR and KNN; trees cope better. | Decide and log: keep as is, or shrink towards the league average. **Umair** |
