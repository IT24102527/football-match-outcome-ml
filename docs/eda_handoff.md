# EDA Handoff — Phase 1 → Phases 2–8

From: Ahamed M.A.U (IT24101779), Data Understanding & EDA
To: Umair M.W.M (preprocessing & features), Yusuf S.A (modelling & evaluation), Jassim M.N.M (business narrative)
Date: 2026-10-09

This brief tells each member what Phase 1 found that **changes their work**. The evidence for every point is in
`logs/eda_insight_log.md` (EDA-xx IDs), `notebooks/01_data_understanding.ipynb`, `notebooks/02_eda.ipynb`, and
`docs/data_dictionary.md`. Decisions already settled by EDA are logged as **D-015 to D-018**. **New decision log
entries should start at D-019.**

---

## 1. For Umair: preprocessing and feature engineering

### 1.1 Already settled (just implement)

| What | Rule | Source |
|---|---|---|
| Load data | Use `src/db.py` (`load_matches()`, `load_team_attributes()`), not ad-hoc `sqlite3` | D-013, README |
| Tables | Match, Team, Team_Attributes, League, Country only | D-015 |
| Columns to drop | 8 event XML columns, 66 line-up / X-Y columns, 30 odds columns. The 11 core columns remain. | D-016 |
| Grouping key | `team_api_id` only, never names (two Polish clubs share names) | D-017 |
| Rows to remove | **None.** No duplicates; keep extreme scores; never de-duplicate on (season, home, away) | D-018 |
| Raw missing values | After the drops above, the modelling table has **no missing raw values** | EDA-11 |
| Ordering | Sort by `date`, then `id`. No team plays twice on one day, so this ordering is unambiguous. | EDA-15 |

**Code note:** the guide's §5.2 target snippet `np.select([...], ['Home Win', 'Draw', 'Away Win'])` **crashes on
numpy 2.5** (our pinned version). Add `default="Unknown"` and assert that no row is `"Unknown"`. See `02_eda.ipynb` §2.

### 1.2 Decisions you need to make and log (EDA evidence + recommendation)

| # | Decision | Evidence | Options | EDA recommendation |
|---|---|---|---|---|
| 1 | **Cold-start rows** (team has < 5 earlier matches) | 1,018 matches (3.9%), 474 in 2008/09; 111 teams are promoted mid-dataset (EDA-30) | (a) Drop 2008/09 as a warm-up season and impute the rest; (b) neutral imputation + `has_history` flag; (c) smaller `min_periods` | **(a)**: 2008/09 has no history and no Team_Attributes for anyone, so it adds the least clean training signal. Impute the remaining ~544 rows. |
| 2 | **Form window across seasons** | Season reset leaves 14.3% of matches without form vs 3.9% rolling across seasons (EDA-31) | Reset each season; roll across seasons | **Roll across seasons** |
| 3 | **Team_Attributes** | Valid for both teams in only 74.5% of matches, 0% in 2008/09; `buildUpPlayDribbling` 66.5% missing (EDA-19–21) | Skip; core feature; optional extra group | **Optional extra group** via `merge_asof(direction="backward")`. Drop `buildUpPlayDribbling` and all `…Class` copies. |
| 4 | **Rest days** (if used) | Median 7 days; gaps up to 1,877 days (EDA-32) | Raw; capped; break flag | **Cap at 30 days**, or add a "first match after a break" flag |
| 5 | **League as a feature** | Home-win rate 41.7%–48.8% by league (EDA-10, EDA-23) | Ignore; one-hot `league_id` (11 values) | **One-hot `league_id`**: only 11 values, so no high-cardinality problem |

### 1.3 Feature list (agreed names, guide §4.1)

All features use **strictly earlier matches only**: `groupby("team_api_id")` → `shift(1)` → `rolling` / `expanding`.

| Feature | Definition | EDA evidence |
|---|---|---|
| `Home_Form_Last_5` / `Away_Form_Last_5` | Points (3/1/0) from the team's last 5 matches, any venue | Home win 29.5% → 68.8% across form-difference bins (EDA-25) |
| `Home_Avg_Goals_Last_5` / `Away_Avg_Goals_Last_5` | Mean goals scored in the last 5 matches | r = 0.72 with form: related but distinct (EDA-33) |
| `Home_Win_Rate` | Home team's win rate **in its earlier home matches** (expanding) | 97% of teams win more at home (EDA-24) |
| `Away_Win_Rate` | Away team's win rate **in its earlier away matches** (expanding) | Typical team 27% away vs 43% home (EDA-24) |
| *Suggested:* `Home_Avg_Conceded_Last_5` / `Away_Avg_Conceded_Last_5` | Mean goals conceded in the last 5 matches | r = −0.65 with form; carries the defensive side (EDA-33) |
| *Suggested:* `Form_Diff` and `Abs_Form_Diff` | Home form − away form, and its absolute value | `Abs_Form_Diff` is the only pre-match draw signal found (EDA-28) |

`02_eda.ipynb` §9–13 contains a working **rough** version of the long format and these features (`team_long()`,
`to_match_level()`). Reuse them as a starting point, and do the full §5.4 spot-check on the final versions.

---

## 2. For Yusuf: split, models and evaluation

| Point | Evidence | Implication |
|---|---|---|
| Majority baseline = 45.9% accuracy but **0.21 macro F1** (full data) | EDA-07 | Report macro F1 first. Recompute the baseline on the **test set**. |
| Home-win rate falls 47.1% → 43.9% over the seasons | EDA-09 | Test seasons have fewer home wins. Report train vs test class balance (guide §5.5). |
| Draw rate barely varies with form (22–28%) | EDA-08, EDA-28 | Expect the lowest recall on Draw. Try `class_weight="balanced"` and report Draw recall explicitly. |
| Form vs goal difference r = 0.27 | EDA-26 | Football is noisy: frame a solid gain over the baseline as success, not a high accuracy. |
| Same-team features correlate up to \|r\| = 0.72 | EDA-33 | LR coefficients are hard to read one by one. Prefer RF importances for "what matters most". |
| Seasons run Jul → May; 8 seasons of ~3,000–3,300 matches each | EDA-02 | Season-based split options: test on **2015/16 only** (3,326 matches, 12.8%), or on **2014/15–2015/16** (6,651, 25.6%; 29.4% if 2008/09 is dropped). The guide's 15–20% target falls between them. Pick one and log the cut-off and the reason. |
| Belgium 2013/14 is nearly empty | EDA-03 | Mention as a limitation; Belgian form early in 2014/15 rests on old matches. |

## 3. For Jassim: points for the business narrative

- **Home advantage is real and universal, but not equal.** Every league shows it, from +7.8 points (Scotland) to
  +20.9 points (Spain). It is shrinking over time (home wins 47.1% → 43.9%). (EDA-09, EDA-23)
- **Recent form matters.** A home side in much better form wins about 69% of the time, vs about 30% when in much
  worse form. Recent form is the most actionable pre-match indicator for a coaching staff. (EDA-25)
- **Draws are inherently hard to call.** No pre-match indicator separates them well. That is an honest limitation to
  state, not a modelling failure. (EDA-28)
- **Data limitations to state:** no injuries, squads or tactics; betting odds and in-match data excluded by design;
  Team_Attributes only partly time-aligned; Belgium 2013/14 largely missing; promoted teams have no prior history.
