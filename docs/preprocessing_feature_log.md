# Preprocessing and Feature Engineering Log

## Input and preprocessing

Match data is loaded through `src/db.py`. Matches are sorted chronologically by
date and match ID before processing. The target `match_outcome` is derived as
`Home Win`, `Draw`, or `Away Win`.

The following columns are excluded from the modelling dataset:

- 8 in-match event XML columns
- 66 lineup/player X-Y columns
- 30 betting-odds columns

The 2008/2009 warm-up season is removed **after** historical features are
calculated across all seasons. This removes 3,326 matches, leaving 22,653.

## Historical features

The match-level dataset contains:

- `Home_Form_Last_5` and `Away_Form_Last_5`
- `Home_Avg_Goals_Last_5` and `Away_Avg_Goals_Last_5`
- `Home_Avg_Conceded_Last_5` and `Away_Avg_Conceded_Last_5`
- `Home_Win_Rate` and `Away_Win_Rate`
- `Form_Diff` and `Abs_Form_Diff`

Historical calculations use `shift(1)`, so they use only matches played before
the current match. Current-match goals are not used as predictors, and
current-match event/stat columns are excluded.

## Validation

- Rows: 22,653
- Columns: 17
- Duplicate match IDs: 0
- Outcomes: Home Win 10,351; Away Win 6,537; Draw 5,765

| Feature | Missing values |
|---|---:|
| `Home_Form_Last_5` | 276 |
| `Away_Form_Last_5` | 279 |
| `Home_Avg_Goals_Last_5` | 276 |
| `Away_Avg_Goals_Last_5` | 279 |
| `Home_Avg_Conceded_Last_5` | 276 |
| `Away_Avg_Conceded_Last_5` | 279 |
| `Home_Win_Rate` | 111 |
| `Away_Win_Rate` | 111 |
| `Form_Diff` | 544 |
| `Abs_Form_Diff` | 544 |

Imputation has **not** been performed yet. Any imputation must use
training-derived values only to avoid temporal leakage.
