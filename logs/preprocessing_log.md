## Phase 5 — Train/test preparation

- The final chronological test season is `2015/2016`; all rows from `2009/2010`
  through `2014/2015` are training rows.
- Rows are sorted by `date` and `id` before splitting.
- Historical-feature medians are fitted on training rows only and then applied to
  both train and test rows.
- Model features are the ten historical numeric features plus `league_id`.
  `id`, `date`, `season`, and `match_outcome` remain traceability/target columns
  and are excluded from the feature matrix.
- No match ID may occur in both partitions, and the latest training date must
  precede the earliest test date.