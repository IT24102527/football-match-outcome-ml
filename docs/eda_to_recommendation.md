# From EDA to Recommendation

Author: Ahamed M.A.U (IT24101779) · Week 6 · For: the Recommendation & Limitations section (Jassim M.N.M, Yusuf S.A)
Evidence: `logs/eda_insight_log.md` (EDA-xx), `notebooks/07_eda_to_recommendation.ipynb`, `outputs/tables/`
Test set: the full **2015/16 season (3,326 matches)**, never used for training or tuning (D-021, D-028)

This document links what the data showed **before** modelling (EDA) to what the models showed **after** (evaluation),
and turns both into recommendations for the stakeholder: a **club's analytics and coaching department preparing for
upcoming matches**.

---

## 1. Final results in one table

| Model | Accuracy | Macro F1 | Draw recall |
|---|---|---|---|
| **Logistic Regression** (balanced) | 45.5% | **0.434** | 28.1% |
| **Random Forest** (balanced) | 44.4% | **0.434** | 36.5% |
| Decision Tree (balanced) | 42.8% | 0.427 | 42.3% |
| KNN | 43.1% | 0.389 | 18.5% |
| Majority baseline ("always Home Win") | 43.9% | 0.203 | 0% |

Logistic Regression and Random Forest are **tied on macro F1** (difference < 0.001). Random Forest's score also varies by
about ±0.003 across operating systems (see the PR #8 review), so neither is a clear winner on performance alone.

## 2. Did the EDA predict the model behaviour?

| EDA finding (before modelling) | What it predicted | What the models showed |
|---|---|---|
| **EDA-07**: always predicting "Home Win" scores 45.9% accuracy but only 0.21 macro F1 | Accuracy would be misleading | ✅ The baseline gets 43.9% accuracy on the test season but 0.20 macro F1. The best models reach **0.43, more than double** |
| **EDA-08 / EDA-28**: draws are the minority class (25%) and the draw rate barely changes with form | Unweighted models would ignore draws | ✅ Without class weights, LR and RF predicted **zero** draws (PR #6). Balanced weights lift Draw recall to 28–42%, but Draw remains the weakest class |
| **EDA-09**: home-win share falls from 47.1% (2008/09) to 43.9% (2015/16) | The test season would have fewer home wins than training | ✅ Home wins: 46.0% in training vs **43.9% in the test season** |
| **EDA-25**: the home side wins 29.5% → 68.8% across form-difference bins | Matches with a clear form gap would be easier to predict | ✅ Accuracy is **~60% when the form gap is 7+ points**, vs ~40% for evenly matched teams |
| **EDA-26**: form vs goal difference correlation is only r = 0.27 | Macro F1 would stay modest | ✅ Best macro F1 is 0.434, a real improvement on the baseline but far from certainty |
| **EDA-28**: closeness of form barely changes the draw rate | Draw predictions would be unreliable | ✅ The models predict a draw in **41–55%** of evenly matched games, but only **28.5%** end level. In mismatches they predict 4–5% draws, but **22%** end level |

**Conclusion:** every major model behaviour was anticipated by the EDA. The results are **explainable from the data**,
not artefacts of a particular algorithm.

## 3. Where the model is useful, and where it is not

Accuracy by pre-match form gap (`outputs/tables/accuracy_by_form_gap.csv`):

| Form gap (points in last 5) | Share of matches | Logistic Regression | Random Forest | Baseline |
|---|---|---|---|---|
| 0–1 (even) | 24.8% | 41.9% | 39.3% | 40.6% |
| 2–3 | 29.7% | 40.2% | 38.2% | **46.1%** |
| 4–6 | 29.3% | 45.5% | 45.9% | 44.3% |
| **7+ (mismatch)** | 16.2% | **60.4%** | **60.7%** | 44.1% |

- **Clear form gap (4+ points, ~45% of matches):** the models beat the baseline, by **+16 points** in mismatches.
- **Close games (gap ≤ 3, ~55% of matches):** the models are **no more accurate than "the home team wins"**. Balanced
  weights make them predict draws and away wins in exactly these games. That improves Draw recall (and macro F1) but
  costs accuracy.
- Balanced weights also make the models **under-predict home wins** (LR 37.4% of predictions vs 43.9% actual).

## 4. Recommendations for the club's analytics and coaching staff

1. **Use the model as a pre-match form-gap indicator, not an oracle.** When a fixture shows a large form gap (7+ points
   over the last 5 matches), the predicted outcome is right about 6 times in 10. Use it to prioritise preparation, for
   example extra opposition analysis before matches the model flags as likely defeats.
2. **Treat close fixtures as open.** For evenly matched teams, the model does no better than assuming a home win. Pair
   it with scouting, tactical and squad information the dataset does not contain.
3. **Do not plan around predicted draws.** About 1 in 4 matches is a draw whatever the form (EDA-28). Where a draw
   matters, use the model's **probabilities** rather than its single predicted label.
4. **Watch recent form and venue record first.** Points from the last 5 matches, goals scored and conceded, and the
   team's record at that venue are the pre-match indicators with measurable signal (EDA-24, EDA-25, EDA-33).
   *To be confirmed with Yusuf's Phase 8 feature importance.*
5. **Do not assume a fixed home advantage.** It exists in every league but varies (+7.8 points in Scotland to +20.9 in
   Spain, EDA-23) and is shrinking over time (EDA-09). League-specific, up-to-date expectations are more accurate.
6. **Retrain each season.** Because the home-win share drifts, a model trained on older seasons gradually over-rates the
   home side. Refit before each new season, keeping the same chronological validation (D-021, D-028).
7. **Choosing between the two top models:** they are tied on performance, so choose on purpose. **Logistic Regression**
   is easier to explain to coaching staff (each feature has a direction and weight). **Random Forest** recognises more
   draws (36.5% vs 28.1% recall). Final choice: Yusuf (Phase 8), recorded in the decision log.

## 5. Limitations that bound these recommendations

| Limitation | Effect | Evidence |
|---|---|---|
| No injuries, suspensions, line-ups, tactics or transfers | The biggest pre-match factors are missing, which caps achievable accuracy | Dataset scope, D-015 |
| Betting odds deliberately excluded | Market information that likely beats form alone is not used | D-011 |
| Team_Attributes and rest days not used | Possible extra signal left out to keep features validated | D-026, D-027 |
| Belgium 2013/14 almost entirely missing | Belgian form early in 2014/15 rests on old matches | EDA-03 |
| Promoted teams start with no history | 544 cold-start matches use imputed (median) form | EDA-30, D-019 |
| Short-history win rates (1–4 matches) can be exactly 0 or 1 | Noisy inputs for about 2% of rows | EDA-39, D-025 |
| One test season only | Results reflect 2015/16; other seasons may differ by a few points | D-021 |
| Random Forest results vary slightly across operating systems | About ±0.003 macro F1; ranking unchanged | PR #8 review |

## 6. Still to add

- **Feature importance / coefficients (Yusuf, Phase 8):** confirm or revise recommendation 4.
- **Final model choice (Yusuf, Phase 8):** confirm recommendation 7.
