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
| **Logistic Regression** (balanced) | 45.46% | **0.4342** | 28.1% |
| **Random Forest** (balanced) | 44.38% | **0.4341** | 36.5% |
| Decision Tree (balanced) | 42.81% | 0.4265 | 42.3% |
| KNN | 43.08% | 0.3893 | 18.5% |
| Majority baseline ("always Home Win") | 43.87% | 0.2033 | 0% |

Logistic Regression and Random Forest are effectively tied on macro F1: 0.434166 vs 0.434126 (difference ≈0.00004). This is much smaller than the reported ±0.003 cross-operating-system variation for Random Forest (PR #8 review).

**Final selection: Logistic Regression (balanced, C=1.0).** Since macro F1 does not meaningfully separate the top two, Logistic Regression is selected for its slightly higher accuracy (45.46% vs 44.38%) and more straightforward explanation through coefficient directions. Random Forest remains preferable if higher Draw recall is the main priority (36.5% vs 28.1%). The choice uses the model configurations selected on 2014/15; the 2015/16 test season was not used for further tuning.

## 2. Did the EDA predict the model behaviour?

| EDA finding (before modelling) | What it predicted | What the models showed |
|---|---|---|
| **EDA-07**: always predicting "Home Win" scores 45.9% accuracy but only 0.21 macro F1 | Accuracy would be misleading | ✅ The baseline gets 43.9% accuracy on the test season but 0.20 macro F1. The best models reach **0.43, more than double** |
| **EDA-08 / EDA-28**: draws are the minority class (25%) and the draw rate barely changes with form | Unweighted models would ignore draws | ✅ Without class weights, LR and RF predicted **zero** draws (PR #6). Balanced weights lift Draw recall to 28–42%, but Draw remains the weakest class |
| **EDA-09**: home-win share falls from 47.1% (2008/09) to 43.9% (2015/16) | The test season would have fewer home wins than training | ✅ Home wins: 46.0% in training vs **43.9% in the test season** |
| **EDA-25**: the home side wins 29.5% → 68.8% across form-difference bins | Matches with a clear form gap would be easier to predict | ✅ Accuracy is **~60% when the form gap is 7+ points**, vs ~40% for evenly matched teams |
| **EDA-26**: form vs goal difference correlation is only r = 0.27 | Macro F1 would stay modest | ✅ Best macro F1 is 0.434, a real improvement on the baseline but far from certainty |
| **EDA-28**: closeness of form barely changes the draw rate | Draw predictions would be unreliable | ✅ The models predict a draw in **41–55%** of evenly matched games, but only **28.5%** end level. In mismatches they predict 4–5% draws, but **22%** end level |

**Conclusion:** the EDA correctly anticipated the class-imbalance challenge, modest overall performance, and the value of form differences. The test-set subgroup analysis adds a useful detail: the models over-predict draws in close games and miss many draws in mismatches. This is an observed limitation, not something the EDA fully predicted.

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
3. **Treat draw predictions cautiously.** About 1 in 4 matches is a draw across form-gap groups (EDA-28). If staff use
   class probabilities rather than one predicted label, validate or calibrate those probabilities before setting operational
   thresholds; calibration was not evaluated in this project.
4. **Prioritise each team's record at the relevant venue.** The home team's earlier home-win rate and the away team's
   earlier away-win rate are the strongest measured signals: shuffling either lowers test macro F1 by about 0.02 in both
   models. Recent form still provides context, but it adds little unique signal once venue win rates are included. Because
   related features can share importance, this does not mean recent form is irrelevant (D-030; see
   `outputs/tables/permutation_importance.csv` and `outputs/figures/phase8_permutation_importance.png`).
5. **Do not assume a fixed home advantage.** It exists in every league but varies (+7.8 points in Scotland to +20.9 in
   Spain, EDA-23) and is shrinking over time (EDA-09). League-specific, up-to-date expectations are more accurate.
6. **Retrain each season.** Because the home-win share drifts, a model trained on older seasons gradually over-rates the
   home side. Refit before each new season, keeping the same chronological validation (D-021, D-028).
7. **Use Logistic Regression as the default decision-support model.** Its macro F1 is effectively tied with Random Forest,
   its accuracy is slightly higher, and its coefficient directions are easier to explain. If identifying more draws becomes
   the priority, compare Random Forest as an alternative because it has higher Draw recall (36.5% vs 28.1%).

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
| Random Forest results vary slightly across operating systems | About ±0.003 macro F1; the tiny LR/RF difference should not be treated as a robust ranking | PR #8 review |
| Predicted probabilities were not calibrated or evaluated | Probability-based thresholds may be unreliable | Not evaluated in this project |

## 6. AI-use declaration note

OpenAI Codex assisted with structuring the Phase 8 notebook, running the feature-importance analysis, and drafting the model-selection explanation and stakeholder-facing text. Reported metrics and interpretation outputs were generated from the project’s saved data and code. The student/team should review the final wording and outputs before submission.
