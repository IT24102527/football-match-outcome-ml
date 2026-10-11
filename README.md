# Football Match Outcome Prediction

IT3091 Machine Learning, Group AI-39 (SLIIT, Y3.S1.WE.AI.01.02)

Predicts whether a European league football match ends in a **Home Win, Draw or Away Win**. The
predictors are historical team form and home/away performance, computed only from matches played
before kickoff.

| Item | Details |
|---|---|
| Track | Guided Data Track: Sports Analytics |
| Primary lens | Match Outcome Prediction (multiclass classification) |
| Supporting lens | Team Form & Home Advantage (engineered as features) |
| Models | Majority-class baseline, Logistic Regression, Decision Tree, Random Forest, KNN |
| Primary metric | Macro F1 |

## Team

| Student ID | Name | Role |
|---|---|---|
| IT24103137 | Jassim M.N.M | Business & Project Understanding |
| IT24101779 | Ahamed M.A.U | Data Understanding & EDA |
| IT24102527 | Umair M.W.M | Preprocessing & Feature Engineering |
| IT24103319 | Yusuf S.A | Modelling & Evaluation |

## Dataset

**European Soccer Database** by Hugo Mathien, Kaggle:
https://www.kaggle.com/datasets/hugomathien/soccer

The dataset is a single SQLite file (`database.sqlite`, about 300 MB). It is **not committed** to this
repository; it is listed in `.gitignore`.

| Fingerprint | Value |
|---|---|
| File | `database.sqlite` |
| Size | 313,090,048 bytes |
| SHA-256 | `4df8569777d59fdd690754b1cc8ca1f7989baf65f2eaddd0f1368285f11139a9` |
| Kaggle version date | 2017-04-15 |
| Downloaded | 2026-10-08 |
| Contents | 25,979 matches, 11 leagues, 8 seasons (2008/09–2015/16), 299 teams |

### Getting the data

Option A: Kaggle CLI (needs a Kaggle API token in `~/.kaggle/`)

```bash
pip install kaggle
kaggle datasets download hugomathien/soccer -p data/raw --unzip
```

Option B: Google Drive mirror. Download `database.sqlite` from the group's shared Drive folder:
**[Google Drive link: to be added]**

Either way, place the file at `data/raw/database.sqlite` and check that it is the same file:

```bash
shasum -a 256 data/raw/database.sqlite      # macOS / Linux
certutil -hashfile data\raw\database.sqlite SHA256   # Windows
```

The hash must match the SHA-256 above.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import sklearn; print(sklearn.__version__)"   # should print 1.9.1
```

Tested with Python 3.12 (macOS) and Python 3.13; the preprocessing notebooks were also run on Windows.
`requirements.txt` pins every library version, but not the Python version itself.

## Reproducing the results

Run the notebooks **in this order**, each top to bottom (**Kernel → Restart & Run All**). The full pipeline takes
under a minute.

| # | Notebook | Owner | Produces | Needs the database? |
|---|---|---|---|---|
| 1 | `01_data_understanding.ipynb` | Ahamed | Table inspection, coverage figure (`outputs/figures/eda_01_*`) | Yes |
| 2 | `02_eda.ipynb` | Ahamed | Target, data-quality and pattern analysis (`outputs/figures/eda_02_*` – `eda_10_*`) | Yes |
| 3 | `03_preprocessing.ipynb` | Umair | Target creation, column exclusion, chronological sort | Yes |
| 4 | `04_feature_engineering.ipynb` | Umair | **`data/processed/features.csv`** (22,653 matches × 17 columns) | Yes |
| 5 | `04a_feature_validation.ipynb` | Ahamed | Independent rebuild and leakage tests of `features.csv` | Yes |
| 6 | `05_modeling.ipynb` | Yusuf | Tuning on 2014/15, final models, `outputs/predictions/test_predictions.csv` | No |
| 7 | `06_evaluation.ipynb` | Yusuf | `outputs/tables/model_comparison.csv`, per-class metrics, confusion matrices | No |
| 8 | `07_eda_to_recommendation.ipynb` | Ahamed | Accuracy and draw predictions by form gap | No |
| 9 | `08_model_selection.ipynb` | Jassim | Final model choice, feature importance and coefficients | No |

`data/processed/features.csv` is committed, so notebooks 05–08 can be run without downloading the database.

**Expected reproducibility** (checked by a clean re-run of all nine notebooks on 2026-10-11):

- Notebooks 01–04a reproduce their outputs exactly, including `features.csv`.
- Logistic Regression (the selected model), Decision Tree, KNN and the baseline reproduce exactly on any machine.
- **Random Forest** can differ slightly between operating systems / CPU architectures even with fixed seeds and identical
  library versions (e.g. macOS on Apple silicon vs Windows on x86): about 2.5% of its test predictions and up to ±0.003
  macro F1. Rankings and conclusions are unchanged. The final committed outputs are produced by one clean run on a
  single machine before submission.

## Key results

Test season 2015/16 (3,326 matches), never used for training or tuning:

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression (selected, D-029)** | 45.5% | **0.434** |
| Random Forest | 44.4% | 0.434 |
| Majority baseline | 43.9% | 0.203 |

The strongest pre-match signals are each team's earlier win rate **at the same venue** (home team at home, away team
away). Full comparison: `outputs/tables/model_comparison.csv`; narrative and recommendations:
`docs/eda_to_recommendation.md`.

## Documentation

| File | Contents |
|---|---|
| `logs/decision_log.md` | Every project decision with options considered and reasons (D-001 onwards) |
| `logs/eda_insight_log.md` | EDA findings with evidence and modelling impact (EDA-01 onwards) |
| `docs/data_dictionary.md` | Raw database columns and the engineered `features.csv` |
| `docs/preprocessing_feature_log.md` | Preprocessing, feature engineering and split details |
| `docs/eda_handoff.md` | What each phase needed from the EDA |
| `docs/eda_to_recommendation.md` | EDA findings vs model results, recommendations and limitations |
| `docs/ai_use_declaration.md` | AI-tool use, declared per member |

## Loading the data

Always load data through the shared helper in `src/db.py`, never with ad-hoc `sqlite3` calls, so every
member works from identical tables. The connection is read-only, so the raw file cannot be changed.

```python
import sys; sys.path.append("..")   # when running from notebooks/
from src.db import list_tables, load_matches, load_teams, load_team_attributes

list_tables()                  # every table with row/column counts
matches = load_matches()       # Match + country/league/team names, date parsed
teams = load_teams()
team_attrs = load_team_attributes()
```

## Repository structure

```
data/raw/          database.sqlite (not committed)
data/processed/    engineered datasets (not committed)
docs/              data dictionary, diagrams, member trackers
logs/              decision log, EDA insight log, preprocessing log
notebooks/         01 data understanding → 06 evaluation, run in order
outputs/           figures, tables, predictions
src/               shared Python modules
models/            saved models (not committed)
```
