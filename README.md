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
