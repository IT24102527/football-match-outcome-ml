# Data Dictionary — European Soccer Database

Owner: Ahamed M.A.U (IT24101779) · Phase 1: Data Understanding & EDA
Source: `data/raw/database.sqlite` (Kaggle `hugomathien/soccer`, SHA-256 `4df85697…f11139a9`, see README)
All counts and percentages below are computed from the full database (`notebooks/01_data_understanding.ipynb`).

### Role legend

| Role | Meaning |
|---|---|
| **Key** | Identifier used only for joins/grouping, never as a model input |
| **Context** | Known before kickoff; used for ordering, splitting or as a feature input |
| **Target source** | Used only to build the target; known only after the match |
| **Feature source** | Raw input from which pre-match features are engineered |
| **Optional feature** | Candidate feature, usable only under conditions (D-012) |
| **Leakage** | Known only after kickoff; must never be a model input (D-010) |
| **Excluded** | Not used, for a documented reason |
| **Lookup** | Human-readable label for EDA and reporting |

---

## 1. `Match`: 25,979 rows × 115 columns

**One row = one league fixture**: a home team against an away team on a date, in one league, season and round.
`id` and `match_api_id` are both unique. Coverage: 11 leagues, 8 seasons (2008/09–2015/16), 2008-07-18 to 2016-05-25.

### 1.1 Identifiers, context and goals

| Column | Type (SQLite → pandas) | Meaning | Example | % missing | Range / values | Role |
|---|---|---|---|---|---|---|
| `id` | INTEGER → int64 | Row ID | 1 | 0.0 | 1 – 25,979, unique | Key |
| `country_id` | INTEGER → int64 | Country of the league (→ `Country.id`) | 1 | 0.0 | 11 values | Key |
| `league_id` | INTEGER → int64 | League (→ `League.id`) | 1 | 0.0 | 11 values (same as `country_id`) | Key / Context |
| `season` | TEXT → str | Season label | `2008/2009` | 0.0 | 8 seasons, `2008/2009` – `2015/2016` | Context (chronological split, D-009) |
| `stage` | INTEGER → int64 | Matchday / round within the season | 1 | 0.0 | 1 – 38 | Context |
| `date` | TEXT → datetime | Match date (stored as text `YYYY-MM-DD 00:00:00`, no kickoff time) | `2008-08-17 00:00:00` | 0.0 | 2008-07-18 – 2016-05-25 (1,694 distinct days) | Context (ordering for all history features) |
| `match_api_id` | INTEGER → int64 | External match ID | 492473 | 0.0 | unique | Key |
| `home_team_api_id` | INTEGER → int64 | Home team (→ `Team.team_api_id`) | 9987 | 0.0 | 299 teams | Key / Feature source |
| `away_team_api_id` | INTEGER → int64 | Away team (→ `Team.team_api_id`) | 9993 | 0.0 | 299 teams | Key / Feature source |
| `home_team_goal` | INTEGER → int64 | Full-time goals by home team | 1 | 0.0 | 0 – 10 | **Target source**; feature source only via *previous* matches |
| `away_team_goal` | INTEGER → int64 | Full-time goals by away team | 1 | 0.0 | 0 – 9 | **Target source**; feature source only via *previous* matches |

### 1.2 Line-ups and player positions (66 columns)

| Column pattern | Count | Type | Meaning | Example | % missing | Role |
|---|---|---|---|---|---|---|
| `home_player_1` … `home_player_11`, `away_player_1` … `away_player_11` | 22 | INTEGER → float64 | `player_api_id` of each starting player (→ `Player`) | 39890.0 | 4.7 – 6.0 per column | Excluded: player-level, outside the team-level lens (D-003) |
| `home_player_X1` … `X11`, `away_player_X1` … `X11` | 22 | INTEGER → float64 | Horizontal grid position of each starter (formation) | 1.0 | 7.0 – 7.1 | Excluded: player-level |
| `home_player_Y1` … `Y11`, `away_player_Y1` … `Y11` | 22 | INTEGER → float64 | Vertical grid position of each starter (formation) | 1.0 | 7.0 – 7.1 | Excluded: player-level |

Declared INTEGER in SQLite but loaded as float64, because pandas represents missing values (NULL) as `NaN`, which needs a float column.

### 1.3 In-match events (8 columns)

| Column | Type | Meaning | Example (truncated) | % missing | Role |
|---|---|---|---|---|---|
| `goal` | TEXT (XML) | Goal events: scorer, assist, minute, type | `<goal><value><comment>n</comment><stats>…` | 45.3 | **Leakage** |
| `shoton` | TEXT (XML) | Shots on target | `<shoton><value><stats><blocked>1…` | 45.3 | **Leakage** |
| `shotoff` | TEXT (XML) | Shots off target | `<shotoff><value><stats><shotoff>1…` | 45.3 | **Leakage** |
| `foulcommit` | TEXT (XML) | Fouls committed | `<foulcommit><value><stats><foulscommitted>…` | 45.3 | **Leakage** |
| `card` | TEXT (XML) | Yellow/red cards | `<card><value><comment>y</comment>…` | 45.3 | **Leakage** |
| `cross` | TEXT (XML) | Crosses | `<cross><value><stats><crosses>1…` | 45.3 | **Leakage** |
| `corner` | TEXT (XML) | Corners | `<corner><value><stats><corners>1…` | 45.3 | **Leakage** |
| `possession` | TEXT (XML) | Ball possession % over time | `<possession><value><comment>56</comment>…` | 45.3 | **Leakage** |

All eight are recorded **during** the match, so they cannot be known before kickoff (D-010). They are also missing for 45% of
matches and need XML parsing, so they are excluded entirely.

### 1.4 Betting odds (30 columns)

All 30 columns are NUMERIC → float64. Decimal odds for Home win (`H`), Draw (`D`) and Away win (`A`) from 10 bookmakers, e.g. `B365H`, `B365D`, `B365A`.
Odds are set **before** kickoff, but they are **excluded as model features** (D-011).

| Prefix | Bookmaker | % missing (H column) | Seasons available | Range (H) | Role |
|---|---|---|---|---|---|
| `B365` | Bet365 | 13.0 | 2008/09 – 2015/16 | 1.04 – 26.0 | Excluded (D-011) |
| `BW` | Bet&Win | 13.1 | 2008/09 – 2015/16 | 1.03 – 34.0 | Excluded (D-011) |
| `IW` | Interwetten | 13.3 | 2008/09 – 2015/16 | 1.03 – 20.0 | Excluded (D-011) |
| `LB` | Ladbrokes | 13.2 | 2008/09 – 2015/16 | 1.04 – 26.0 | Excluded (D-011) |
| `PS` | Pinnacle | 57.0 | 2012/13 – 2015/16 | 1.04 – 36.0 | Excluded (D-011) |
| `WH` | William Hill | 13.1 | 2008/09 – 2015/16 | 1.02 – 26.0 | Excluded (D-011) |
| `SJ` | Stan James | 34.2 | 2008/09 – 2014/15 | 1.04 – 23.0 | Excluded (D-011) |
| `VC` | VC Bet | 13.1 | 2008/09 – 2015/16 | 1.03 – 36.0 | Excluded (D-011) |
| `GB` | Gamebookers | 45.5 | 2008/09 – 2012/13 | 1.05 – 21.0 | Excluded (D-011) |
| `BS` | Blue Square | 45.5 | 2008/09 – 2012/13 | 1.04 – 17.0 | Excluded (D-011) |

Several bookmakers only cover part of the period: Pinnacle starts in 2012/13, and Gamebookers and Blue Square stop
after 2012/13. Bookmaker names follow the standard Football-Data column codes.

### 1.5 Derived column (built from the raw data)

| Column | Definition | Values | Role |
|---|---|---|---|
| `target` | `home_team_goal > away_team_goal` → Home Win; `=` → Draw; `<` → Away Win (D-004) | Home Win / Draw / Away Win | **Target** |

---

## 2. `Team`: 299 rows × 5 columns

One row = one team. All 299 teams appear in `Match`.

| Column | Type | Meaning | Example | % missing | Notes | Role |
|---|---|---|---|---|---|---|
| `id` | INTEGER → int64 | Row ID | 1 | 0.0 | unique | Excluded |
| `team_api_id` | INTEGER → int64 | Team ID used by `Match` and `Team_Attributes` | 9987 | 0.0 | unique, **the grouping key** | Key |
| `team_fifa_api_id` | INTEGER → float64 | Team ID in the FIFA ratings source | 673.0 | 3.7 (11 teams) | 285 distinct; 3 values shared by two teams | Excluded |
| `team_long_name` | TEXT → str | Full team name | `KRC Genk` | 0.0 | 296 distinct: **3 names used by two IDs** (see below) | Lookup |
| `team_short_name` | TEXT → str | 3-letter abbreviation | `GEN` | 0.0 | 259 distinct, not unique | Lookup |

**Data-quality issue:** "Polonia Bytom" (IDs 8031, 8020) and "Widzew Łódź" (8244, 8024) are each attached to two
different clubs, which played each other in the data. "Royal Excel Mouscron" (9996, 274581) covers two non-overlapping
periods. **Always group by `team_api_id`, never by name.**

---

## 3. `Team_Attributes`: 1,458 rows × 25 columns

One row = one team's FIFA-style tactical ratings at one **snapshot date**. Six snapshot dates: 2010-02-22, 2011-02-22,
2012-02-22, 2013-09-20, 2014-09-19, 2015-09-10. 288 of 299 teams have at least one snapshot.
**Use only the latest snapshot dated on or before the match (D-012).** That is possible for both teams in 74.5% of
matches, 0% in 2008/09.

Numeric ratings run on a 20–80 scale. Each `…Class` column is a categorical bin of the rating next to it. Three
dimensions exist only as classes.

| Column | Type | Meaning | Example | % missing | Range / categories | Role |
|---|---|---|---|---|---|---|
| `id` | int64 | Row ID | 1 | 0.0 | unique | Excluded |
| `team_fifa_api_id` | int64 | FIFA team ID | 434 | 0.0 | 285 distinct | Excluded |
| `team_api_id` | int64 | Team (→ `Team.team_api_id`) | 9930 | 0.0 | 288 teams | Key |
| `date` | TEXT → datetime | Snapshot date | `2010-02-22 00:00:00` | 0.0 | 6 dates | Key (time alignment) |
| `buildUpPlaySpeed` | int64 | Speed of attacking build-up | 60 | 0.0 | 20 – 80 | Optional feature |
| `buildUpPlaySpeedClass` | str | Binned speed | Balanced | 0.0 | Balanced 1184 / Fast 172 / Slow 102 | Excluded (duplicates the rating) |
| `buildUpPlayDribbling` | float64 | Amount of dribbling in build-up | 48.0 | **66.5** | 24 – 77 | Excluded (mostly missing) |
| `buildUpPlayDribblingClass` | str | Binned dribbling | Little | 0.0 | Little 1004 / Normal 433 / Lots 21 | Excluded |
| `buildUpPlayPassing` | int64 | Short (low) vs long (high) passing | 50 | 0.0 | 20 – 80 | Optional feature |
| `buildUpPlayPassingClass` | str | Binned passing | Mixed | 0.0 | Mixed 1236 / Short 128 / Long 94 | Excluded |
| `buildUpPlayPositioningClass` | str | Positional discipline in build-up | Organised | 0.0 | Organised 1386 / Free Form 72 | Optional feature (categorical) |
| `chanceCreationPassing` | int64 | Risk taken in chance-creating passes | 60 | 0.0 | 21 – 80 | Optional feature |
| `chanceCreationPassingClass` | str | Binned | Normal | 0.0 | Normal 1231 / Risky 171 / Safe 56 | Excluded |
| `chanceCreationCrossing` | int64 | Amount of crossing | 65 | 0.0 | 20 – 80 | Optional feature |
| `chanceCreationCrossingClass` | str | Binned | Normal | 0.0 | Normal 1195 / Lots 211 / Little 52 | Excluded |
| `chanceCreationShooting` | int64 | Amount of shooting | 55 | 0.0 | 22 – 80 | Optional feature |
| `chanceCreationShootingClass` | str | Binned | Normal | 0.0 | Normal 1224 / Lots 197 / Little 37 | Excluded |
| `chanceCreationPositioningClass` | str | Positional discipline in attack | Organised | 0.0 | Organised 1309 / Free Form 149 | Optional feature (categorical) |
| `defencePressure` | int64 | How high the team presses | 50 | 0.0 | 23 – 72 | Optional feature |
| `defencePressureClass` | str | Binned | Medium | 0.0 | Medium 1243 / Deep 154 / High 61 | Excluded |
| `defenceAggression` | int64 | Defensive aggression | 55 | 0.0 | 24 – 72 | Optional feature |
| `defenceAggressionClass` | str | Binned | Press | 0.0 | Press 1274 / Double 99 / Contain 85 | Excluded |
| `defenceTeamWidth` | int64 | Defensive width | 45 | 0.0 | 29 – 73 | Optional feature |
| `defenceTeamWidthClass` | str | Binned | Normal | 0.0 | Normal 1286 / Wide 111 / Narrow 61 | Excluded |
| `defenceDefenderLineClass` | str | Defensive line style | Cover | 0.0 | Cover 1362 / Offside Trap 96 | Optional feature (categorical) |

---

## 4. Lookup tables

| Table | Rows | Columns | Meaning | Role |
|---|---|---|---|---|
| `League` | 11 | `id`, `country_id`, `name` (e.g. `England Premier League`) | One top division per country; `id` = `country_id` | Lookup |
| `Country` | 11 | `id`, `name` (e.g. `Belgium`) | Belgium, England, France, Germany, Italy, Netherlands, Poland, Portugal, Scotland, Spain, Switzerland | Lookup |

## 5. Out-of-scope tables

| Table | Rows × columns | Contents | Why excluded |
|---|---|---|---|
| `Player` | 11,060 × 7 | Player name, birthday, height, weight | Player-level; the lens is team form and home advantage (D-003) |
| `Player_Attributes` | 183,978 × 42 | Dated FIFA player ratings | Player-level; would need line-up reconstruction and time alignment |
| `sqlite_sequence` | 7 × 2 | SQLite internal | Not data |
