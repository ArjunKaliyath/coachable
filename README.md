# CoachLens

Streamlit app for exploring FBref-style season stats:
- Role clustering with PCA visualization
- Impact-based squad selection suggestions
- Talent finder leaderboards

The app is configured to load the local CSV file `fbref_PL_2024-25.csv` automatically (no upload needed).

## Project structure
- `app.py` — Streamlit app
- `fbref_PL_2024-25.csv` — data source

## Requirements
- Python 3.10+ recommended
- Packages:
  - streamlit
  - pandas
  - numpy
  - plotly
  - scikit-learn

## Setup
Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install streamlit pandas numpy plotly scikit-learn
```

## Run the app
```powershell
.\.venv\Scripts\streamlit run app.py
```

## Data expectations
The app expects FBref-style columns including (not exhaustive):

```
Player, Nation, Pos, Squad, Age, Born, MP, Starts, Minutes Played, 90s,
Yellow Cards, Red Cards,
Non-penalty xG per 90, Expected assists per 90 minutes,
Non-penalty xG+xAG per 90, xG+xAG per 90,
Non-penalty G+A per 90,
Progressive Carries, Progressive Passes, Progressive Receptions
```

The app normalizes column headers by trimming whitespace and removing trailing colons
(e.g., `xG+xAG per 90:` becomes `xG+xAG per 90`).

## Troubleshooting
- **"Local CSV not found"**: Ensure `fbref_PL_2024-25.csv` is in the project root.
- **Missing columns error**: Make sure the CSV includes the required columns above.
- **Not enough players to cluster**: Reduce filters or reduce the number of clusters.

## Notes
This is a decision-support dashboard based on season aggregates, not match-by-match analysis.
