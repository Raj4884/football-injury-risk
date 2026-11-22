# Football injury risk prediction

Predicts the chance that a player picks up an injury in the next 7 days from their recent training load, so staff can adjust sessions before it happens.

**Status:** early work. Data loading and feature pipeline are in place; modelling is next.

## Data
[SoccerMon](https://zenodo.org/records/10033832): two seasons of subjective training-load, wellness and injury reports from two elite Norwegian women's teams (Midoglu et al., *Scientific Data*, 2024). Only the subjective files are used. The data is not stored in this repo; the notebook downloads it from Zenodo.

Limits to keep in mind: women's elite football only, two teams, and injuries are rare, so the class balance is heavily skewed.

## Approach
1. Turn per-session reports into a daily load series per player.
2. Build rolling features from past days only: 7-day load, 28-day load, acute:chronic ratio, monotony, strain, rest days, days since last injury.
3. Target: any injury onset in the next 7 days.
4. Split by time (not by random row). Compare against simple baselines before trying LightGBM.
5. Report PR-AUC, calibration and precision at a fixed number of weekly alerts, not accuracy.

## Run it
Open `notebooks/01_data_and_baseline.ipynb` in Google Colab and run top to bottom.

## Layout
```
notebooks/   Colab notebooks
src/         feature code
data/        downloaded data (git-ignored)
reports/     figures and results
```

## Not for
This is a research prototype. It is not a medical tool and should not decide a player's availability by itself.
