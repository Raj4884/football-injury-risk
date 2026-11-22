"""Feature engineering for injury risk prediction.

Input: one row per player per day with columns
    player_id, date, load (e.g. session RPE x minutes, 0 on rest days), injury (0/1 onset flag)
Output: same frame plus rolling-load features and a forward-looking target.
"""
import numpy as np
import pandas as pd


def complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every player has a row for every day (missing days = rest, load 0)."""
    out = []
    for pid, g in df.groupby("player_id"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g["player_id"] = pid
        g["load"] = g["load"].fillna(0.0)
        g["injury"] = g["injury"].fillna(0).astype(int)
        out.append(g.rename_axis("date").reset_index())
    return pd.concat(out, ignore_index=True)


def add_load_features(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    df = df.sort_values(["player_id", "date"]).copy()
    g = df.groupby("player_id", group_keys=False)

    # Rolling windows use only past and current day, so there is no leakage.
    df["acute_7d"] = g["load"].transform(lambda s: s.rolling(7, min_periods=7).sum())
    df["chronic_28d"] = g["load"].transform(lambda s: s.rolling(28, min_periods=28).mean() * 7)
    df["acwr"] = df["acute_7d"] / df["chronic_28d"].replace(0, np.nan)

    mean7 = g["load"].transform(lambda s: s.rolling(7, min_periods=7).mean())
    std7 = g["load"].transform(lambda s: s.rolling(7, min_periods=7).std())
    df["monotony"] = mean7 / std7.replace(0, np.nan)
    df["strain"] = df["acute_7d"] * df["monotony"]

    df["load_change_wow"] = df["acute_7d"] - g["acute_7d"].shift(7)
    df["rest_days_7d"] = g["load"].transform(lambda s: (s == 0).rolling(7, min_periods=7).sum())

    # Days since the previous injury onset (NaN if none yet).
    df["_last_inj"] = df["date"].where(df["injury"] == 1)
    df["_last_inj"] = df.groupby("player_id")["_last_inj"].ffill()
    prev = df.groupby("player_id")["_last_inj"].shift(1)
    df["days_since_injury"] = (df["date"] - prev).dt.days
    df = df.drop(columns="_last_inj")

    # Target: any injury onset in the next `horizon` days (days t+1 .. t+horizon).
    def forward_any(s):
        rev = s[::-1].shift(1).rolling(horizon, min_periods=horizon).max()
        return rev[::-1]

    df["target"] = g["injury"].transform(forward_any)
    return df


FEATURES = [
    "acute_7d", "chronic_28d", "acwr", "monotony", "strain",
    "load_change_wow", "rest_days_7d", "days_since_injury",
]
