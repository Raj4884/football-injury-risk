"""Load SoccerMon subjective files and build the daily player table.

Files are in wide format for load (Date + one column per player), and in long
format for injuries (player_name, type, timestamp). Dates are dd.mm.yyyy.
"""
import pandas as pd

DATE_FMT = "%d.%m.%Y"


def load_wide_daily_load(path: str) -> pd.DataFrame:
    """daily_load.csv (wide) -> long frame: player_id, date, load."""
    wide = pd.read_csv(path)
    wide["Date"] = pd.to_datetime(wide["Date"], format=DATE_FMT)
    long = wide.melt(id_vars="Date", var_name="player_id", value_name="load")
    long = long.rename(columns={"Date": "date"})
    long["load"] = long["load"].fillna(0.0)
    return long


def load_injury_onsets(path: str, gap_days: int = 7) -> pd.DataFrame:
    """injury.csv -> one row per injury onset: player_id, date.

    Players file a report on each day an injury is active, so consecutive daily
    reports are one episode. A report counts as a new onset when the player has
    no report in the previous `gap_days` days.
    """
    inj = pd.read_csv(path)
    inj["date"] = pd.to_datetime(inj["timestamp"], format=DATE_FMT)
    inj = inj.rename(columns={"player_name": "player_id"})
    days = inj[["player_id", "date"]].drop_duplicates().sort_values(["player_id", "date"])
    prev = days.groupby("player_id")["date"].shift(1)
    new = prev.isna() | ((days["date"] - prev).dt.days > gap_days)
    return days[new].reset_index(drop=True)


def build_daily(load_path: str, injury_path: str, gap_days: int = 7) -> pd.DataFrame:
    """Return player_id, date, load, injury (1 on onset day), trimmed to each
    player's active window (first day with load > 0 until last load or onset)."""
    load = load_wide_daily_load(load_path)
    onsets = load_injury_onsets(injury_path, gap_days)
    onsets["injury"] = 1
    df = load.merge(onsets, on=["player_id", "date"], how="outer")
    df["load"] = df["load"].fillna(0.0)
    df["injury"] = df["injury"].fillna(0).astype(int)

    keep = []
    for pid, g in df.groupby("player_id"):
        active = g[(g["load"] > 0) | (g["injury"] == 1)]
        if active.empty:
            continue
        keep.append(g[(g["date"] >= active["date"].min()) & (g["date"] <= active["date"].max())])
    return pd.concat(keep, ignore_index=True).sort_values(["player_id", "date"]).reset_index(drop=True)
