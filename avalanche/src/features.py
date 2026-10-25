"""Shared, leakage-aware predictor feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = {"avalanche", "size_d", "size_raw", "event_id", "report_id", "trigger_type", "caught", "buried", "number"}
LEAKAGE_COLUMNS = TARGET_COLUMNS | {"date_certainty", "date"}


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "date" in out:
        out = out.sort_values([c for c in ["point_id", "date"] if c in out], kind="stable")
    date = pd.to_datetime(out["date"])
    out["day_of_year_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    if "aspect" in out:
        out["aspect_sin"] = np.sin(np.deg2rad(pd.to_numeric(out["aspect"], errors="coerce")))
        out["aspect_cos"] = np.cos(np.deg2rad(pd.to_numeric(out["aspect"], errors="coerce")))
    for column in ("snowfall_sum", "precipitation_sum"):
        if column in out:
            if "point_id" in out:
                out[f"{column}_3d"] = out.groupby("point_id", dropna=False)[column].transform(
                    lambda s: s.shift(1).rolling(3, min_periods=1).sum().fillna(0)
                )
            else:
                out[f"{column}_3d"] = out[column].shift(1).rolling(3, min_periods=1).sum().fillna(0)
    # Strictly prohibit labels and post-event facts from entering either stage.
    return out.sort_index(kind="stable").drop(columns=list(LEAKAGE_COLUMNS & set(out.columns)), errors="ignore")


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = LEAKAGE_COLUMNS | {"latitude", "longitude", "forecast_zone", "point_id"}
    return [c for c in frame.columns if c not in excluded and
            (pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]))]
