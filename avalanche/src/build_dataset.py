"""Build daily slope-point rows and sampled non-event labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CAIC_CSV, NEGATIVE_RATIO, PROCESSED_DIR, RAW_DIR, SNOTEL_FILE, TERRAIN_FILE, TRAINING_FILE, WEATHER_FILE, START_SEASON, END_SEASON, ZONES, DATA_START_DATE, DATA_END_DATE


def build(caic_path: Path, weather_path: Path, terrain_path: Path, snotel_path: Path,
          output: Path, start_season: int, end_season: int, zones: list[str], negative_ratio: int,
          start_date: str | None = DATA_START_DATE, end_date: str | None = DATA_END_DATE) -> pd.DataFrame:
    events = pd.read_parquet(caic_path) if caic_path.suffix == ".parquet" else pd.read_csv(caic_path)
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    events["latitude"] = pd.to_numeric(events["latitude"], errors="coerce")
    events["longitude"] = pd.to_numeric(events["longitude"], errors="coerce")
    events = events.dropna(subset=["date", "latitude", "longitude"])
    if zones:
        events = events[events.forecast_zone.isin(zones)]
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp(start_season, 11, 1)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(end_season + 1, 5, 31)
    events = events[events.date.between(start, end)]
    events = events[(events.date.dt.month >= 11) | (events.date.dt.month <= 5)]
    events["point_id"] = events.latitude.round(4).astype(str) + ":" + events.longitude.round(4).astype(str)
    points = events[["point_id", "latitude", "longitude", "forecast_zone", "elevation_m", "aspect", "slope_angle"]].drop_duplicates("point_id")
    positives = events.copy()
    positives["avalanche"] = 1
    # Pseudo-absence sampling: unreported point-days are not confirmed negatives.
    all_days = [day for day in pd.date_range(start, end, freq="D")
                if day.month >= 11 or day.month <= 5]
    rng = np.random.default_rng(42)
    negative_rows = []
    existing = set(zip(events.point_id, events.date))
    for _, point in points.iterrows():
        candidates = [day for day in all_days if (point.point_id, day) not in existing]
        chosen = rng.choice(len(candidates), size=min(len(candidates), max(negative_ratio, 1) * max(1, int((events.point_id == point.point_id).sum()))), replace=False) if candidates else []
        for ix in chosen:
            negative_rows.append({**point.to_dict(), "date": candidates[int(ix)], "avalanche": 0,
                                 "size_d": np.nan, "size_raw": np.nan, "event_id": None,
                                 "trigger_type": None, "caught": np.nan, "buried": np.nan})
    negatives = pd.DataFrame(negative_rows)
    data = pd.concat([positives, negatives], ignore_index=True, sort=False)
    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        weather["date"] = pd.to_datetime(weather["time"] if "time" in weather else weather["date"]).dt.normalize()
        weather["point_id"] = weather.latitude.round(4).astype(str) + ":" + weather.longitude.round(4).astype(str)
        data = data.merge(weather.drop(columns=["latitude", "longitude"], errors="ignore"),
                          on=["date", "point_id"], how="left")
    if terrain_path.exists():
        terrain = pd.read_parquet(terrain_path)
        terrain["point_id"] = terrain.latitude.round(4).astype(str) + ":" + terrain.longitude.round(4).astype(str)
        cols = [c for c in ["point_id", "elevation_m", "slope_angle", "aspect"] if c in terrain]
        data = data.drop(columns=[c for c in cols if c != "point_id" and c in data], errors="ignore").merge(terrain[cols], on="point_id", how="left")
    if snotel_path.exists():
        snow = pd.read_parquet(snotel_path)
        snow = snow.pivot_table(index="date", columns="element", values="value", aggfunc="mean").reset_index()
        snow["date"] = pd.to_datetime(snow.date).dt.normalize()
        snow.columns.name = None
        data = data.merge(snow, on="date", how="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caic", type=Path, default=CAIC_CSV.with_name("normalized.parquet"))
    p.add_argument("--weather", type=Path, default=WEATHER_FILE)
    p.add_argument("--terrain", type=Path, default=TERRAIN_FILE)
    p.add_argument("--snotel", type=Path, default=SNOTEL_FILE)
    p.add_argument("--output", type=Path, default=TRAINING_FILE)
    p.add_argument("--start-season", type=int, default=START_SEASON)
    p.add_argument("--end-season", type=int, default=END_SEASON)
    p.add_argument("--zones", nargs="+", default=ZONES)
    p.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    p.add_argument("--start-date", default=DATA_START_DATE,
                   help="Inclusive date override for a non-winter/small demonstration window")
    p.add_argument("--end-date", default=DATA_END_DATE,
                   help="Inclusive date override for a non-winter/small demonstration window")
    a = p.parse_args()
    result = build(a.caic, a.weather, a.terrain, a.snotel, a.output, a.start_season, a.end_season,
                   a.zones, a.negative_ratio, a.start_date, a.end_date)
    print(f"Wrote {len(result)} rows to {a.output}")


if __name__ == "__main__":
    main()
