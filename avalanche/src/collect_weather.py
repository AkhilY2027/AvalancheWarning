"""Download daily ERA5-Land history from the Open-Meteo archive API."""
import argparse
from pathlib import Path

import pandas as pd

from config import OPEN_METEO_URL, RAW_DIR, WEATHER_FILE, START_SEASON, END_SEASON
from src.common import LOG, cached_get

VARIABLES = "snowfall_sum,precipitation_sum,temperature_2m_min,temperature_2m_max,wind_speed_10m_max,wind_gusts_10m_max"


def collect(points: Path, start_season: int, end_season: int, output: Path) -> None:
    locations = pd.read_csv(points).dropna(subset=["latitude", "longitude"]).drop_duplicates(["latitude", "longitude"])
    frames = []
    cache = RAW_DIR / "weather" / "cache"
    for i, row in locations.reset_index(drop=True).iterrows():
        for season in range(start_season, end_season + 1):
            begin, end = f"{season}-11-01", f"{season + 1}-05-31"
            params = {"latitude": row.latitude, "longitude": row.longitude, "start_date": begin,
                      "end_date": end, "daily": VARIABLES, "timezone": "America/Denver", "models": "era5_land"}
            payload = cached_get(OPEN_METEO_URL, params=params, cache_file=cache / f"{i}_{season}.json")
            daily = payload.get("daily", {})
            if not daily:
                continue
            frame = pd.DataFrame(daily)
            frame["latitude"], frame["longitude"] = row.latitude, row.longitude
            frames.append(frame)
        LOG.info("Weather locations %s/%s", i + 1, len(locations))
    output.parent.mkdir(parents=True, exist_ok=True)
    (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_parquet(output, index=False)
    LOG.info("Saved weather data to %s", output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, help="CSV with latitude,longitude columns; defaults to CAIC CSV")
    p.add_argument("--start-season", type=int, default=START_SEASON)
    p.add_argument("--end-season", type=int, default=END_SEASON)
    p.add_argument("--output", type=Path, default=WEATHER_FILE)
    a = p.parse_args()
    if not a.points:
        from config import CAIC_CSV
        a.points = CAIC_CSV
    collect(a.points, a.start_season, a.end_season, a.output)


if __name__ == "__main__":
    main()
