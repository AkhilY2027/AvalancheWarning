"""Fetch elevation at report locations; slope/aspect are retained from CAIC reports."""
import argparse
from pathlib import Path

import pandas as pd

from config import CAIC_CSV, RAW_DIR, TERRAIN_FILE, USGS_EPQS_URL
from src.common import cached_get, LOG


def collect(points: Path, output: Path) -> None:
    df = pd.read_csv(points).dropna(subset=["latitude", "longitude"])
    df = df.drop_duplicates(["latitude", "longitude"])
    rows = []
    cache = RAW_DIR / "terrain" / "cache"
    for i, row in df.reset_index(drop=True).iterrows():
        response = cached_get(USGS_EPQS_URL, params={"x": row.longitude, "y": row.latitude,
                                "wkid": 4326, "units": "Meters", "output": "json", "includeDate": "false"},
                             cache_file=cache / f"{i}.json")
        result = response.get("USGS_Elevation_Point_Query_Service", {}).get("Elevation_Query", {})
        rows.append({"latitude": row.latitude, "longitude": row.longitude,
                     "elevation_m": result.get("Elevation"),
                     "slope_angle": row.get("slope_angle"), "aspect": row.get("aspect")})
        LOG.info("Terrain locations %s/%s", i + 1, len(df))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, default=CAIC_CSV)
    p.add_argument("--output", type=Path, default=TERRAIN_FILE)
    a = p.parse_args()
    collect(a.points, a.output)


if __name__ == "__main__":
    main()
