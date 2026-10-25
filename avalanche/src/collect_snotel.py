"""Collect daily SNOTEL SWE/precipitation for Colorado stations via AWDB REST."""
import argparse
import json
from pathlib import Path

import pandas as pd

from config import AWDB_BASE_URL, RAW_DIR, SNOTEL_FILE, START_SEASON, END_SEASON
from src.common import LOG, cached_get


def collect(start_season: int, end_season: int, output: Path) -> None:
    cache = RAW_DIR / "snotel" / "cache"
    stations = cached_get(
        f"{AWDB_BASE_URL}/services/v1/stations",
        params={"stationTriplets": "*:CO:SNTL", "elements": "WTEQ,PREC",
                "durations": "DAILY", "returnStationElements": "true", "activeOnly": "false"},
        cache_file=cache / "stations.json",
    )
    frames = []
    for station in stations:
        triplet = station.get("stationTriplet") or station.get("station_triplet")
        if not triplet:
            continue
        for season in range(start_season, end_season + 1):
            begin, end = f"{season}-11-01", f"{season + 1}-05-31"
            key = triplet.replace(":", "_") + f"_{season}.json"
            try:
                payload = cached_get(
                    f"{AWDB_BASE_URL}/services/v1/data",
                    params={"stationTriplets": triplet, "elements": "WTEQ,PREC", "duration": "DAILY",
                            "beginDate": begin, "endDate": end, "returnFlags": "true"},
                    cache_file=cache / key,
                )
            except Exception as exc:
                LOG.warning("Skipping %s season %s: %s", triplet, season, exc)
                continue
            # OpenAPI StationDataDTO -> data[] -> values[{date,value,...}].
            station_obj = payload[0] if isinstance(payload, list) and payload else {}
            metadata = station_obj.get("station", station_obj)
            series_items = station_obj.get("data")
            if series_items is None and station_obj.get("stationElement") is not None:
                series_items = [station_obj]
            for series in series_items or []:
                element = series.get("stationElement", {}).get("elementCode")
                for value in series.get("values", []):
                    frames.append({"station_triplet": triplet, "station_name": metadata.get("name"),
                                   "latitude": metadata.get("latitude"), "longitude": metadata.get("longitude"),
                                   "date": value.get("date"), "element": element,
                                   "value": value.get("value"), "flags": value.get("flags")})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frames).to_parquet(output, index=False)
    LOG.info("Saved %s station-day-element rows to %s", len(frames), output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-season", type=int, default=START_SEASON)
    p.add_argument("--end-season", type=int, default=END_SEASON)
    p.add_argument("--output", type=Path, default=SNOTEL_FILE)
    a = p.parse_args()
    collect(a.start_season, a.end_season, a.output)


if __name__ == "__main__":
    main()
