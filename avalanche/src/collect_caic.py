"""Normalize a human-downloaded CAIC avalanche CSV export."""
import argparse
import re
from pathlib import Path

import pandas as pd

from config import CAIC_CSV, CAIC_DOWNLOAD_FILE

REQUIRED = {
    "event_id", "date", "date_certainty", "latitude", "longitude", "forecast_zone",
    "elevation_m", "aspect", "slope_angle", "size_raw", "trigger_type", "caught", "buried",
}
HUMAN_TRIGGER_CODES = {"AS", "AM", "AR", "AF", "AE", "AC", "AB", "AO", "AI", "AU",
                       "AW", "AK", "AH", "AV", "AN", "AA", "AL", "AX"}


def _trigger_type(value):
    code = str(value).strip().upper()
    if code == "N":
        return "natural"
    if code in HUMAN_TRIGGER_CODES:
        return "human"
    return None


def collect(input_path: Path, output_path: Path) -> None:
    """Validate/normalize CSV fields; intentionally performs no automated CAIC requests."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"CAIC CSV not found: {input_path}. Download it manually from the CAIC avalanche table, "
            "then convert it to the documented schema in README.md. Automated scraping is not permitted by CAIC terms."
        )
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if "Observation ID" in df.columns:
        df = _normalize_caic_table_export(df)
    else:
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"CAIC CSV missing required normalized columns: {sorted(missing)}")
        df["date"] = df["date"].map(_local_date)
    df["date"] = df["date"].map(_local_date)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["date", "latitude", "longitude"])
    df = df[df["date_certainty"].astype(str).str.lower().isin(["exact", "known", "within_1_day", "±1_day", "+/-1 day"])]
    df["size_raw"] = pd.to_numeric(df["size_raw"], errors="coerce")
    df["size_d"] = (df["size_raw"] * 2).round() / 2
    # In CAIC's table export Observation ID identifies a report and can repeat for
    # several slide rows. Preserve distinct coordinates, deduplicating repeated rows.
    dedup_columns = [c for c in ["event_id", "date", "latitude", "longitude", "forecast_zone", "aspect", "size_d"] if c in df]
    df = df.drop_duplicates(subset=dedup_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Rows retained: {len(df)}; exact dates: {(df.date_certainty == 'exact').sum()}; "
          f"date range: {df.date.min() if len(df) else 'none'} to {df.date.max() if len(df) else 'none'}")


def _local_date(value):
    if pd.isna(value):
        return pd.NaT
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/Denver").tz_localize(None)
    return stamp.normalize().date()


def _numeric_size(value):
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def _normalize_caic_table_export(df: pd.DataFrame) -> pd.DataFrame:
    """Translate the observed CAIC table CSV headers and codes to model columns."""
    required_export = {"Observation ID", "Date", "Date Known", "Longitude", "latitude",
                       "Destructive Size", "Area", "Aspect", "Trigger"}
    missing = required_export - set(df.columns)
    if missing:
        raise ValueError(f"CAIC table export is missing columns: {sorted(missing)}")
    date_known = df["Date Known"].astype(str).str.strip().str.lower()
    aspect_degrees = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180,
                      "SW": 225, "W": 270, "NW": 315}
    rows = pd.DataFrame({
        "event_id": df["Observation ID"].astype(str) + ":" + df["Longitude"].astype(str) + ":" + df["latitude"].astype(str),
        "report_id": df["Observation ID"],
        "date": df["Date"].map(_local_date),
        "date_certainty": date_known.map(lambda v: "exact" if v == "known" else "estimated"),
        "latitude": df["latitude"],
        "longitude": df["Longitude"],
        "forecast_zone": df["Area"],
        # UI Elevation is an ATL/TL/BTL band, not a numeric elevation.
        "elevation_m": pd.NA,
        "aspect": df["Aspect"].astype(str).str.upper().map(aspect_degrees),
        # The CSV does not expose a slope angle.
        "slope_angle": pd.NA,
        "size_raw": df["Destructive Size"].map(_numeric_size),
        "trigger_type": df["Trigger"].map(_trigger_type),
        # CAIC's '#' is avalanche count in a report, not people caught/buried.
        "caught": pd.NA,
        "buried": pd.NA,
        "avalanche_type": df.get("Type"),
        "number": df.get("#"),
        "location": df.get("Location"),
        "raw_description": df.get("Description"),
        "raw_comments": df.get("Comments"),
        "location_quality": "public_export_coordinate_may_be_obfuscated",
    })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_input = CAIC_DOWNLOAD_FILE if CAIC_DOWNLOAD_FILE.exists() else CAIC_CSV
    p.add_argument("--input", type=Path, default=default_input,
                   help="Manually downloaded CAIC table export or normalized CSV")
    p.add_argument("--output", type=Path, default=CAIC_CSV.with_name("normalized.parquet"))
    a = p.parse_args()
    collect(a.input, a.output)
    print(f"Normalized CAIC rows written to {a.output}")


if __name__ == "__main__":
    main()
