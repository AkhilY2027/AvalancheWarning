"""Central configuration for the Colorado avalanche research prototype."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Full historical target from the requested winter-season plan.
START_SEASON = 2010
END_SEASON = 2025
FULL_START_SEASON = 2010
FULL_END_SEASON = 2025  # Last completed season as of 2026-10-24.
# Leave unset to use the configured Nov 1–May 31 seasons.
DATA_START_DATE = None
DATA_END_DATE = None
ZONES = ["Aspen", "Steamboat & Flat Tops", "Vail & Summit County", "Northern San Juan",
         "Southern San Juan", "Gunnison", "Sawatch", "Front Range", "Sangre de Cristo", "Grand Mesa"]
NEGATIVE_RATIO = 5
USER_AGENT = "ColoradoAvalancheResearchPrototype/0.1 (research; contact: prototype-maintainer)"
REQUEST_TIMEOUT = 45
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 5

CAIC_CSV = RAW_DIR / "caic" / "avalanche_observations.csv"
CAIC_DOWNLOAD_FILE = RAW_DIR / "caic" / "caic-avalanche-records-2026-10-25.csv"
SNOTEL_FILE = RAW_DIR / "snotel" / "daily.parquet"
WEATHER_FILE = RAW_DIR / "weather" / "daily.parquet"
TERRAIN_FILE = RAW_DIR / "terrain" / "points.parquet"
TRAINING_FILE = PROCESSED_DIR / "training.parquet"
MODEL_FILE = MODEL_DIR / "hurdle_model.joblib"
REPORT_FILE = REPORT_DIR / "metrics.json"

AWDB_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USGS_EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
