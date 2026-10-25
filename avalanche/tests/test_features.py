import pandas as pd

from src.features import engineer_features, predictor_columns
from src.collect_caic import collect


def test_calendar_aspect_and_rolling_features():
    frame = pd.DataFrame({
        "point_id": ["a", "a", "a"], "date": pd.date_range("2024-01-01", periods=3),
        "aspect": [0, 90, 180], "snowfall_sum": [1.0, 2.0, 3.0],
        "precipitation_sum": [1.0, 1.0, 1.0],
    })
    result = engineer_features(frame)
    assert "day_of_year_sin" in result
    assert "aspect_sin" in result
    assert result["snowfall_sum_3d"].tolist() == [0.0, 1.0, 3.0]


def test_labels_and_post_event_fields_are_excluded():
    frame = pd.DataFrame({"date": ["2024-01-01"], "avalanche": [1], "size_d": [2.0],
                          "event_id": ["secret"], "trigger_type": ["human"], "caught": [1],
                          "buried": [0], "date_certainty": ["exact"], "elevation_m": [3000]})
    features = engineer_features(frame)
    assert not ({"avalanche", "size_d", "event_id", "trigger_type", "caught", "buried", "date"} & set(features.columns))
    assert predictor_columns(features) == ["elevation_m", "day_of_year_sin", "day_of_year_cos"]


def test_caic_table_csv_keeps_distinct_slides_and_drops_estimated_dates(tmp_path):
    source = tmp_path / "caic.csv"
    output = tmp_path / "normalized.parquet"
    pd.DataFrame([
        {"Observation ID": "report-1", "Date": "2024-12-01T18:00:00Z", "Date Known": "Known",
         "Longitude": -106.0, "latitude": 39.0, "Destructive Size": "D1.5", "Area": "Front Range",
         "Aspect": "NE", "Trigger": "N", "Type": "WL"},
        {"Observation ID": "report-1", "Date": "2024-12-01T18:00:00Z", "Date Known": "Known",
         "Longitude": -106.1, "latitude": 39.1, "Destructive Size": "D2", "Area": "Front Range",
         "Aspect": "N", "Trigger": "AS", "Type": "SS"},
        {"Observation ID": "report-2", "Date": "2024-12-02T18:00:00Z", "Date Known": "Estimated",
         "Longitude": -106.2, "latitude": 39.2, "Destructive Size": "D1", "Area": "Front Range",
         "Aspect": "S", "Trigger": "N", "Type": "WL"},
    ]).to_csv(source, index=False)

    collect(source, output)
    result = pd.read_parquet(output)
    assert len(result) == 2
    assert result["date_certainty"].tolist() == ["exact", "exact"]
    assert result["size_d"].tolist() == [1.5, 2.0]
    assert result["caught"].isna().all()
