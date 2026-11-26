import pandas as pd


def test_basic_dataframe_check():
    df = pd.DataFrame([
        {"line": "Red Line", "rain_mm": 3.0, "day_type": "weekend"},
        {"line": "Yellow Line", "rain_mm": 0.0, "day_type": "weekday"},
    ])
    assert len(df) == 2
    assert "line" in df.columns