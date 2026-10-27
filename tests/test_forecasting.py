"""
Tests for Demand Forecasting Engine and Leak-Free Feature Engineering (src/forecasting.py).
"""

import pytest
import pandas as pd
import numpy as np
from src.forecasting import (
    create_forecasting_features,
    train_forecasting_model,
    forecast_next_7_days,
    compute_moving_average_baseline
)


@pytest.fixture
def product_history_df():
    """Create a 120-day historical time-series for a product."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    np.random.seed(42)
    records = []
    base = 25
    for i, d in enumerate(dates):
        weekend_boost = 5 if d.weekday() in [5, 6] else 0
        noise = int(np.random.randint(-3, 4))
        units = max(0, base + weekend_boost + noise)
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "product_id": "P001",
            "product_name": "Wireless Mouse",
            "units_sold": units,
            "promotion": 1 if i % 15 == 0 else 0,
            "selling_price": 24.99
        })
    return pd.DataFrame(records)


def test_feature_engineering_no_leakage(product_history_df):
    """
    Verify that rolling and lag features do NOT leak current day information.
    Specifically:
    - lag_1 on row i equals units_sold on row i-1
    - lag_7 on row i equals units_sold on row i-7
    - rolling_mean_7 on row i is calculated strictly from row i-7 to i-1
    """
    df_feat = create_forecasting_features(product_history_df)
    
    # Check row index 10
    row_10 = df_feat.iloc[10]
    expected_lag_1 = df_feat.iloc[9]["units_sold"]
    expected_lag_7 = df_feat.iloc[3]["units_sold"]
    expected_rolling_7 = df_feat.iloc[3:10]["units_sold"].mean()
    
    assert row_10["lag_1"] == expected_lag_1
    assert row_10["lag_7"] == expected_lag_7
    assert pytest.approx(row_10["rolling_mean_7"], 0.01) == expected_rolling_7


def test_train_forecasting_model_adequate_data(product_history_df):
    """Verify that model trains successfully on adequate sample data."""
    model, features, meta = train_forecasting_model(product_history_df)
    assert model is not None
    assert meta["is_fallback"] is False
    assert meta["training_samples"] > 100
    assert len(meta["feature_importances"]) > 0


def test_forecast_horizon_exactly_7_days(product_history_df):
    """Verify that the forecast output contains exactly 7 daily predictions."""
    result = forecast_next_7_days(product_history_df)
    assert len(result["forecast_dates"]) == 7
    assert len(result["daily_forecast"]) == 7
    assert result["total_7_day_forecast"] == pytest.approx(sum(result["daily_forecast"]), 0.1)


def test_forecasts_strictly_non_negative(product_history_df):
    """Ensure that all daily forecasted demands are non-negative."""
    result = forecast_next_7_days(product_history_df)
    for pred in result["daily_forecast"]:
        assert pred >= 0.0


def test_small_dataset_triggers_fallback():
    """Verify that insufficient data triggers the documented moving-average fallback."""
    # Only 5 days of data
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    records = [{"date": d.strftime("%Y-%m-%d"), "product_id": "P001", "product_name": "Item", "units_sold": 10, "promotion": 0, "selling_price": 5.0} for d in dates]
    tiny_df = pd.DataFrame(records)
    
    result = forecast_next_7_days(tiny_df)
    assert result["is_fallback"] is True
    assert "Fallback" in result["model_name"]
    assert len(result["daily_forecast"]) == 7
    assert result["daily_forecast"][0] == 10.0


def test_adaptive_window_filters_properly(product_history_df):
    """Verify that adaptive window restricts training to recent days."""
    res_standard = forecast_next_7_days(product_history_df, adaptive_window=None)
    res_adaptive = forecast_next_7_days(product_history_df, adaptive_window=45)
    
    assert res_standard["forecast_method"] == "Standard (Full History)"
    assert "Adaptive (Recent 45 Days)" in res_adaptive["forecast_method"]
    assert res_adaptive["training_samples"] <= 45
