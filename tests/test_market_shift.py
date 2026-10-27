"""
Tests for Market-Shift Detection Engine (src/market_shift.py).
"""

import pytest
import pandas as pd
import numpy as np
from src.market_shift import detect_market_shift
from generate_data import generate_sales_data


@pytest.fixture
def stable_demand_df():
    """Create a 60-day dataset with steady, stable demand around 20 units."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    records = []
    np.random.seed(123)
    for d in dates:
        noise = int(np.random.randint(-2, 3))
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "product_id": "P001",
            "product_name": "Stable Item",
            "units_sold": 20 + noise,
            "promotion": 0,
            "selling_price": 20.0
        })
    return pd.DataFrame(records)


@pytest.fixture
def collapsed_demand_df():
    """
    Create a 60-day dataset where the first 46 days have demand ~30,
    and the final 14 days collapse to ~4 units.
    """
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    records = []
    for i, d in enumerate(dates):
        # Days 0 to 45 (first 46 days): units ~30
        # Days 46 to 59 (last 14 days): units ~4
        units = 4 if i >= 46 else 30
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "product_id": "P005",
            "product_name": "Collapsing Item",
            "units_sold": units,
            "promotion": 0,
            "selling_price": 50.0
        })
    return pd.DataFrame(records)


def test_stable_demand_no_shift(stable_demand_df):
    """Verify that a steady demand dataset does NOT trigger false market-shift alerts."""
    result = detect_market_shift(stable_demand_df)
    assert result["shift_detected"] is False
    assert result["rule_a_triggered"] is False
    assert result["rule_b_triggered"] is False
    assert abs(result["percentage_change"]) < 35.0


def test_demand_collapse_triggers_rule_b(collapsed_demand_df):
    """Verify that a sustained 14-day collapse triggers Rule B (> 35% drop)."""
    result = detect_market_shift(collapsed_demand_df)
    assert result["shift_detected"] is True
    assert result["rule_b_triggered"] is True
    assert result["percentage_change"] < -50.0  # collapsed from 30 down to 4 (~ -86%)


def test_qualifying_deviations_trigger_rule_a():
    """
    Verify that >= 3 large forecast deviations in the last 7 days trigger Rule A.
    """
    # 50 days of history: days 0-42 normal ~20 units.
    # Last 7 days: day 44, 46, 48 spike wildly to 70 units while baseline is ~20.
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    records = []
    for i, d in enumerate(dates):
        if i in [44, 46, 48]:  # 3 days in the last 7 days
            units = 70  # > 200% deviation from ~20
        else:
            units = 20
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "product_id": "P002",
            "product_name": "Volatile Item",
            "units_sold": units,
            "promotion": 0,
            "selling_price": 20.0
        })
    df_volatile = pd.DataFrame(records)
    result = detect_market_shift(df_volatile)
    assert result["rule_a_triggered"] is True
    assert result["deviation_breach_count"] >= 3
    assert result["shift_detected"] is True


def test_insufficient_history_no_fabricated_shift():
    """Verify that short datasets report insufficient data rather than fabricating an alert."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df_short = pd.DataFrame([{
        "date": d.strftime("%Y-%m-%d"),
        "product_id": "P001",
        "product_name": "Short Item",
        "units_sold": 10,
        "promotion": 0,
        "selling_price": 10.0
    } for d in dates])
    
    result = detect_market_shift(df_short)
    assert result["shift_detected"] is False
    assert result["insufficient_data"] is True
    assert "Insufficient" in result["detection_reason"]


def test_winter_jacket_synthetic_data_triggers_shift():
    """
    Verify that the synthetic Winter Jacket (P005) from generate_data.py
    legitimately triggers market-shift detection due to its real sales drop.
    """
    full_df = generate_sales_data(num_days=540, end_date_str="2024-06-30", random_seed=42)
    jacket_df = full_df[full_df["product_id"] == "P005"].copy()
    
    result = detect_market_shift(jacket_df)
    assert result["shift_detected"] is True
    assert result["percentage_change"] < -35.0  # substantial decline
    assert "Winter Jacket" in result["product_name"]
