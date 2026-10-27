"""
Tests for Data Loader, Validation, Cleaning, and Aggregation (src/data_loader.py).
"""

import io
import pytest
import pandas as pd
from src.data_loader import (
    load_sales_data,
    validate_sales_data,
    clean_sales_data,
    aggregate_daily_sales,
    get_dataset_summary
)


@pytest.fixture
def valid_sales_df():
    """Create a minimal valid synthetic dataset."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    records = []
    for d in dates:
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "product_id": "P001",
            "product_name": "Wireless Mouse",
            "units_sold": 20,
            "promotion": 0,
            "selling_price": 24.99
        })
    return pd.DataFrame(records)


def test_load_sales_data_from_buffer(valid_sales_df):
    """Test loading sales data from an in-memory CSV buffer."""
    csv_buffer = io.StringIO()
    valid_sales_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    loaded_df = load_sales_data(csv_buffer)
    assert not loaded_df.empty
    assert len(loaded_df) == 60
    assert "units_sold" in loaded_df.columns


def test_validate_sales_data_valid(valid_sales_df):
    """Test that a well-formed dataframe passes schema validation."""
    is_valid, errors, warnings = validate_sales_data(valid_sales_df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_sales_data_missing_columns():
    """Test rejection when required columns are absent."""
    broken_df = pd.DataFrame({
        "date": ["2024-01-01"],
        "product_id": ["P001"],
        "units_sold": [10]
        # missing product_name, promotion, selling_price
    })
    is_valid, errors, warnings = validate_sales_data(broken_df)
    assert is_valid is False
    assert any("Missing required columns" in err for err in errors)


def test_validate_sales_data_negative_units():
    """Test rejection when negative sales quantities are present."""
    invalid_df = pd.DataFrame([{
        "date": "2024-01-01",
        "product_id": "P001",
        "product_name": "Test Item",
        "units_sold": -5,
        "promotion": 0,
        "selling_price": 10.0
    }])
    is_valid, errors, warnings = validate_sales_data(invalid_df)
    assert is_valid is False
    assert any("negative sales" in err.lower() for err in errors)


def test_validate_sales_data_invalid_date():
    """Test rejection of unparseable dates."""
    invalid_df = pd.DataFrame([{
        "date": "NOT_A_DATE_XYZ",
        "product_id": "P001",
        "product_name": "Test Item",
        "units_sold": 10,
        "promotion": 0,
        "selling_price": 10.0
    }])
    is_valid, errors, warnings = validate_sales_data(invalid_df)
    assert is_valid is False
    assert any("date" in err.lower() for err in errors)


def test_aggregate_daily_sales_duplicates():
    """Test that duplicate product-date rows are aggregated (units summed)."""
    raw_df = pd.DataFrame([
        {"date": "2024-01-01", "product_id": "P001", "product_name": "Item A", "units_sold": 15, "promotion": 0, "selling_price": 10.0},
        {"date": "2024-01-01", "product_id": "P001", "product_name": "Item A", "units_sold": 10, "promotion": 1, "selling_price": 10.0},
        {"date": "2024-01-02", "product_id": "P001", "product_name": "Item A", "units_sold": 25, "promotion": 0, "selling_price": 10.0}
    ])
    aggregated, imputed = aggregate_daily_sales(raw_df)
    assert len(aggregated) == 2
    row_1 = aggregated[aggregated["date"] == pd.to_datetime("2024-01-01")].iloc[0]
    assert row_1["units_sold"] == 25  # 15 + 10
    assert row_1["promotion"] == 1    # max(0, 1)


def test_aggregate_daily_sales_missing_dates():
    """Test that missing dates in the sequence are imputed with zero sales."""
    raw_df = pd.DataFrame([
        {"date": "2024-01-01", "product_id": "P001", "product_name": "Item A", "units_sold": 10, "promotion": 0, "selling_price": 10.0},
        # Jan 2 is missing
        {"date": "2024-01-03", "product_id": "P001", "product_name": "Item A", "units_sold": 12, "promotion": 0, "selling_price": 10.0}
    ])
    aggregated, imputed = aggregate_daily_sales(raw_df)
    assert len(aggregated) == 3
    assert imputed == 1
    imputed_row = aggregated[aggregated["date"] == pd.to_datetime("2024-01-02")].iloc[0]
    assert imputed_row["units_sold"] == 0
    assert imputed_row["promotion"] == 0


def test_get_dataset_summary(valid_sales_df):
    """Test dataset summary metadata extraction."""
    summary = get_dataset_summary(valid_sales_df)
    assert summary["total_records"] == 60
    assert summary["unique_products"] == 1
    assert summary["min_date"] == "2024-01-01"
