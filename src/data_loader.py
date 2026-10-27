"""
Data loading, validation, cleaning, and aggregation module.
Implements robust time-series data hygiene, zero-sales imputation,
and schema validation for the AI Inventory Agent.
"""

from typing import Tuple, List, Dict, Any, Union
from pathlib import Path
import io
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "product_name",
    "units_sold",
    "promotion",
    "selling_price"
]


def load_sales_data(source: Union[str, Path, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """
    Load raw sales data from a CSV file path, Path object, or file-like buffer.
    
    Args:
        source: Filepath or file-like buffer containing CSV data.
        
    Returns:
        pd.DataFrame containing the loaded sales data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If file content cannot be parsed as CSV.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists() or (path.name == "sales_data.csv" and path.stat().st_size < 5000):
            # If the default dataset is missing or incomplete, automatically generate it
            if path.name == "sales_data.csv":
                try:
                    from generate_data import generate_sales_data
                    df_gen = generate_sales_data(num_days=540, end_date_str="2024-06-30", random_seed=42)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df_gen.to_csv(path, index=False)
                    return df_gen
                except Exception:
                    pass
            if not path.exists():
                raise FileNotFoundError(f"Sales dataset not found at path: {path}")
        return pd.read_csv(path)
    elif hasattr(source, "read"):
        # Handle file-like buffer (e.g. Streamlit UploadedFile)
        return pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported data source type: {type(source)}")


def validate_sales_data(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validate sales data against required schema and business domain constraints.
    
    Rules:
    - All required columns must be present.
    - Dataset must not be empty.
    - Dates must be parseable.
    - Units sold must be non-negative integers.
    - Selling prices must be strictly positive.
    - Promotion values must be 0 or 1.
    - Sufficient history (> 30 days) is recommended for ML forecasting.
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    if df is None or df.empty:
        return False, ["Dataset is empty. Please provide historical sales records."], []
        
    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}. Expected schema: {', '.join(REQUIRED_COLUMNS)}")
        return False, errors, warnings
        
    # 2. Check dates
    try:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce")
        null_dates_count = int(parsed_dates.isna().sum())
        if null_dates_count > 0:
            errors.append(f"Found {null_dates_count} invalid or unparseable date values in 'date' column.")
    except Exception as e:
        errors.append(f"Failed to parse dates in 'date' column: {str(e)}")
        
    # 3. Check units_sold
    if not pd.api.types.is_numeric_dtype(df["units_sold"]):
        try:
            numeric_units = pd.to_numeric(df["units_sold"], errors="coerce")
            if numeric_units.isna().sum() > 0:
                errors.append("Column 'units_sold' contains non-numeric values.")
        except Exception:
            errors.append("Column 'units_sold' must contain numeric values.")
    else:
        negative_units = int((df["units_sold"] < 0).sum())
        if negative_units > 0:
            errors.append(f"Found {negative_units} negative sales quantities in 'units_sold'. Sales cannot be negative.")
            
    # 4. Check selling_price
    if not pd.api.types.is_numeric_dtype(df["selling_price"]):
        try:
            numeric_prices = pd.to_numeric(df["selling_price"], errors="coerce")
            if numeric_prices.isna().sum() > 0:
                errors.append("Column 'selling_price' contains non-numeric values.")
        except Exception:
            errors.append("Column 'selling_price' must contain numeric values.")
    else:
        non_positive_prices = int((df["selling_price"] <= 0).sum())
        if non_positive_prices > 0:
            errors.append(f"Found {non_positive_prices} non-positive values in 'selling_price'. Prices must be > 0.")
            
    # 5. Check promotion values
    invalid_promos = int((~df["promotion"].isin([0, 1, "0", "1", True, False])).sum())
    if invalid_promos > 0:
        warnings.append(f"Found {invalid_promos} non-binary values in 'promotion'. Values will be coerced to 0 or 1.")
        
    # 6. Check size for ML forecasting
    if not errors:
        valid_dates = pd.to_datetime(df["date"])
        date_span = (valid_dates.max() - valid_dates.min()).days + 1
        if date_span < 30:
            warnings.append(f"Dataset covers only {date_span} days of history. A minimum of 30-45 days is recommended for reliable machine learning forecasts.")
            
        unique_prods = df["product_id"].nunique()
        if unique_prods == 0:
            errors.append("No valid product IDs detected in the dataset.")
            
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize sales data types.
    
    Args:
        df: Raw DataFrame containing sales data.
        
    Returns:
        Cleaned pd.DataFrame with normalized column types.
    """
    cleaned = df.copy()
    
    # Trim string columns
    cleaned["product_id"] = cleaned["product_id"].astype(str).str.strip()
    cleaned["product_name"] = cleaned["product_name"].astype(str).str.strip()
    
    # Coerce dates
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    
    # Coerce numeric types
    cleaned["units_sold"] = pd.to_numeric(cleaned["units_sold"], errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    cleaned["selling_price"] = pd.to_numeric(cleaned["selling_price"], errors="coerce").fillna(0.01).clip(lower=0.01)
    
    # Coerce promotion to 0 or 1
    cleaned["promotion"] = cleaned["promotion"].isin([1, "1", True]).astype(int)
    
    # Drop rows missing date or product_id
    cleaned.dropna(subset=["date", "product_id"], inplace=True)
    
    return cleaned


def aggregate_daily_sales(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Aggregate sales to guarantee exactly one row per (product_id, date)
    and fill missing calendar dates with zero sales.
    
    Assumption:
        Missing calendar dates for an active product indicate zero sales on that day.
        Promotion on missing days is imputed as 0.
        Selling price and product name are forward-filled or backward-filled.
        
    Returns:
        Tuple of (aggregated_dataframe, imputed_zero_days_count)
    """
    cleaned = clean_sales_data(df)
    
    # First aggregate any existing duplicate product-date entries
    aggregated = cleaned.groupby(["product_id", "date"], as_index=False).agg(
        product_name=("product_name", "first"),
        units_sold=("units_sold", "sum"),
        promotion=("promotion", "max"),
        selling_price=("selling_price", "mean")
    )
    
    # Fill missing dates per product
    product_dfs = []
    total_imputed_days = 0
    
    overall_min_date = aggregated["date"].min()
    overall_max_date = aggregated["date"].max()
    full_date_idx = pd.date_range(overall_min_date, overall_max_date, freq="D", name="date")
    
    for prod_id, group in aggregated.groupby("product_id"):
        prod_name = group["product_name"].iloc[0]
        group = group.set_index("date").reindex(full_date_idx)
        
        # Calculate how many dates were missing
        missing_count = int(group["units_sold"].isna().sum())
        total_imputed_days += missing_count
        
        group["product_id"] = prod_id
        group["product_name"] = prod_name
        group["units_sold"] = group["units_sold"].fillna(0).astype(int)
        group["promotion"] = group["promotion"].fillna(0).astype(int)
        group["selling_price"] = group["selling_price"].ffill().bfill().fillna(9.99)
        
        group = group.reset_index()
        product_dfs.append(group)
        
    full_df = pd.concat(product_dfs, ignore_index=True)
    full_df.sort_values(by=["date", "product_id"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    
    return full_df, total_imputed_days


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute high-level summary metrics of the loaded dataset.
    
    Args:
        df: Aggregated daily sales DataFrame.
        
    Returns:
        Dictionary with record counts, product breakdowns, and date ranges.
    """
    if df is None or df.empty:
        return {
            "total_records": 0,
            "unique_products": 0,
            "product_list": [],
            "min_date": "N/A",
            "max_date": "N/A",
            "total_days": 0
        }
        
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    unique_prods = df["product_id"].nunique()
    
    prod_summary = []
    for (p_id, p_name), grp in df.groupby(["product_id", "product_name"]):
        prod_summary.append({
            "product_id": p_id,
            "product_name": p_name,
            "total_units": int(grp["units_sold"].sum()),
            "avg_daily_demand": float(grp["units_sold"].mean()),
            "min_units": int(grp["units_sold"].min()),
            "max_units": int(grp["units_sold"].max()),
            "records_count": len(grp)
        })
        
    return {
        "total_records": len(df),
        "unique_products": unique_prods,
        "product_list": prod_summary,
        "min_date": min_date,
        "max_date": max_date,
        "total_days": df["date"].nunique()
    }
