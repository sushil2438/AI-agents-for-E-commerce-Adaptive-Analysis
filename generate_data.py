"""
Synthetic Sales Dataset Generator for AI Inventory Management Agent.

Generates ~18 months (~540 days) of daily historical sales data for 5 products:
- P001: Wireless Mouse (Stable demand with weekly cycle)
- P002: Bluetooth Speaker (Moderate demand with moderate variance)
- P003: USB-C Cable (High, steady daily demand)
- P004: Phone Case (Moderate demand with periodic promotional spikes)
- P005: Winter Jacket (Seasonal winter demand with a sharp, recent post-season collapse)

Ensures reproducible data generation via a fixed random seed.
Output is saved to data/sales_data.csv.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_sales_data(
    num_days: int = 540,
    end_date_str: str = "2024-06-30",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic daily sales data for 5 products over the given number of days.
    
    Args:
        num_days: Number of consecutive daily records per product (default 540 = ~18 months).
        end_date_str: Latest date in the dataset (format YYYY-MM-DD).
        random_seed: Fixed random seed for reproducibility.
        
    Returns:
        pd.DataFrame containing historical sales records.
    """
    np.random.seed(random_seed)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=num_days - 1)
    
    date_list = [start_date + timedelta(days=i) for i in range(num_days)]
    
    products = [
        {
            "product_id": "P001",
            "product_name": "Wireless Mouse",
            "selling_price": 24.99,
            "base_demand": 22.0,
            "weekend_lift": 6.0,
            "noise_std": 2.5,
            "promo_freq": 0.08,
            "promo_lift": 8.0,
            "pattern": "stable"
        },
        {
            "product_id": "P002",
            "product_name": "Bluetooth Speaker",
            "selling_price": 49.99,
            "base_demand": 14.0,
            "weekend_lift": 5.0,
            "noise_std": 3.0,
            "promo_freq": 0.10,
            "promo_lift": 7.0,
            "pattern": "moderate"
        },
        {
            "product_id": "P003",
            "product_name": "USB-C Cable",
            "selling_price": 9.99,
            "base_demand": 45.0,
            "weekend_lift": 3.0,
            "noise_std": 3.5,
            "promo_freq": 0.05,
            "promo_lift": 10.0,
            "pattern": "high_stable"
        },
        {
            "product_id": "P004",
            "product_name": "Phone Case",
            "selling_price": 14.99,
            "base_demand": 28.0,
            "weekend_lift": 7.0,
            "noise_std": 3.5,
            "promo_freq": 0.15,
            "promo_lift": 14.0,
            "pattern": "promo_driven"
        },
        {
            "product_id": "P005",
            "product_name": "Winter Jacket",
            "selling_price": 89.99,
            "base_demand": 32.0,
            "weekend_lift": 8.0,
            "noise_std": 4.0,
            "promo_freq": 0.07,
            "promo_lift": 10.0,
            "pattern": "seasonal_decline"
        }
    ]
    
    records = []
    
    for prod in products:
        p_id = prod["product_id"]
        p_name = prod["product_name"]
        price = prod["selling_price"]
        base = prod["base_demand"]
        weekend_boost = prod["weekend_lift"]
        noise_std = prod["noise_std"]
        promo_prob = prod["promo_freq"]
        promo_boost = prod["promo_lift"]
        pattern = prod["pattern"]
        
        for i, dt in enumerate(date_list):
            day_of_week = dt.weekday()  # Monday = 0, Sunday = 6
            is_weekend = 1 if day_of_week in [5, 6] else 0
            
            # Determine promotion status
            has_promo = 1 if np.random.rand() < promo_prob else 0
            
            # Base calculation
            expected_demand = base + (is_weekend * weekend_boost) + (has_promo * promo_boost)
            
            # Specific product variations
            if pattern == "seasonal_decline":
                # For Winter Jacket:
                # Previous history: strong seasonal demand (peak around 28-36 units)
                # In the last 30 days: sharp demand collapse down to 2 - 6 units (spring arrival / off-season)
                days_from_end = (end_date - dt).days
                if days_from_end <= 30:
                    # Sharp drop: demand collapses by ~80-85%
                    expected_demand = 3.5 + (is_weekend * 1.5) + (has_promo * 1.0)
                    noise = np.random.normal(0, 1.2)
                else:
                    # Normal seasonal winter pattern
                    expected_demand = base + (is_weekend * weekend_boost) + (has_promo * promo_boost)
                    noise = np.random.normal(0, noise_std)
            else:
                noise = np.random.normal(0, noise_std)
                
            units = int(np.maximum(0, np.round(expected_demand + noise)))
            
            records.append({
                "date": dt.strftime("%Y-%m-%d"),
                "product_id": p_id,
                "product_name": p_name,
                "units_sold": units,
                "promotion": has_promo,
                "selling_price": round(price, 2)
            })
            
    df = pd.DataFrame(records)
    df.sort_values(by=["date", "product_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    """Generate synthetic sales data and write to data/sales_data.csv."""
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / "sales_data.csv"
    
    print("=" * 60)
    print("Generating Synthetic E-Commerce Sales Dataset...")
    print("=" * 60)
    
    df = generate_sales_data(num_days=540, end_date_str="2024-06-30", random_seed=42)
    df.to_csv(output_file, index=False)
    
    print(f"Dataset successfully created at: {output_file}")
    print(f"Total Records: {len(df):,}")
    print(f"Date Range:    {df['date'].min()} to {df['date'].max()} ({df['date'].nunique()} distinct days)")
    print(f"Products ({df['product_id'].nunique()}):")
    
    summary = df.groupby(["product_id", "product_name"]).agg(
        total_units=("units_sold", "sum"),
        avg_daily_units=("units_sold", "mean"),
        min_units=("units_sold", "min"),
        max_units=("units_sold", "max"),
        promo_days=("promotion", "sum")
    ).reset_index()
    
    for _, row in summary.iterrows():
        print(f"  [{row['product_id']}] {row['product_name']:<18} | "
              f"Total: {row['total_units']:>5} | Avg/Day: {row['avg_daily_units']:>5.1f} | "
              f"Range: [{row['min_units']:>2} - {row['max_units']:>2}] | Promos: {row['promo_days']:>2}")
        
    # Specifically inspect Winter Jacket recent 30 days vs prior 30 days
    wj = df[df["product_id"] == "P005"].copy()
    wj["date_dt"] = pd.to_datetime(wj["date"])
    max_d = wj["date_dt"].max()
    d14 = wj[wj["date_dt"] > (max_d - timedelta(days=14))]["units_sold"].mean()
    d30_prior = wj[(wj["date_dt"] <= (max_d - timedelta(days=14))) & 
                   (wj["date_dt"] > (max_d - timedelta(days=44)))]["units_sold"].mean()
    pct_change = ((d14 - d30_prior) / max(d30_prior, 1)) * 100
    
    print("-" * 60)
    print(f"Winter Jacket Validation Check:")
    print(f"  Last 14 Days Avg:       {d14:.2f} units/day")
    print(f"  Prior 30 Days Avg:      {d30_prior:.2f} units/day")
    print(f"  Calculated Shift:       {pct_change:.1f}% change (Threshold: > 35%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
