"""
Demand Forecasting Engine for AI Inventory Agent.
Implements leak-free feature engineering, chronological RandomForest training,
recursive multi-step forecasting, moving-average baselines, and adaptive 45-day retraining.
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "time_index",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "prev_promotion"
]


def create_forecasting_features(df_product: pd.DataFrame) -> pd.DataFrame:
    """
    Construct time-series lag and calendar features strictly without data leakage.
    
    All rolling statistics and lag variables are shifted by at least 1 day
    so that only past information is available to predict the current day.
    
    Args:
        df_product: Daily sales DataFrame for a single product, sorted by date.
        
    Returns:
        DataFrame enriched with feature columns.
    """
    df = df_product.copy()
    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Calendar features
    df["date_dt"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["day_of_year"] = df["date_dt"].dt.dayofyear
    df["month"] = df["date_dt"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Sequential time index
    min_date = df["date_dt"].min()
    df["time_index"] = (df["date_dt"] - min_date).dt.days
    
    # Past sales lags (STRICTLY SHIFTED - NO LEAKAGE)
    df["lag_1"] = df["units_sold"].shift(1)
    df["lag_7"] = df["units_sold"].shift(7)
    
    # Rolling 7-day average of PAST sales (shifted by 1 so current day actual is NOT included)
    df["rolling_mean_7"] = df["units_sold"].shift(1).rolling(window=7, min_periods=1).mean()
    
    # Previous day promotion status
    if "promotion" in df.columns:
        df["prev_promotion"] = df["promotion"].shift(1).fillna(0).astype(int)
    else:
        df["prev_promotion"] = 0
        
    return df


def train_forecasting_model(
    df_product: pd.DataFrame,
    adaptive_window: Optional[int] = None,
    random_state: int = 42
) -> Tuple[Optional[RandomForestRegressor], pd.DataFrame, Dict[str, Any]]:
    """
    Train a RandomForestRegressor demand forecasting model using chronological data.
    
    Args:
        df_product: Cleaned daily sales records for a single product.
        adaptive_window: If provided (e.g. 45), trains exclusively on the most recent N days.
        random_state: Random state for deterministic model fitting.
        
    Returns:
        Tuple of (model, feature_dataframe, metadata_dict)
    """
    df_features = create_forecasting_features(df_product)
    
    # If adaptive mode is active, filter to the latest N calendar days
    if adaptive_window is not None and adaptive_window > 0:
        max_date = df_features["date_dt"].max()
        cutoff_date = max_date - timedelta(days=adaptive_window)
        df_train = df_features[df_features["date_dt"] >= cutoff_date].copy()
        method_label = f"Adaptive (Recent {adaptive_window} Days)"
    else:
        df_train = df_features.copy()
        method_label = "Standard (Full History)"
        
    # Drop warm-up rows where lag_7 is NaN
    valid_train = df_train.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).copy()
    
    meta = {
        "method_label": method_label,
        "adaptive_window": adaptive_window,
        "total_available_rows": len(df_product),
        "training_samples": len(valid_train),
        "is_fallback": False,
        "feature_importances": {}
    }
    
    # Guardrail: check if sufficient samples exist to train an ensemble ML model
    MIN_TRAINING_SAMPLES = 14
    if len(valid_train) < MIN_TRAINING_SAMPLES:
        meta["is_fallback"] = True
        meta["fallback_reason"] = f"Insufficient training observations ({len(valid_train)} < {MIN_TRAINING_SAMPLES}). Using Moving Average fallback."
        return None, df_features, meta
        
    X_train = valid_train[FEATURE_COLUMNS]
    y_train = valid_train["units_sold"]
    
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Extract feature importances
    importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, rf.feature_importances_)}
    meta["feature_importances"] = importances
    
    return rf, df_features, meta


def forecast_next_7_days(
    df_product: pd.DataFrame,
    adaptive_window: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Generate a 7-day future demand forecast using recursive multi-step forecasting.
    
    Prevents negative predictions by bounding results at 0.
    
    Args:
        df_product: Daily sales records for a single product.
        adaptive_window: Optional window size for recent-data retraining (e.g. 45 days).
        random_state: Seed for reproducibility.
        
    Returns:
        Structured dictionary with forecast dates, predictions, and model metadata.
    """
    prod_id = str(df_product["product_id"].iloc[0])
    prod_name = str(df_product["product_name"].iloc[0])
    
    model, df_features, meta = train_forecasting_model(
        df_product,
        adaptive_window=adaptive_window,
        random_state=random_state
    )
    
    sorted_df = df_product.sort_values(by="date").copy()
    last_date = pd.to_datetime(sorted_df["date"].max())
    min_date = pd.to_datetime(sorted_df["date"].min())
    
    forecast_dates = [(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(7)]
    
    # Handle Fallback if model could not be fitted
    if model is None or meta["is_fallback"]:
        # Fallback: 7-day moving average or mean sales
        recent_sales = sorted_df["units_sold"].tail(7).values
        if len(recent_sales) > 0:
            fallback_val = float(np.mean(recent_sales))
        else:
            fallback_val = float(sorted_df["units_sold"].mean()) if not sorted_df.empty else 0.0
            
        daily_preds = [round(max(0.0, fallback_val), 1) for _ in range(7)]
        
        return {
            "product_id": prod_id,
            "product_name": prod_name,
            "forecast_dates": forecast_dates,
            "daily_forecast": daily_preds,
            "total_7_day_forecast": round(float(sum(daily_preds)), 1),
            "model_name": "Fallback (7-Day Moving Average)",
            "training_date_range": (min_date.strftime("%Y-%m-%d"), last_date.strftime("%Y-%m-%d")),
            "training_samples": meta["training_samples"],
            "training_timestamp": datetime.now().isoformat(),
            "forecast_method": meta["method_label"] + " [Fallback]",
            "feature_importances": {},
            "is_fallback": True,
            "fallback_reason": meta.get("fallback_reason", "Insufficient data")
        }
        
    # Recursive 7-day forecasting with lag maintenance
    # We maintain a simulation history of past actual units_sold + future predictions
    history_units = list(sorted_df["units_sold"].values)
    history_promos = list(sorted_df["promotion"].values) if "promotion" in sorted_df.columns else [0] * len(sorted_df)
    
    daily_preds = []
    
    for step in range(1, 8):
        future_date = last_date + timedelta(days=step)
        day_of_week = future_date.weekday()
        day_of_year = future_date.timetuple().tm_yday
        month = future_date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0
        time_index = (future_date - min_date).days
        
        # Lag 1 is the most recent observation (actual or predicted from step - 1)
        lag_1 = float(history_units[-1])
        # Lag 7 is the observation from 7 days ago
        lag_7 = float(history_units[-7]) if len(history_units) >= 7 else float(np.mean(history_units))
        # Rolling mean of the last 7 observations in our history
        rolling_mean_7 = float(np.mean(history_units[-7:]))
        # Previous day promotion
        prev_promotion = int(history_promos[-1])
        
        feat_vector = pd.DataFrame([{
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "month": month,
            "is_weekend": is_weekend,
            "time_index": time_index,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
            "prev_promotion": prev_promotion
        }])[FEATURE_COLUMNS]
        
        pred = float(model.predict(feat_vector)[0])
        # Bound at 0 (sales cannot be negative)
        bounded_pred = round(max(0.0, pred), 1)
        daily_preds.append(bounded_pred)
        
        # Append prediction to history for next recursive steps
        history_units.append(bounded_pred)
        history_promos.append(0)  # assume standard non-promo day for future horizon
        
    return {
        "product_id": prod_id,
        "product_name": prod_name,
        "forecast_dates": forecast_dates,
        "daily_forecast": daily_preds,
        "total_7_day_forecast": round(float(sum(daily_preds)), 1),
        "model_name": "RandomForestRegressor",
        "training_date_range": (
            (last_date - timedelta(days=adaptive_window)).strftime("%Y-%m-%d")
            if adaptive_window else min_date.strftime("%Y-%m-%d"),
            last_date.strftime("%Y-%m-%d")
        ),
        "training_samples": meta["training_samples"],
        "training_timestamp": datetime.now().isoformat(),
        "forecast_method": meta["method_label"],
        "feature_importances": meta["feature_importances"],
        "is_fallback": False,
        "fallback_reason": None
    }


def compute_moving_average_baseline(df_product: pd.DataFrame, window: int = 7) -> pd.Series:
    """
    Compute a leak-free 7-day moving average baseline for historical observations.
    
    Baseline prediction for day t is the average of actual sales from day (t-7) to (t-1).
    
    Args:
        df_product: Daily product sales DataFrame.
        window: Rolling window size (default 7).
        
    Returns:
        pd.Series of moving average baseline predictions aligned with df_product index.
    """
    df = df_product.sort_values(by="date").copy()
    baseline = df["units_sold"].shift(1).rolling(window=window, min_periods=1).mean().fillna(df["units_sold"].mean())
    return baseline
