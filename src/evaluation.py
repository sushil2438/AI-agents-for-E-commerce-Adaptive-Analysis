"""
Forecast Evaluation and Explainability Module for AI Inventory Agent.
Computes chronological holdout metrics (MAE, RMSE, Forecast Bias)
and transparent heuristic-based forecast confidence ratings.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.forecasting import create_forecasting_features, FEATURE_COLUMNS


def evaluate_forecast_chronological(
    df_product: pd.DataFrame,
    holdout_days: int = 14,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform a strict chronological holdout evaluation comparing RandomForest
    against a 7-day Moving Average baseline.
    
    The latest `holdout_days` are reserved exclusively for out-of-sample testing.
    The model is fitted on data preceding the holdout cutoff.
    
    Args:
        df_product: Cleaned daily sales records for a single product.
        holdout_days: Number of recent consecutive days reserved for holdout evaluation (default 14).
        random_state: Seed for reproducible model training.
        
    Returns:
        Structured evaluation dictionary with MAE, RMSE, Bias, and Baseline comparisons.
    """
    df_sorted = df_product.sort_values(by="date").copy()
    total_len = len(df_sorted)
    
    # Check if dataset is large enough for a holdout split
    min_required_len = holdout_days + 21
    if total_len < min_required_len:
        # Graceful fallback evaluation on available data
        mean_demand = float(df_sorted["units_sold"].mean()) if not df_sorted.empty else 1.0
        return {
            "rf_mae": 0.0,
            "rf_rmse": 0.0,
            "rf_bias": 0.0,
            "ma_mae": 0.0,
            "ma_rmse": 0.0,
            "ma_bias": 0.0,
            "mae_improvement_pct": 0.0,
            "holdout_days": 0,
            "evaluation_period": ("N/A", "N/A"),
            "test_actuals": [],
            "test_rf_preds": [],
            "test_ma_preds": [],
            "test_dates": [],
            "mean_demand": mean_demand,
            "evaluation_warning": f"Insufficient data for {holdout_days}-day holdout split. Need >= {min_required_len} records."
        }
        
    # Generate leak-free features
    df_features = create_forecasting_features(df_sorted)
    valid_features = df_features.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).copy()
    
    # Split chronologically
    train_df = valid_features.iloc[:-holdout_days].copy()
    test_df = valid_features.iloc[-holdout_days:].copy()
    
    if len(train_df) < 10:
        return {
            "rf_mae": 0.0,
            "rf_rmse": 0.0,
            "rf_bias": 0.0,
            "ma_mae": 0.0,
            "ma_rmse": 0.0,
            "ma_bias": 0.0,
            "mae_improvement_pct": 0.0,
            "holdout_days": len(test_df),
            "evaluation_period": (test_df["date"].min(), test_df["date"].max()),
            "test_actuals": [],
            "test_rf_preds": [],
            "test_ma_preds": [],
            "test_dates": [],
            "mean_demand": float(test_df["units_sold"].mean()),
            "evaluation_warning": "Training set too small after lag initialization."
        }
        
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["units_sold"]
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["units_sold"].values
    
    # Fit Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = np.maximum(0, rf.predict(X_test))
    
    # Compute 7-day Moving Average baseline on test set
    # Using the rolling_mean_7 feature which is already strictly lag-1 shifted
    ma_preds = test_df["rolling_mean_7"].fillna(train_df["units_sold"].mean()).values
    
    # Metrics
    rf_mae = float(np.mean(np.abs(y_test - rf_preds)))
    rf_rmse = float(np.sqrt(np.mean((y_test - rf_preds) ** 2)))
    rf_bias = float(np.mean(rf_preds - y_test))
    
    ma_mae = float(np.mean(np.abs(y_test - ma_preds)))
    ma_rmse = float(np.sqrt(np.mean((y_test - ma_preds) ** 2)))
    ma_bias = float(np.mean(ma_preds - y_test))
    
    improvement = ((ma_mae - rf_mae) / max(ma_mae, 1e-4)) * 100.0 if ma_mae > 0 else 0.0
    
    test_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in test_df["date"].values]
    
    return {
        "rf_mae": round(rf_mae, 2),
        "rf_rmse": round(rf_rmse, 2),
        "rf_bias": round(rf_bias, 2),
        "ma_mae": round(ma_mae, 2),
        "ma_rmse": round(ma_rmse, 2),
        "ma_bias": round(ma_bias, 2),
        "mae_improvement_pct": round(improvement, 1),
        "holdout_days": holdout_days,
        "evaluation_period": (test_dates[0], test_dates[-1]),
        "test_actuals": [int(y) for y in y_test],
        "test_rf_preds": [round(float(p), 1) for p in rf_preds],
        "test_ma_preds": [round(float(p), 1) for p in ma_preds],
        "test_dates": test_dates,
        "mean_demand": round(float(np.mean(y_test)), 2),
        "evaluation_warning": None
    }


def compute_forecast_confidence(
    mae: float,
    mean_demand: float,
    training_days: int,
    market_shift_detected: bool,
    is_fallback: bool = False
) -> Dict[str, Any]:
    """
    Compute an empirical heuristic confidence level (High, Medium, Low) for demand forecasts.
    
    Factors considered:
    1. Validation Normalized Error Ratio (MAE / mean demand).
    2. Available historical data volume (training days).
    3. Structural stability (whether a market shift alert has been flagged).
    4. Model operational status (whether using ML or fallback).
    
    Returns:
        Dictionary with level ("High", "Medium", "Low"), score (0-100), and plain-language explanation.
    """
    reasons = []
    
    if is_fallback:
        return {
            "level": "Low",
            "score": 30,
            "explanation": "Confidence is Low because model training was bypassed in favor of a simple moving-average fallback due to insufficient historical records."
        }
        
    safe_mean = max(mean_demand, 1.0)
    error_ratio = mae / safe_mean
    
    # 1. Base Score from error ratio
    if error_ratio <= 0.20:
        base_score = 90
        base_level = "High"
        reasons.append(f"Strong validation accuracy (MAE {mae:.1f} is {error_ratio*100:.1f}% of avg demand)")
    elif error_ratio <= 0.40:
        base_score = 70
        base_level = "Medium"
        reasons.append(f"Moderate validation error (MAE {mae:.1f} is {error_ratio*100:.1f}% of avg demand)")
    else:
        base_score = 45
        base_level = "Low"
        reasons.append(f"Elevated prediction variance (MAE {mae:.1f} is {error_ratio*100:.1f}% of avg demand)")
        
    score = base_score
    
    # 2. Penalty for limited training history
    if training_days < 60:
        score -= 15
        reasons.append(f"Limited historical dataset ({training_days} days available)")
        
    # 3. Penalty for active market shift
    if market_shift_detected:
        score -= 25
        reasons.append("Structural market shift detected, destabilizing historical demand patterns")
        
    # Determine final level
    score = max(10, min(95, score))
    if score >= 75:
        final_level = "High"
    elif score >= 50:
        final_level = "Medium"
    else:
        final_level = "Low"
        
    explanation = f"Confidence is rated {final_level} ({score}%). " + "; ".join(reasons) + "."
    
    return {
        "level": final_level,
        "score": score,
        "explanation": explanation
    }
