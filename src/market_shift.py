"""
Market-Shift Detection and Adaptive Retraining Engine for AI Inventory Agent.
Detects demand regime shifts via:
- Rule A: Forecast Deviation Threshold Breaches (>= 30% deviation on >= 3 of last 7 days)
- Rule B: Structural Demand Disruption (Recent 14-day vs Prior 30-day shift > 35%)
Triggers automatic 45-day adaptive model retraining upon detection.
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta
import numpy as np
import pandas as pd


def detect_market_shift(
    df_product: pd.DataFrame,
    deviation_threshold: float = 0.30,
    deviation_min_breaches: int = 3,
    demand_change_threshold: float = 35.0
) -> Dict[str, Any]:
    """
    Evaluate transparent, data-driven rules to detect significant market shifts.
    
    Args:
        df_product: Daily sales records for a single product, sorted by date.
        deviation_threshold: Ratio threshold for prediction deviation (default 0.30 = 30%).
        deviation_min_breaches: Minimum number of breach days in the last 7 days (default 3).
        demand_change_threshold: Absolute percentage change threshold between recent 14d and prior 30d (default 35.0%).
        
    Returns:
        Structured dictionary detailing shift status, triggering rules, and metrics.
    """
    df = df_product.sort_values(by="date").copy()
    df.reset_index(drop=True, inplace=True)
    total_days = len(df)
    
    prod_name = str(df["product_name"].iloc[0]) if "product_name" in df.columns and not df.empty else "Product"
    prod_id = str(df["product_id"].iloc[0]) if "product_id" in df.columns and not df.empty else "P000"
    
    # Check for sufficient history: Rule B requires at least 14 + 30 = 44 days
    if total_days < 44:
        return {
            "product_id": prod_id,
            "product_name": prod_name,
            "shift_detected": False,
            "insufficient_data": True,
            "detection_reason": f"Insufficient historical records ({total_days} days available, minimum 44 days required for structural shift detection).",
            "rule_a_triggered": False,
            "rule_b_triggered": False,
            "deviation_breach_count": 0,
            "deviation_details": [],
            "recent_average": round(float(df["units_sold"].mean()), 2) if not df.empty else 0.0,
            "previous_average": 0.0,
            "percentage_change": 0.0,
            "affected_dates": [],
            "recommended_action": "Collect more daily sales data before attempting automated market shift detection."
        }
        
    # --- RULE B: Recent Demand Change ---
    # recent_average: last 14 days
    recent_14 = df.iloc[-14:]["units_sold"].values
    recent_avg = float(np.mean(recent_14))
    
    # previous_average: 30 days immediately preceding those 14 days (days -44 to -14)
    prior_30 = df.iloc[-44:-14]["units_sold"].values
    prior_avg = float(np.mean(prior_30))
    
    safe_prior = max(prior_avg, 1.0)
    pct_change = ((recent_avg - prior_avg) / safe_prior) * 100.0
    
    rule_b_triggered = abs(pct_change) >= demand_change_threshold
    
    # --- RULE A: Forecast Deviation ---
    # Evaluate the last 7 eligible days using a leak-free 7-day moving average baseline
    # constructed strictly from preceding observations
    last_7_days_df = df.iloc[-7:].copy()
    deviation_details: List[Dict[str, Any]] = []
    breach_dates: List[str] = []
    breach_count = 0
    
    for idx in last_7_days_df.index:
        actual_val = float(df.loc[idx, "units_sold"])
        date_str = str(df.loc[idx, "date"])
        
        # Preceding 7 days baseline (leak-free)
        preceding_start = max(0, idx - 7)
        preceding_window = df.loc[preceding_start : idx - 1, "units_sold"].values
        predicted_val = float(np.mean(preceding_window)) if len(preceding_window) > 0 else float(prior_avg)
        
        safe_pred = max(predicted_val, 1.0)
        deviation = abs(actual_val - predicted_val) / safe_pred
        deviation_pct = deviation * 100.0
        
        is_breach = deviation >= deviation_threshold
        if is_breach:
            breach_count += 1
            breach_dates.append(date_str)
            
        deviation_details.append({
            "date": date_str,
            "actual_sales": int(actual_val),
            "predicted_sales": round(predicted_val, 1),
            "deviation_pct": round(deviation_pct, 1),
            "is_breach": is_breach
        })
        
    rule_a_triggered = breach_count >= deviation_min_breaches
    
    # Combined Decision
    shift_detected = rule_a_triggered or rule_b_triggered
    
    # Formulate clear explainable reason
    reasons = []
    if rule_a_triggered:
        reasons.append(f"Rule A triggered: {breach_count}/7 recent days exceeded {deviation_threshold*100:.0f}% forecast deviation")
    if rule_b_triggered:
        direction = "dropped" if pct_change < 0 else "surged"
        reasons.append(f"Rule B triggered: 14-day demand {direction} by {abs(pct_change):.1f}% compared to preceding 30 days")
        
    if shift_detected:
        detection_reason = "; ".join(reasons)
        if pct_change < 0:
            recommended_action = "Demand collapse detected. Lower safety buffer and switch to 45-day adaptive model to avoid overstocking."
        else:
            recommended_action = "Demand surge detected. Increase safety buffer and review supplier lead time to prevent stockouts."
    else:
        detection_reason = "Demand pattern is stable. Deviations remain within normal statistical bounds."
        recommended_action = "Maintain standard full-history replenishment schedule."
        
    return {
        "product_id": prod_id,
        "product_name": prod_name,
        "shift_detected": shift_detected,
        "insufficient_data": False,
        "detection_reason": detection_reason,
        "rule_a_triggered": rule_a_triggered,
        "rule_b_triggered": rule_b_triggered,
        "deviation_breach_count": breach_count,
        "deviation_details": deviation_details,
        "recent_average": round(recent_avg, 2),
        "previous_average": round(prior_avg, 2),
        "percentage_change": round(pct_change, 1),
        "affected_dates": breach_dates,
        "recommended_action": recommended_action
    }
