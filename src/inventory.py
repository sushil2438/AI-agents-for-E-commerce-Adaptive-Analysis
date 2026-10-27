"""
Inventory Calculation Engine for AI Inventory Agent.
Implements lead-time demand, reorder points, target stock,
MOQ rounding, inventory risk scoring, and step-by-step explainability.
"""

from typing import Dict, Any, Union
import numpy as np


def calculate_lead_time_demand(average_daily_demand: float, lead_time_days: int) -> float:
    """
    Calculate expected sales demand during the supplier replenishment lead time.
    Formula: average_daily_demand * lead_time_days
    """
    safe_demand = max(0.0, float(average_daily_demand))
    safe_lead_time = max(0, int(lead_time_days))
    return round(safe_demand * safe_lead_time, 2)


def calculate_reorder_point(lead_time_demand: float, safety_stock: float) -> float:
    """
    Calculate the inventory threshold that triggers replenishment.
    Formula: lead_time_demand + safety_stock
    """
    return round(max(0.0, float(lead_time_demand)) + max(0.0, float(safety_stock)), 2)


def calculate_target_stock(forecasted_7_day_demand: float, safety_stock: float) -> float:
    """
    Calculate the optimal stock level to cover upcoming forecast and buffer.
    Formula: forecasted_7_day_demand + safety_stock
    """
    return round(max(0.0, float(forecasted_7_day_demand)) + max(0.0, float(safety_stock)), 2)


def calculate_raw_order_quantity(
    target_stock: float,
    current_stock: float,
    incoming_stock: float
) -> float:
    """
    Calculate the raw net stock required before applying supplier minimum order quantities.
    Formula: max(0, target_stock - current_stock - incoming_stock)
    """
    effective_inventory = max(0.0, float(current_stock)) + max(0.0, float(incoming_stock))
    raw_qty = max(0.0, float(target_stock) - effective_inventory)
    return round(raw_qty, 2)


def apply_moq_rounding(raw_quantity: float, moq: int) -> int:
    """
    Adjust the raw order requirement to satisfy supplier Minimum Order Quantity (MOQ).
    
    Rules:
    - If raw_quantity == 0: recommended quantity is 0.
    - If raw_quantity > 0: round UPWARD to the next multiple of MOQ.
    - Never recommend a negative quantity.
    
    Example: raw_qty = 23, moq = 10 -> 30 units.
    """
    if raw_quantity <= 0:
        return 0
    safe_moq = max(1, int(moq))
    batches = int(np.ceil(raw_quantity / safe_moq))
    return batches * safe_moq


def determine_inventory_risk(
    current_stock: float,
    reorder_point: float,
    target_stock: float
) -> str:
    """
    Classify inventory risk based on current stock vs reorder thresholds.
    
    Rules:
    - High: if current_stock <= reorder_point (critical risk of stockout during lead time)
    - Medium: if current_stock <= target_stock (below target buffer, replenishment advised)
    - Low: if current_stock > target_stock (sufficiently buffered)
    """
    cur = float(current_stock)
    rop = float(reorder_point)
    tgt = float(target_stock)
    
    if cur <= rop:
        return "High"
    elif cur <= tgt:
        return "Medium"
    else:
        return "Low"


def compute_inventory_recommendation(
    product_id: str,
    product_name: str,
    average_daily_demand: float,
    forecasted_7_day_demand: float,
    current_stock: float,
    incoming_stock: float,
    lead_time_days: int,
    safety_stock: float,
    moq: int,
    unit_cost: float = 0.0
) -> Dict[str, Any]:
    """
    Execute complete end-to-end inventory replenishment calculation with step-by-step explanation.
    
    Returns:
        Structured dictionary containing all intermediate parameters, recommended order,
        risk level, estimated expenditure, and step-by-step derivation text.
    """
    # Validation & coercion
    avg_demand = max(0.0, float(average_daily_demand))
    forecast_7d = max(0.0, float(forecasted_7_day_demand))
    curr_stock = max(0.0, float(current_stock))
    inc_stock = max(0.0, float(incoming_stock))
    lt_days = max(1, int(lead_time_days))
    safe_stock = max(0.0, float(safety_stock))
    safe_moq = max(1, int(moq))
    cost = max(0.0, float(unit_cost))
    
    # Mathematical steps
    lt_demand = calculate_lead_time_demand(avg_demand, lt_days)
    rop = calculate_reorder_point(lt_demand, safe_stock)
    tgt_stock = calculate_target_stock(forecast_7d, safe_stock)
    raw_qty = calculate_raw_order_quantity(tgt_stock, curr_stock, inc_stock)
    suggested_qty = apply_moq_rounding(raw_qty, safe_moq)
    risk_level = determine_inventory_risk(curr_stock, rop, tgt_stock)
    
    estimated_total_cost = round(suggested_qty * cost, 2)
    
    # Step-by-step explainability rationale
    steps = [
        f"1. Average Daily Demand = {avg_demand:.1f} units/day over recent history.",
        f"2. Supplier Lead Time = {lt_days} days -> Lead-Time Demand = {avg_demand:.1f} × {lt_days} = {lt_demand:.1f} units.",
        f"3. Safety Stock Buffer = {safe_stock:.1f} units -> Reorder Point (ROP) = {lt_demand:.1f} + {safe_stock:.1f} = {rop:.1f} units.",
        f"4. 7-Day Forecast Demand = {forecast_7d:.1f} units -> Target Stock = {forecast_7d:.1f} + {safe_stock:.1f} = {tgt_stock:.1f} units.",
        f"5. Effective On-Hand = Current Stock ({curr_stock:.1f}) + Incoming Stock ({inc_stock:.1f}) = {curr_stock + inc_stock:.1f} units.",
        f"6. Net Requirement before MOQ = Target ({tgt_stock:.1f}) - Effective ({curr_stock + inc_stock:.1f}) = {raw_qty:.1f} units.",
        f"7. Supplier MOQ = {safe_moq} units -> Suggested Order Quantity = {suggested_qty} units.",
        f"8. Inventory Risk Status = {risk_level} (Current Stock {curr_stock:.1f} vs ROP {rop:.1f} and Target {tgt_stock:.1f})."
    ]
    
    explanation_text = "\n".join(steps)
    
    return {
        "product_id": product_id,
        "product_name": product_name,
        "average_daily_demand": avg_demand,
        "forecasted_7_day_demand": forecast_7d,
        "current_stock": curr_stock,
        "incoming_stock": inc_stock,
        "effective_stock": curr_stock + inc_stock,
        "lead_time_days": lt_days,
        "safety_stock": safe_stock,
        "moq": safe_moq,
        "unit_cost": cost,
        "lead_time_demand": lt_demand,
        "reorder_point": rop,
        "target_stock": tgt_stock,
        "raw_order_quantity": raw_qty,
        "suggested_order_quantity": suggested_qty,
        "inventory_risk": risk_level,
        "estimated_total_cost": estimated_total_cost,
        "explanation": explanation_text,
        "steps": steps
    }
