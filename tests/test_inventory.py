"""
Tests for Inventory Calculation Engine (src/inventory.py).
"""

import pytest
from src.inventory import (
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_target_stock,
    calculate_raw_order_quantity,
    apply_moq_rounding,
    determine_inventory_risk,
    compute_inventory_recommendation
)


def test_lead_time_demand():
    """Verify lead_time_demand = avg_daily_demand * lead_time_days."""
    assert calculate_lead_time_demand(20.0, 7) == 140.0
    assert calculate_lead_time_demand(0.0, 10) == 0.0


def test_reorder_point():
    """Verify reorder_point = lead_time_demand + safety_stock."""
    assert calculate_reorder_point(140.0, 25.0) == 165.0
    assert calculate_reorder_point(50.0, 0.0) == 50.0


def test_target_stock():
    """Verify target_stock = forecasted_7_day_demand + safety_stock."""
    assert calculate_target_stock(150.0, 30.0) == 180.0


def test_raw_order_quantity():
    """Verify raw_order_quantity = max(0, target_stock - current_stock - incoming_stock)."""
    # Target = 200, current = 120, incoming = 30 -> raw = 50
    assert calculate_raw_order_quantity(200.0, 120.0, 30.0) == 50.0
    # Current + incoming exceeds target -> raw = 0
    assert calculate_raw_order_quantity(200.0, 180.0, 50.0) == 0.0


def test_moq_rounding():
    """
    Verify MOQ rounding up to next multiple:
    - raw = 0 -> 0
    - raw = 23, moq = 10 -> 30
    - raw = 20, moq = 10 -> 20
    - raw = 1, moq = 25 -> 25
    """
    assert apply_moq_rounding(0.0, 10) == 0
    assert apply_moq_rounding(23.0, 10) == 30
    assert apply_moq_rounding(20.0, 10) == 20
    assert apply_moq_rounding(1.0, 25) == 25


def test_inventory_risk_classification():
    """
    Verify risk levels:
    - High: current <= reorder_point
    - Medium: current <= target_stock (and > reorder_point)
    - Low: current > target_stock
    """
    # rop = 50, target = 100
    assert determine_inventory_risk(current_stock=45, reorder_point=50, target_stock=100) == "High"
    assert determine_inventory_risk(current_stock=50, reorder_point=50, target_stock=100) == "High"
    assert determine_inventory_risk(current_stock=75, reorder_point=50, target_stock=100) == "Medium"
    assert determine_inventory_risk(current_stock=100, reorder_point=50, target_stock=100) == "Medium"
    assert determine_inventory_risk(current_stock=120, reorder_point=50, target_stock=100) == "Low"


def test_compute_inventory_recommendation_step_by_step():
    """Verify comprehensive recommendation output and step-by-step explainability."""
    res = compute_inventory_recommendation(
        product_id="P001",
        product_name="Wireless Mouse",
        average_daily_demand=20.0,
        forecasted_7_day_demand=150.0,
        current_stock=40.0,
        incoming_stock=0.0,
        lead_time_days=7,
        safety_stock=25.0,
        moq=20,
        unit_cost=12.50
    )
    # Expected:
    # lt_demand = 20 * 7 = 140
    # rop = 140 + 25 = 165
    # target = 150 + 25 = 175
    # raw_qty = 175 - 40 - 0 = 135
    # moq = 20 -> ceil(135 / 20) * 20 = 7 * 20 = 140
    # risk = High (current 40 <= rop 165)
    # total cost = 140 * 12.50 = 1750.0
    assert res["lead_time_demand"] == 140.0
    assert res["reorder_point"] == 165.0
    assert res["target_stock"] == 175.0
    assert res["raw_order_quantity"] == 135.0
    assert res["suggested_order_quantity"] == 140
    assert res["inventory_risk"] == "High"
    assert res["estimated_total_cost"] == 1750.0
    assert len(res["steps"]) == 8
    assert "Reorder Point" in res["explanation"]
