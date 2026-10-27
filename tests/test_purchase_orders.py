"""
Tests for Purchase Order Drafts, SQLite Persistence, and Human-in-the-Loop Review (src/purchase_orders.py).
"""

import pytest
import sqlite3
from pathlib import Path
from src.database import init_db, get_connection, get_purchase_order_by_id
from src.purchase_orders import (
    generate_unique_po_id,
    create_purchase_order_draft,
    approve_purchase_order,
    reject_purchase_order,
    export_po_to_csv_string
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create an isolated test database in a temporary directory."""
    db_file = tmp_path / "test_inventory.db"
    init_db(db_file)
    return db_file


def test_generate_unique_po_id():
    """Verify unique PO ID generation format."""
    po_id_1 = generate_unique_po_id("P001")
    po_id_2 = generate_unique_po_id("P001")
    assert po_id_1.startswith("PO-")
    assert "P001" in po_id_1
    assert po_id_1 != po_id_2  # Must be unique


def test_create_purchase_order_draft_success(temp_db_path):
    """Verify purchase order draft creation and SQLite persistence."""
    success, po_record, msg = create_purchase_order_draft(
        product_id="P001",
        product_name="Wireless Mouse",
        supplier_name="TechLogix Global",
        suggested_quantity=60,
        lead_time_days=7,
        unit_cost=12.50,
        reason="Stock is below ROP",
        forecast_model_info="RandomForestRegressor",
        inventory_risk="High",
        market_shift_status="Normal",
        db_path=temp_db_path
    )
    assert success is True
    assert po_record is not None
    assert po_record["quantity"] == 60
    assert po_record["estimated_total_cost"] == 750.0  # 60 * 12.50
    assert po_record["status"] == "Draft — Requires Approval"
    
    # Verify persisted in SQLite
    saved = get_purchase_order_by_id(po_record["po_id"], db_path=temp_db_path)
    assert saved is not None
    assert saved["quantity"] == 60
    assert saved["status"] == "Draft — Requires Approval"


def test_zero_quantity_cannot_create_draft(temp_db_path):
    """Verify that zero order recommendations do not generate an unnecessary draft."""
    success, po_record, msg = create_purchase_order_draft(
        product_id="P001",
        product_name="Wireless Mouse",
        supplier_name="TechLogix Global",
        suggested_quantity=0,
        lead_time_days=7,
        unit_cost=12.50,
        reason="No order needed",
        forecast_model_info="RandomForestRegressor",
        inventory_risk="Low",
        market_shift_status="Normal",
        db_path=temp_db_path
    )
    assert success is False
    assert po_record is None
    assert "0 units" in msg


def test_approve_purchase_order_status_update(temp_db_path):
    """Verify that human approval updates draft status to Approved with timestamp."""
    success, po_record, _ = create_purchase_order_draft(
        product_id="P002",
        product_name="Bluetooth Speaker",
        supplier_name="AcousticWare Ltd",
        suggested_quantity=20,
        lead_time_days=10,
        unit_cost=24.0,
        reason="Medium risk buffer replenish",
        forecast_model_info="RandomForestRegressor",
        inventory_risk="Medium",
        market_shift_status="Normal",
        db_path=temp_db_path
    )
    po_id = po_record["po_id"]
    
    approved, app_msg = approve_purchase_order(po_id, notes="Approved by Manager", db_path=temp_db_path)
    assert approved is True
    
    updated = get_purchase_order_by_id(po_id, db_path=temp_db_path)
    assert updated["status"] == "Approved"
    assert updated["decision_timestamp"] is not None
    assert updated["decision_notes"] == "Approved by Manager"


def test_reject_purchase_order_status_update(temp_db_path):
    """Verify that rejection updates status to Rejected with audit trail preserved."""
    success, po_record, _ = create_purchase_order_draft(
        product_id="P003",
        product_name="USB-C Cable",
        supplier_name="ConnectFast Components",
        suggested_quantity=50,
        lead_time_days=5,
        unit_cost=3.20,
        reason="Test PO",
        forecast_model_info="RandomForestRegressor",
        inventory_risk="High",
        market_shift_status="Normal",
        db_path=temp_db_path
    )
    po_id = po_record["po_id"]
    
    rejected, rej_msg = reject_purchase_order(po_id, notes="Supplier lead time too long", db_path=temp_db_path)
    assert rejected is True
    
    updated = get_purchase_order_by_id(po_id, db_path=temp_db_path)
    assert updated["status"] == "Rejected"
    assert updated["decision_notes"] == "Supplier lead time too long"


def test_export_po_to_csv_string():
    """Verify CSV string export of purchase order record."""
    po_dict = {
        "po_id": "PO-TEST-1234",
        "product_id": "P001",
        "product_name": "Test Item",
        "quantity": 30,
        "estimated_total_cost": 300.0,
        "status": "Draft — Requires Approval"
    }
    csv_str = export_po_to_csv_string(po_dict)
    assert "PO-TEST-1234" in csv_str
    assert "Test Item" in csv_str
    assert "300.0" in csv_str
