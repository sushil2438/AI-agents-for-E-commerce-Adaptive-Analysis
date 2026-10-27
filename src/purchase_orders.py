"""
Purchase Order Draft Generation, Local Approval Workflow, and Audit Logging.
Enforces human-in-the-loop review: POs always start as Drafts and require explicit approval.
Never transmits orders to external supplier APIs.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import io
import pandas as pd
from src.database import insert_purchase_order, update_purchase_order_status, get_purchase_order_by_id


def generate_unique_po_id(product_id: str) -> str:
    """Generate a clean, professional, collision-resistant Purchase Order ID."""
    today_str = datetime.now().strftime("%Y%m%d")
    short_hash = uuid.uuid4().hex[:5].upper()
    clean_pid = product_id.replace(" ", "").upper()
    return f"PO-{today_str}-{clean_pid}-{short_hash}"


def create_purchase_order_draft(
    product_id: str,
    product_name: str,
    supplier_name: str,
    suggested_quantity: int,
    lead_time_days: int,
    unit_cost: float,
    reason: str,
    forecast_model_info: str,
    inventory_risk: str,
    market_shift_status: str,
    db_path: Optional[Path] = None
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Generate and persist a formal Purchase Order draft.
    
    Rules:
    - If suggested_quantity <= 0: reject generation with clear notice (stock is sufficient).
    - Status is strictly set to "Draft — Requires Approval".
    - Computes required-by date as current date + supplier lead time.
    - Computes estimated total cost = suggested_quantity * unit_cost.
    - Persists record to SQLite.
    
    Returns:
        Tuple of (success_bool, po_record_dict_or_None, user_message)
    """
    if suggested_quantity <= 0:
        return (
            False,
            None,
            f"Cannot create replenishment draft: Suggested order quantity for {product_name} ({product_id}) is 0 units. Stock levels are currently sufficient."
        )
        
    po_id = generate_unique_po_id(product_id)
    now = datetime.now()
    created_at = now.isoformat()
    
    # Required by date = creation date + lead time days
    required_by_dt = (now + timedelta(days=max(1, lead_time_days))).date()
    required_by_str = required_by_dt.strftime("%Y-%m-%d")
    
    est_total_cost = round(suggested_quantity * unit_cost, 2)
    
    po_record = {
        "po_id": po_id,
        "created_at": created_at,
        "product_id": product_id,
        "product_name": product_name,
        "supplier_name": supplier_name,
        "quantity": int(suggested_quantity),
        "required_by_date": required_by_str,
        "unit_cost": float(unit_cost),
        "estimated_total_cost": est_total_cost,
        "reason": reason,
        "forecast_model_info": forecast_model_info,
        "inventory_risk": inventory_risk,
        "market_shift_status": market_shift_status,
        "status": "Draft — Requires Approval",
        "decision_timestamp": None,
        "decision_notes": None
    }
    
    try:
        insert_purchase_order(po_record, db_path=db_path)
        msg = f"Purchase Order draft {po_id} generated successfully. Status: 'Draft — Requires Approval'."
        return True, po_record, msg
    except Exception as e:
        return False, None, f"Database error while saving Purchase Order draft: {str(e)}"


def approve_purchase_order(
    po_id: str,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """
    Record explicit human approval for a purchase order draft.
    
    Notice: Operates as an internal decision record.
    Never transmits orders to external supplier APIs.
    """
    existing = get_purchase_order_by_id(po_id, db_path=db_path)
    if not existing:
        return False, f"Purchase order {po_id} not found."
        
    if existing["status"] == "Approved":
        return True, f"Purchase order {po_id} is already approved."
        
    default_note = notes if notes else "Approved by inventory manager after forecast review."
    success = update_purchase_order_status(po_id, "Approved", default_note, db_path=db_path)
    if success:
        return True, f"Purchase Order {po_id} has been Approved locally. (Audit record updated; no external API transmission)."
    return False, f"Failed to update status for {po_id}."


def reject_purchase_order(
    po_id: str,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Tuple[bool, str]:
    """Record explicit human rejection of a purchase order draft."""
    existing = get_purchase_order_by_id(po_id, db_path=db_path)
    if not existing:
        return False, f"Purchase order {po_id} not found."
        
    default_note = notes if notes else "Rejected during review."
    success = update_purchase_order_status(po_id, "Rejected", default_note, db_path=db_path)
    if success:
        return True, f"Purchase Order {po_id} has been Rejected. (Audit record updated)."
    return False, f"Failed to update status for {po_id}."


def export_po_to_csv_string(po_record: Dict[str, Any]) -> str:
    """Format a single purchase order record into clean CSV format."""
    df = pd.DataFrame([po_record])
    return df.to_csv(index=False)


def export_po_list_to_csv_string(po_list: list) -> str:
    """Format a list of purchase order records into clean CSV format."""
    if not po_list:
        return "No records found"
    df = pd.DataFrame(po_list)
    return df.to_csv(index=False)
