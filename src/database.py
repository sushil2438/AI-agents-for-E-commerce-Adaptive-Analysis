"""
SQLite Database Persistence Module for AI Inventory Agent.
Stores and restores inventory parameters and maintains an audit trail
of purchase-order drafts, human approvals, and rejection decisions.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import sqlite3


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inventory.db"


DEFAULT_SETTINGS_SEED = [
    {
        "product_id": "P001",
        "product_name": "Wireless Mouse",
        "supplier_name": "TechLogix Global",
        "lead_time_days": 7,
        "current_stock": 35.0,
        "incoming_stock": 0.0,
        "safety_stock": 25.0,
        "moq": 20,
        "unit_cost": 12.50
    },
    {
        "product_id": "P002",
        "product_name": "Bluetooth Speaker",
        "supplier_name": "AcousticWare Ltd",
        "lead_time_days": 10,
        "current_stock": 22.0,
        "incoming_stock": 10.0,
        "safety_stock": 20.0,
        "moq": 10,
        "unit_cost": 24.00
    },
    {
        "product_id": "P003",
        "product_name": "USB-C Cable",
        "supplier_name": "ConnectFast Components",
        "lead_time_days": 5,
        "current_stock": 60.0,
        "incoming_stock": 0.0,
        "safety_stock": 40.0,
        "moq": 50,
        "unit_cost": 3.20
    },
    {
        "product_id": "P004",
        "product_name": "Phone Case",
        "supplier_name": "ShieldGuard Accessories",
        "lead_time_days": 8,
        "current_stock": 30.0,
        "incoming_stock": 20.0,
        "safety_stock": 30.0,
        "moq": 25,
        "unit_cost": 4.50
    },
    {
        "product_id": "P005",
        "product_name": "Winter Jacket",
        "supplier_name": "Alpine Apparel Co",
        "lead_time_days": 14,
        "current_stock": 85.0,
        "incoming_stock": 0.0,
        "safety_stock": 30.0,
        "moq": 10,
        "unit_cost": 42.00
    }
]


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection with row factory enabled."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables for inventory settings and purchase orders."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Table 1: Inventory Settings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory_settings (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            supplier_name TEXT NOT NULL,
            lead_time_days INTEGER NOT NULL,
            current_stock REAL NOT NULL,
            incoming_stock REAL NOT NULL,
            safety_stock REAL NOT NULL,
            moq INTEGER NOT NULL,
            unit_cost REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Table 2: Purchase Order Drafts & Review History
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS purchase_orders (
            po_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            product_id TEXT NOT NULL,
            product_name TEXT NOT NULL,
            supplier_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            required_by_date TEXT NOT NULL,
            unit_cost REAL NOT NULL,
            estimated_total_cost REAL NOT NULL,
            reason TEXT NOT NULL,
            forecast_model_info TEXT NOT NULL,
            inventory_risk TEXT NOT NULL,
            market_shift_status TEXT NOT NULL,
            status TEXT NOT NULL,
            decision_timestamp TEXT,
            decision_notes TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    
    # Seed default inventory parameters if empty
    seed_default_inventory_settings(db_path)


def seed_default_inventory_settings(db_path: Optional[Path] = None) -> None:
    """Seed initial realistic settings for demo products if table is currently empty."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) as cnt FROM inventory_settings")
    count = cursor.fetchone()["cnt"]
    
    if count == 0:
        now_str = datetime.now().isoformat()
        for s in DEFAULT_SETTINGS_SEED:
            cursor.execute("""
                INSERT INTO inventory_settings (
                    product_id, product_name, supplier_name, lead_time_days,
                    current_stock, incoming_stock, safety_stock, moq, unit_cost, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s["product_id"], s["product_name"], s["supplier_name"], s["lead_time_days"],
                s["current_stock"], s["incoming_stock"], s["safety_stock"], s["moq"], s["unit_cost"], now_str
            ))
        conn.commit()
        
    conn.close()


def get_inventory_settings(product_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve saved inventory settings for a specific product ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inventory_settings WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_inventory_settings(settings: Dict[str, Any], db_path: Optional[Path] = None) -> bool:
    """Insert or update inventory settings for a product."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO inventory_settings (
            product_id, product_name, supplier_name, lead_time_days,
            current_stock, incoming_stock, safety_stock, moq, unit_cost, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(product_id) DO UPDATE SET
            product_name = excluded.product_name,
            supplier_name = excluded.supplier_name,
            lead_time_days = excluded.lead_time_days,
            current_stock = excluded.current_stock,
            incoming_stock = excluded.incoming_stock,
            safety_stock = excluded.safety_stock,
            moq = excluded.moq,
            unit_cost = excluded.unit_cost,
            updated_at = excluded.updated_at
    """, (
        settings["product_id"],
        settings["product_name"],
        settings["supplier_name"],
        int(settings["lead_time_days"]),
        float(settings["current_stock"]),
        float(settings["incoming_stock"]),
        float(settings["safety_stock"]),
        int(settings["moq"]),
        float(settings["unit_cost"]),
        now_str
    ))
    conn.commit()
    conn.close()
    return True


def get_all_inventory_settings(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all saved product inventory settings."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inventory_settings ORDER BY product_id")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_purchase_order(po: Dict[str, Any], db_path: Optional[Path] = None) -> bool:
    """Save a newly generated purchase-order draft into SQLite."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO purchase_orders (
            po_id, created_at, product_id, product_name, supplier_name,
            quantity, required_by_date, unit_cost, estimated_total_cost,
            reason, forecast_model_info, inventory_risk, market_shift_status,
            status, decision_timestamp, decision_notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        po["po_id"],
        po["created_at"],
        po["product_id"],
        po["product_name"],
        po["supplier_name"],
        int(po["quantity"]),
        po["required_by_date"],
        float(po["unit_cost"]),
        float(po["estimated_total_cost"]),
        po["reason"],
        po["forecast_model_info"],
        po["inventory_risk"],
        po["market_shift_status"],
        po["status"],
        po.get("decision_timestamp"),
        po.get("decision_notes")
    ))
    conn.commit()
    conn.close()
    return True


def update_purchase_order_status(
    po_id: str,
    status: str,
    notes: Optional[str] = None,
    db_path: Optional[Path] = None
) -> bool:
    """Update approval or rejection status of a purchase order draft."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().isoformat()
    
    cursor.execute("""
        UPDATE purchase_orders
        SET status = ?, decision_timestamp = ?, decision_notes = ?
        WHERE po_id = ?
    """, (status, now_str, notes, po_id))
    
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def get_all_purchase_orders(
    status_filter: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieve purchase orders sorted by creation date descending."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if status_filter and status_filter != "All":
        cursor.execute("SELECT * FROM purchase_orders WHERE status = ? ORDER BY created_at DESC", (status_filter,))
    else:
        cursor.execute("SELECT * FROM purchase_orders ORDER BY created_at DESC")
        
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_purchase_order_by_id(po_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Fetch single purchase order record by its unique PO ID."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM purchase_orders WHERE po_id = ?", (po_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
