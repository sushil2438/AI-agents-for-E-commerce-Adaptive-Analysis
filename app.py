"""
AI Inventory Management Agent with Adaptive Demand Forecasting
Interactive Streamlit Decision-Support Dashboard.

College Project:
- Students: Sushil Kumar, Vivek Tripathi, Yashraj Singh
- Mentor: Ms. Pragya Agarwal
- Institution: GNIOT, Greater Noida
- Domain: Computer Science and Engineering (AI & ML)
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Local modules
from src.data_loader import (
    load_sales_data,
    validate_sales_data,
    aggregate_daily_sales,
    get_dataset_summary
)
from src.forecasting import (
    forecast_next_7_days,
    compute_moving_average_baseline,
    train_forecasting_model
)
from src.evaluation import (
    evaluate_forecast_chronological,
    compute_forecast_confidence
)
from src.market_shift import detect_market_shift
from src.inventory import compute_inventory_recommendation
from src.database import (
    init_db,
    get_inventory_settings,
    save_inventory_settings,
    get_all_inventory_settings,
    get_all_purchase_orders,
    DEFAULT_DB_PATH
)
from src.purchase_orders import (
    create_purchase_order_draft,
    approve_purchase_order,
    reject_purchase_order,
    export_po_to_csv_string,
    export_po_list_to_csv_string
)
from generate_data import generate_sales_data

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Inventory Management Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Premium Cream + Burgundy Design System (CSS Tokens)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* ---------------------------------------------------------
       Cream + Burgundy Design Tokens (:root)
       --------------------------------------------------------- */
    :root {
        --color-background: #FAF4EB;
        --color-background-secondary: #F0E5D8;
        --color-card: #FFFCF7;
        --color-primary: #873B46;
        --color-primary-hover: #642B35;
        --color-text: #332426;
        --color-text-secondary: #78666A;
        --color-border: #E7D8CE;
        --color-accent: #C58B91;
        --color-highlight: #F4E2E3;

        /* Compatibility Aliases */
        --bg-main: var(--color-background);
        --bg-secondary: var(--color-background-secondary);
        --bg-card: var(--color-card);
        --primary-burgundy: var(--color-primary);
        --dark-burgundy-hover: var(--color-primary-hover);
        --main-text: var(--color-text);
        --secondary-text: var(--color-text-secondary);
        --border-color: var(--color-border);
        --soft-accent: var(--color-accent);
        --light-burgundy-highlight: var(--color-highlight);
    }

    /* ---------------------------------------------------------
       Global Base Canvas & Typography
       --------------------------------------------------------- */
    html, body, .stApp {
        background: linear-gradient(180deg, #FAF4EB 0%, #F5EDE2 50%, #FAF4EB 100%) !important;
        background-color: var(--color-background) !important;
        color: var(--color-text) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--color-text) !important;
        letter-spacing: -0.02em;
    }

    p, span, label {
        color: inherit;
    }

    /* ---------------------------------------------------------
       Headers & Titles
       --------------------------------------------------------- */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--color-primary) !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }

    .sub-header {
        font-size: 1.05rem;
        color: var(--color-text-secondary) !important;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }

    /* Academic Credit Box */
    .credit-box {
        background-color: var(--color-card);
        border: 1px solid var(--color-border);
        border-left: 4px solid var(--color-primary);
        padding: 0.85rem 1.15rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
        font-size: 0.9rem;
        color: var(--color-text);
        box-shadow: 0 2px 8px rgba(135, 59, 70, 0.04);
    }

    /* ---------------------------------------------------------
       Sidebar Styling (No Black, No Cyan)
       --------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--color-background-secondary) !important;
        border-right: 1px solid var(--color-border) !important;
        color: var(--color-text) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: var(--color-primary) !important;
        letter-spacing: -0.01em;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: var(--color-text) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--color-text) !important;
    }

    /* ---------------------------------------------------------
       Metric Cards & Built-in Metrics
       --------------------------------------------------------- */
    .metric-card {
        background: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 1.15rem 1rem;
        box-shadow: 0 4px 16px rgba(135, 59, 70, 0.05), 0 1px 3px rgba(135, 59, 70, 0.03);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--color-accent);
        box-shadow: 0 6px 20px rgba(135, 59, 70, 0.08);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.78rem;
        color: var(--color-text-secondary);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--color-text);
        line-height: 1.15;
    }

    div[data-testid="stMetric"] {
        background-color: var(--color-card) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 8px !important;
        padding: 0.85rem 1rem !important;
        box-shadow: 0 2px 6px rgba(135, 59, 70, 0.03) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--color-accent) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--color-text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.03em !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--color-text) !important;
        font-weight: 800 !important;
    }

    /* ---------------------------------------------------------
       Badges & Risk Indicators
       --------------------------------------------------------- */
    .badge-high {
        background-color: var(--color-highlight);
        color: var(--color-primary);
        border: 1px solid var(--color-accent);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FDF3E8;
        color: #8C5310;
        border: 1px solid #E5D5C0;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #EAF5EE;
        color: #276749;
        border: 1px solid #C2E2CC;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }

    /* ---------------------------------------------------------
       Navigation Tabs
       --------------------------------------------------------- */
    button[data-baseweb="tab"] {
        color: var(--color-text-secondary) !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 0.55rem 1.1rem !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: var(--color-primary-hover) !important;
        background-color: rgba(135, 59, 70, 0.05) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--color-primary) !important;
        border-bottom: 2px solid var(--color-primary) !important;
        background-color: rgba(135, 59, 70, 0.08) !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: var(--color-primary) !important;
    }

    /* ---------------------------------------------------------
       Buttons & Call-to-Actions (Accessible Light Text on Burgundy)
       --------------------------------------------------------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--color-primary) 0%, #994350 100%) !important;
        color: var(--color-card) !important;
        border: 1px solid var(--color-primary) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.15rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(135, 59, 70, 0.15) !important;
    }
    .stButton > button:hover {
        background: var(--color-primary-hover) !important;
        border-color: var(--color-primary-hover) !important;
        color: var(--color-card) !important;
        box-shadow: 0 4px 14px rgba(100, 43, 53, 0.28) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        background: #4D1F27 !important;
        transform: translateY(0) !important;
    }

    .stDownloadButton > button {
        background-color: var(--color-card) !important;
        color: var(--color-primary) !important;
        border: 1px solid var(--color-primary) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: var(--color-primary) !important;
        color: var(--color-card) !important;
        box-shadow: 0 4px 12px rgba(135, 59, 70, 0.2) !important;
    }

    button:disabled, input:disabled, select:disabled {
        background-color: var(--color-border) !important;
        color: var(--color-text-secondary) !important;
        border-color: var(--color-border) !important;
        cursor: not-allowed !important;
        opacity: 0.65 !important;
    }

    /* ---------------------------------------------------------
       Form Controls, Inputs & Steppers (No Harsh White, No Cyan)
       --------------------------------------------------------- */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="input"] > div {
        background-color: var(--color-card) !important;
        border: 1px solid var(--color-border) !important;
        color: var(--color-text) !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-baseweb="select"]:hover > div, 
    div[data-baseweb="input"]:hover > div {
        border-color: var(--color-accent) !important;
    }
    div[data-baseweb="select"]:focus-within > div, 
    div[data-baseweb="input"]:focus-within > div {
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 0 2px rgba(135, 59, 70, 0.18) !important;
    }
    input, select, textarea {
        color: var(--color-text) !important;
        background-color: var(--color-card) !important;
    }
    input::placeholder, textarea::placeholder {
        color: var(--color-text-secondary) !important;
        opacity: 0.7 !important;
    }

    /* Number input stepper buttons */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: var(--color-card) !important;
        color: var(--color-primary) !important;
        border-color: var(--color-border) !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background-color: var(--color-highlight) !important;
        color: var(--color-primary-hover) !important;
    }
    button[data-testid="stNumberInputStepDown"] svg,
    button[data-testid="stNumberInputStepUp"] svg {
        fill: var(--color-primary) !important;
    }

    /* ---------------------------------------------------------
       Dropdown Menus & Popovers (Portal Fix: Eliminates Black & Cyan)
       --------------------------------------------------------- */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: var(--color-card) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 20px rgba(135, 59, 70, 0.08) !important;
    }
    li[role="option"] {
        background-color: var(--color-card) !important;
        color: var(--color-text) !important;
        font-weight: 500 !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: var(--color-highlight) !important;
        color: var(--color-primary) !important;
    }
    li[role="option"] * {
        color: inherit !important;
    }

    /* ---------------------------------------------------------
       File Uploader
       --------------------------------------------------------- */
    div[data-testid="stFileUploader"] section {
        background-color: var(--color-card) !important;
        border: 1px dashed var(--color-border) !important;
        border-radius: 8px !important;
        color: var(--color-text) !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--color-primary) !important;
    }
    div[data-testid="stFileUploader"] small {
        color: var(--color-text-secondary) !important;
    }

    /* ---------------------------------------------------------
       Radio Buttons & Checkboxes (No Cyan Accents)
       --------------------------------------------------------- */
    div[role="radiogroup"] label {
        color: var(--color-text) !important;
    }
    div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
        color: var(--color-text) !important;
    }
    div[role="radiogroup"] div[data-checked="true"] {
        background-color: var(--color-primary) !important;
        border-color: var(--color-primary) !important;
    }
    div[role="radiogroup"] div[data-checked="false"] {
        border-color: var(--color-border) !important;
    }

    span[data-baseweb="checkbox"] span {
        border-color: var(--color-border) !important;
    }
    span[data-baseweb="checkbox"] [aria-checked="true"] {
        background-color: var(--color-primary) !important;
        border-color: var(--color-primary) !important;
    }

    /* ---------------------------------------------------------
       Alerts & Notifications (High-Contrast, Harmonious)
       --------------------------------------------------------- */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border: 1px solid var(--color-border) !important;
        background-color: var(--color-card) !important;
        box-shadow: 0 2px 6px rgba(135, 59, 70, 0.03) !important;
    }
    div[data-testid="stAlert"] * {
        color: inherit !important;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] strong,
    div[data-testid="stAlert"] div {
        color: inherit !important;
    }
    div[data-testid="stAlert"] svg {
        fill: currentColor !important;
    }

    /* Alert semantic color grading */
    div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
        background-color: var(--color-highlight) !important;
        border-color: var(--color-accent) !important;
        color: var(--color-primary) !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
        background-color: #FDF3E8 !important;
        border-color: #E5D5C0 !important;
        color: #8C5310 !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
        background-color: #EAF5EE !important;
        border-color: #C2E2CC !important;
        color: #276749 !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
        background-color: var(--color-background) !important;
        border-color: var(--color-border) !important;
        color: var(--color-text) !important;
    }

    /* ---------------------------------------------------------
       Inline Code & Preformatted Blocks (Eliminates Black Boxes)
       --------------------------------------------------------- */
    code {
        background-color: var(--color-background-secondary) !important;
        color: var(--color-primary) !important;
        border: 1px solid var(--color-border) !important;
        padding: 0.15rem 0.45rem !important;
        border-radius: 4px !important;
        font-size: 0.88em !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }
    pre {
        background-color: var(--color-card) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 8px !important;
    }
    pre code {
        border: none !important;
        background-color: transparent !important;
        color: var(--color-text) !important;
    }

    /* ---------------------------------------------------------
       Expanders & Form Containers
       --------------------------------------------------------- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--color-border) !important;
        background-color: var(--color-card) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(135, 59, 70, 0.03) !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--color-text) !important;
        font-weight: 600 !important;
        background-color: var(--color-card) !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: var(--color-primary) !important;
    }
    div[data-testid="stExpander"] summary svg {
        fill: var(--color-text-secondary) !important;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background-color: var(--color-card) !important;
        color: var(--color-text) !important;
    }

    div[data-testid="stForm"] {
        background-color: var(--color-background-secondary) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 8px !important;
    }

    /* ---------------------------------------------------------
       Dataframes & HTML Tables
       --------------------------------------------------------- */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--color-border) !important;
        border-radius: 8px !important;
        background-color: var(--color-card) !important;
    }
    table {
        background-color: var(--color-card) !important;
        color: var(--color-text) !important;
    }
    th {
        background-color: var(--color-background-secondary) !important;
        color: var(--color-primary) !important;
        border-bottom: 2px solid var(--color-border) !important;
        font-weight: 700 !important;
    }
    td {
        background-color: var(--color-card) !important;
        color: var(--color-text) !important;
        border-bottom: 1px solid var(--color-border) !important;
    }
    tr:hover td {
        background-color: var(--color-highlight) !important;
    }

    div[data-testid="stCaptionContainer"] p,
    .stCaption {
        color: var(--color-text-secondary) !important;
    }

    /* ---------------------------------------------------------
       Tooltips & Popovers
       --------------------------------------------------------- */
    div[data-testid="stTooltipContent"],
    div[data-baseweb="tooltip"] {
        background-color: var(--color-card) !important;
        color: var(--color-text) !important;
        border: 1px solid var(--color-border) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(135, 59, 70, 0.08) !important;
    }
    div[data-testid="stTooltipIcon"] svg {
        fill: var(--color-text-secondary) !important;
    }

    /* ---------------------------------------------------------
       Links & Focus Outlines
       --------------------------------------------------------- */
    a {
        color: var(--color-primary) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        transition: color 0.15s ease !important;
    }
    a:hover {
        color: var(--color-primary-hover) !important;
        text-decoration: underline !important;
    }

    button:focus-visible, 
    input:focus-visible, 
    select:focus-visible,
    [tabindex="0"]:focus-visible {
        outline: 2px solid var(--color-primary) !important;
        outline-offset: 2px !important;
    }

    /* ---------------------------------------------------------
       Clean Header & Top Accent Line
       --------------------------------------------------------- */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    header[data-testid="stHeader"] svg {
        fill: var(--color-text) !important;
    }
    div[data-testid="stDecoration"] {
        background: linear-gradient(90deg, var(--color-primary) 0%, var(--color-accent) 50%, var(--color-primary) 100%) !important;
        height: 3px !important;
    }

    /* ---------------------------------------------------------
       Scrollbars & Dividers
       --------------------------------------------------------- */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--color-background);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--color-border);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--color-accent);
    }

    hr {
        border-color: var(--color-border) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Ensure Database & Default Data Availability
# ---------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CSV_PATH = DATA_DIR / "sales_data.csv"

# Auto-initialize SQLite database
init_db(DEFAULT_DB_PATH)

# Auto-create synthetic dataset if not already present or incomplete on disk
if not DEFAULT_CSV_PATH.exists() or DEFAULT_CSV_PATH.stat().st_size < 5000:
    raw_df = generate_sales_data(num_days=540, end_date_str="2024-06-30", random_seed=42)
    raw_df.to_csv(DEFAULT_CSV_PATH, index=False)


# ---------------------------------------------------------
# Sidebar Controls & Data Loading
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Control Center")

data_source_option = st.sidebar.radio(
    "Data Source",
    ["Standard Synthetic Dataset", "Upload Custom CSV"],
    index=0
)

uploaded_file = None
if data_source_option == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (date, product_id, units_sold, ...)", type=["csv"])
    if uploaded_file is not None:
        raw_source = uploaded_file
    else:
        st.sidebar.info("Awaiting CSV upload. Using default dataset in the meantime.")
        raw_source = DEFAULT_CSV_PATH
else:
    raw_source = DEFAULT_CSV_PATH

# Button to regenerate synthetic dataset if needed
if st.sidebar.button("🔄 Regenerate Default Dataset", use_container_width=True):
    raw_df = generate_sales_data(num_days=540, end_date_str="2024-06-30", random_seed=42)
    raw_df.to_csv(DEFAULT_CSV_PATH, index=False)
    st.sidebar.success("Dataset refreshed with reproducible seed (540 days, 5 products)!")
    st.rerun()

# Load and validate dataset
try:
    df_raw = load_sales_data(raw_source)
    is_valid, validation_errors, validation_warnings = validate_sales_data(df_raw)
    
    if not is_valid:
        st.error("### ❌ Dataset Validation Failed")
        for err in validation_errors:
            st.error(f"- {err}")
        st.stop()
        
    df_clean, imputed_days = aggregate_daily_sales(df_raw)
    data_summary = get_dataset_summary(df_clean)
    
except Exception as e:
    st.error(f"Error loading sales data: {str(e)}")
    st.stop()

# Product selection
product_choices = df_clean[["product_id", "product_name"]].drop_duplicates().sort_values("product_id")
prod_options = {f"[{row['product_id']}] {row['product_name']}": row["product_id"] for _, row in product_choices.iterrows()}

selected_prod_label = st.sidebar.selectbox("Select Target Product", list(prod_options.keys()), index=0)
selected_product_id = prod_options[selected_prod_label]
selected_product_name = selected_prod_label.split("] ")[1]

# Load saved inventory settings from SQLite for selected product
saved_settings = get_inventory_settings(selected_product_id, DEFAULT_DB_PATH)
if not saved_settings:
    saved_settings = {
        "product_id": selected_product_id,
        "product_name": selected_product_name,
        "supplier_name": "Standard Supplier",
        "lead_time_days": 7,
        "current_stock": 40.0,
        "incoming_stock": 0.0,
        "safety_stock": 25.0,
        "moq": 10,
        "unit_cost": 15.0
    }

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📦 Inventory Settings: {selected_product_id}")

with st.sidebar.form(key=f"inventory_settings_form_{selected_product_id}"):
    s_curr_stock = st.number_input("Current Stock on Hand", min_value=0.0, value=float(saved_settings["current_stock"]), step=5.0)
    s_inc_stock = st.number_input("Incoming Purchase Stock", min_value=0.0, value=float(saved_settings["incoming_stock"]), step=5.0)
    s_supplier = st.text_input("Supplier Name", value=str(saved_settings["supplier_name"]))
    s_lead_time = st.number_input("Supplier Lead Time (Days)", min_value=1, value=int(saved_settings["lead_time_days"]), step=1)
    s_safety_stock = st.number_input("Safety Stock Buffer", min_value=0.0, value=float(saved_settings["safety_stock"]), step=5.0)
    s_moq = st.number_input("Minimum Order Quantity (MOQ)", min_value=1, value=int(saved_settings["moq"]), step=5)
    s_unit_cost = st.number_input("Unit Cost ($ / ₹)", min_value=0.01, value=float(saved_settings["unit_cost"]), step=1.0)
    
    save_settings_btn = st.form_submit_button("💾 Save Settings to Database", use_container_width=True)
    if save_settings_btn:
        new_settings = {
            "product_id": selected_product_id,
            "product_name": selected_product_name,
            "supplier_name": s_supplier,
            "lead_time_days": s_lead_time,
            "current_stock": s_curr_stock,
            "incoming_stock": s_inc_stock,
            "safety_stock": s_safety_stock,
            "moq": s_moq,
            "unit_cost": s_unit_cost
        }
        save_inventory_settings(new_settings, DEFAULT_DB_PATH)
        st.sidebar.success("Settings saved to SQLite!")
        st.rerun()


# ---------------------------------------------------------
# Header & Project Information Banner
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Inventory Management Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decision-Support System with Adaptive Demand Forecasting & Market-Shift Detection</div>', unsafe_allow_html=True)

with st.expander("🎓 Project Academic Information (GNIOT)", expanded=False):
    st.markdown("""
    **College Project:** AI Inventory Management Agent for E-Commerce with Adaptive Demand Forecasting  
    **Institution:** Greater Noida Institute of Technology (GNIOT), Greater Noida  
    **Department:** Computer Science and Engineering — Artificial Intelligence and Machine Learning  
    **Faculty Mentor:** Ms. Pragya Agarwal  
    **Developed by:** Sushil Kumar | Vivek Tripathi | Yashraj Singh  
    """)

if validation_warnings:
    for w in validation_warnings:
        st.warning(f"⚠️ {w}")


# ---------------------------------------------------------
# Core Analytics Computation for Selected Product
# ---------------------------------------------------------
df_prod_daily = df_clean[df_clean["product_id"] == selected_product_id].sort_values("date").copy()
avg_daily_demand = float(df_prod_daily["units_sold"].mean())

# 1. Market Shift Detection
shift_result = detect_market_shift(df_prod_daily)
is_shift = shift_result["shift_detected"]

# 2. Standard 7-Day ML Forecast
standard_forecast = forecast_next_7_days(df_prod_daily, adaptive_window=None)

# 3. Adaptive 45-Day Retraining if shift is detected
if is_shift:
    active_forecast = forecast_next_7_days(df_prod_daily, adaptive_window=45)
    active_forecast_mode = "Adaptive (Recent 45 Days)"
else:
    active_forecast = standard_forecast
    active_forecast_mode = "Standard (Full History)"

# 4. Chronological Holdout Evaluation & Baseline Comparison
eval_result = evaluate_forecast_chronological(df_prod_daily, holdout_days=14)

# 5. Forecast Confidence Assessment
confidence = compute_forecast_confidence(
    mae=eval_result["rf_mae"],
    mean_demand=eval_result["mean_demand"],
    training_days=len(df_prod_daily),
    market_shift_detected=is_shift,
    is_fallback=active_forecast.get("is_fallback", False)
)

# 6. Inventory Recommendations
inv_rec = compute_inventory_recommendation(
    product_id=selected_product_id,
    product_name=selected_product_name,
    average_daily_demand=avg_daily_demand,
    forecasted_7_day_demand=active_forecast["total_7_day_forecast"],
    current_stock=saved_settings["current_stock"],
    incoming_stock=saved_settings["incoming_stock"],
    lead_time_days=saved_settings["lead_time_days"],
    safety_stock=saved_settings["safety_stock"],
    moq=saved_settings["moq"],
    unit_cost=saved_settings["unit_cost"]
)


# ---------------------------------------------------------
# Multi-Product Catalog Aggregation (For Global KPIs)
# ---------------------------------------------------------
all_settings_list = get_all_inventory_settings(DEFAULT_DB_PATH)
settings_by_id = {s["product_id"]: s for s in all_settings_list}

catalog_records = []
total_inventory_valuation = 0.0
replenishment_needed_count = 0
high_risk_count = 0
shift_alert_count = 0

for p_id, p_df in df_clean.groupby("product_id"):
    p_name = p_df["product_name"].iloc[0]
    p_sett = settings_by_id.get(p_id, {
        "current_stock": 30.0,
        "incoming_stock": 0.0,
        "lead_time_days": 7,
        "safety_stock": 20.0,
        "moq": 10,
        "unit_cost": 15.0
    })
    
    # Check shift
    p_shift = detect_market_shift(p_df)
    if p_shift["shift_detected"]:
        shift_alert_count += 1
        p_fc = forecast_next_7_days(p_df, adaptive_window=45)
    else:
        p_fc = forecast_next_7_days(p_df, adaptive_window=None)
        
    p_eval = evaluate_forecast_chronological(p_df, holdout_days=14)
    p_conf = compute_forecast_confidence(
        mae=p_eval["rf_mae"],
        mean_demand=float(p_df["units_sold"].mean()),
        training_days=len(p_df),
        market_shift_detected=p_shift["shift_detected"]
    )
    
    p_rec = compute_inventory_recommendation(
        product_id=p_id,
        product_name=p_name,
        average_daily_demand=float(p_df["units_sold"].mean()),
        forecasted_7_day_demand=p_fc["total_7_day_forecast"],
        current_stock=p_sett["current_stock"],
        incoming_stock=p_sett["incoming_stock"],
        lead_time_days=p_sett["lead_time_days"],
        safety_stock=p_sett["safety_stock"],
        moq=p_sett["moq"],
        unit_cost=p_sett["unit_cost"]
    )
    
    val = p_sett["current_stock"] * p_sett["unit_cost"]
    total_inventory_valuation += val
    if p_rec["suggested_order_quantity"] > 0:
        replenishment_needed_count += 1
    if p_rec["inventory_risk"] == "High":
        high_risk_count += 1
        
    catalog_records.append({
        "product_id": p_id,
        "product_name": p_name,
        "7d_forecast": p_fc["total_7_day_forecast"],
        "current_stock": p_sett["current_stock"],
        "incoming_stock": p_sett["incoming_stock"],
        "reorder_point": p_rec["reorder_point"],
        "suggested_order": p_rec["suggested_order_quantity"],
        "risk": p_rec["inventory_risk"],
        "shift_detected": "⚠️ Shift Detected" if p_shift["shift_detected"] else "Normal",
        "confidence": p_conf["level"]
    })


# ---------------------------------------------------------
# Top Global Summary KPI Cards (Cream + Burgundy Grading)
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Catalog Valuation</div>
        <div class="metric-value" style="color: #873B46;">${total_inventory_valuation:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    reorder_color = "#873B46" if replenishment_needed_count > 0 else "#332426"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Reorder Required</div>
        <div class="metric-value" style="color: {reorder_color};">{replenishment_needed_count} / {len(catalog_records)}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    risk_color = "#873B46" if high_risk_count > 0 else "#276749"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">High Risk Stock</div>
        <div class="metric-value" style="color: {risk_color};">{high_risk_count} Items</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    shift_color = "#873B46" if shift_alert_count > 0 else "#276749"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Market Shifts</div>
        <div class="metric-value" style="color: {shift_color};">{shift_alert_count} Alerts</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Selected 7d Forecast</div>
        <div class="metric-value" style="color: #332426;">{active_forecast['total_7_day_forecast']:.0f} units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Suggested Order</div>
        <div class="metric-value" style="color: #873B46;">{inv_rec['suggested_order_quantity']} units</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecasting & Trends",
    "🧪 Model Evaluation & Baseline",
    "⚡ Adaptive Market-Shift Engine",
    "📋 Multi-Product Catalog Table",
    "📝 Purchase Order Draft Workflow"
])


# =========================================================
# TAB 1: Demand Forecasting & Trends
# =========================================================
with tab1:
    st.markdown(f"### Forecast & Demand Analysis: **{selected_product_name}** (`{selected_product_id}`)")
    
    # Status badges row
    risk_badge_class = f"badge-{inv_rec['inventory_risk'].lower()}"
    conf_badge_class = f"badge-{confidence['level'].lower()}"
    
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    with col_stat1:
        st.markdown(f"**Inventory Risk:** <span class='{risk_badge_class}'>{inv_rec['inventory_risk']}</span>", unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"**Forecast Confidence:** <span class='{conf_badge_class}'>{confidence['level']} ({confidence['score']}%)</span>", unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"**Active Model Mode:** `{active_forecast_mode}`")
    with col_stat4:
        st.markdown(f"**Reorder Point:** `{inv_rec['reorder_point']} units`")
    with col_stat5:
        st.markdown(f"**Suggested Order:** `{inv_rec['suggested_order_quantity']} units`")
        
    st.markdown("---")
    
    # Historical Sales + Forecast Interactive Plotly Chart (Cream + Burgundy Palette)
    recent_history = df_prod_daily.tail(60).copy()
    
    fig_fc = go.Figure()
    
    # Actual sales line
    fig_fc.add_trace(go.Scatter(
        x=pd.to_datetime(recent_history["date"]),
        y=recent_history["units_sold"],
        mode="lines+markers",
        name="Historical Actual Sales",
        line=dict(color="#78666A", width=2),
        marker=dict(size=5, color="#332426")
    ))
    
    # 7-day Moving Average Baseline on history
    ma_hist = compute_moving_average_baseline(recent_history)
    fig_fc.add_trace(go.Scatter(
        x=pd.to_datetime(recent_history["date"]),
        y=ma_hist,
        mode="lines",
        name="7-Day Moving Avg Baseline",
        line=dict(color="#C58B91", width=1.5, dash="dot")
    ))
    
    # Future 7-day forecast line (Burgundy highlight)
    fig_fc.add_trace(go.Scatter(
        x=pd.to_datetime(active_forecast["forecast_dates"]),
        y=active_forecast["daily_forecast"],
        mode="lines+markers",
        name="7-Day ML Forecast",
        line=dict(color="#873B46", width=3),
        marker=dict(size=8, symbol="diamond", color="#873B46")
    ))
    
    # Cutoff vertical line
    last_hist_date = pd.to_datetime(recent_history["date"].max())
    fig_fc.add_vline(
        x=last_hist_date,
        line_width=1.5,
        line_dash="dash",
        line_color="#873B46",
        annotation_text="Forecast Horizon Cutoff",
        annotation_position="top left",
        annotation_font=dict(color="#873B46", size=11)
    )
    
    fig_fc.update_layout(
        title=dict(
            text=f"<b>Actual Historical Demand vs 7-Day Predicted Horizon ({selected_product_name})</b>",
            font=dict(color="#332426", size=15)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFCF7",
        xaxis=dict(
            title="Date",
            title_font=dict(color="#78666A", size=12),
            tickfont=dict(color="#78666A"),
            gridcolor="#E7D8CE",
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Units Sold / Forecasted",
            title_font=dict(color="#78666A", size=12),
            tickfont=dict(color="#78666A"),
            gridcolor="#E7D8CE",
            showgrid=True,
            zeroline=False
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFCF7",
            bordercolor="#E7D8CE",
            font=dict(color="#332426")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#332426")
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Daily forecast breakdown table & feature importances
    c_fc1, c_fc2 = st.columns([1, 1])
    
    with c_fc1:
        st.markdown("#### 📅 Day-by-Day 7-Day Forecast")
        fc_df = pd.DataFrame({
            "Forecast Date": active_forecast["forecast_dates"],
            "Day of Week": [pd.to_datetime(d).strftime("%A") for d in active_forecast["forecast_dates"]],
            "Predicted Demand (Units)": active_forecast["daily_forecast"]
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
        st.info(f"**Total 7-Day Projected Demand:** **{active_forecast['total_7_day_forecast']} units**")
        
    with c_fc2:
        st.markdown("#### 🔍 Model Explainability: Feature Importances")
        feat_imp = active_forecast.get("feature_importances", {})
        if feat_imp:
            fi_df = pd.DataFrame([
                {"Feature": k, "Importance": v} for k, v in feat_imp.items()
            ]).sort_values("Importance", ascending=True)
            
            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[[0, "#F4E2E3"], [1, "#873B46"]],
                title="RandomForestRegressor Feature Contributions"
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFCF7",
                font=dict(color="#332426"),
                xaxis=dict(gridcolor="#E7D8CE", tickfont=dict(color="#78666A")),
                yaxis=dict(gridcolor="#E7D8CE", tickfont=dict(color="#78666A")),
                height=280,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            fig_fi.update_coloraxes(showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.write("Feature importances unavailable (fallback method active).")


# =========================================================
# TAB 2: Model Performance & Explainability
# =========================================================
with tab2:
    st.markdown("### Chronological Holdout Evaluation & Baseline Comparison")
    st.markdown("""
    To ensure genuine out-of-sample rigor, the latest 14 days of historical records were reserved
    exclusively as an out-of-sample holdout test set. The model was trained strictly on data
    preceding this test period with **no time-series shuffling or data leakage**.
    """)
    
    if eval_result.get("evaluation_warning"):
        st.warning(eval_result["evaluation_warning"])
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Random Forest MAE", f"{eval_result['rf_mae']:.2f}")
        m2.metric("Random Forest RMSE", f"{eval_result['rf_rmse']:.2f}")
        m3.metric("Forecast Bias", f"{eval_result['rf_bias']:+.2f}", help="Positive indicates over-forecasting, negative indicates under-forecasting.")
        m4.metric("Moving Avg Baseline MAE", f"{eval_result['ma_mae']:.2f}")
        m5.metric("RF Improvement vs Baseline", f"{eval_result['mae_improvement_pct']:+.1f}%")
        
        st.markdown("---")
        
        # Holdout Actual vs Predicted Chart (Cream + Burgundy Palette)
        eval_df = pd.DataFrame({
            "Date": eval_result["test_dates"],
            "Actual Sales": eval_result["test_actuals"],
            "Random Forest Prediction": eval_result["test_rf_preds"],
            "7-Day Moving Avg Baseline": eval_result["test_ma_preds"]
        })
        
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=eval_df["Date"], y=eval_df["Actual Sales"], mode="lines+markers", name="Actual Holdout Sales", line=dict(color="#332426", width=2.5), marker=dict(color="#332426")))
        fig_eval.add_trace(go.Scatter(x=eval_df["Date"], y=eval_df["Random Forest Prediction"], mode="lines+markers", name="Random Forest Holdout Pred", line=dict(color="#873B46", width=2, dash="dash"), marker=dict(color="#873B46")))
        fig_eval.add_trace(go.Scatter(x=eval_df["Date"], y=eval_df["7-Day Moving Avg Baseline"], mode="lines+markers", name="Moving Average Baseline", line=dict(color="#C58B91", width=1.5, dash="dot"), marker=dict(color="#C58B91")))
        
        fig_eval.update_layout(
            title=dict(
                text=f"<b>Holdout Period ({eval_result['evaluation_period'][0]} to {eval_result['evaluation_period'][1]}): Actuals vs Predictions</b>",
                font=dict(color="#332426", size=14)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFCF7",
            xaxis=dict(
                title="Holdout Date",
                gridcolor="#E7D8CE",
                tickfont=dict(color="#78666A"),
                title_font=dict(color="#78666A")
            ),
            yaxis=dict(
                title="Units",
                gridcolor="#E7D8CE",
                tickfont=dict(color="#78666A"),
                title_font=dict(color="#78666A")
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#FFFCF7",
                bordercolor="#E7D8CE",
                font=dict(color="#332426")
            ),
            legend=dict(font=dict(color="#332426")),
            height=380
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        
        st.markdown("#### 🛡️ Forecast Confidence Rating Rationale")
        st.info(f"**Confidence Rating:** **{confidence['level']}** ({confidence['score']} / 100)\n\n"
                f"**Rationale:** {confidence['explanation']}\n\n"
                "*Note: Confidence is an empirical domain heuristic based on validation error ratio, dataset volume, and market structural stability.*")


# =========================================================
# TAB 3: Adaptive Market-Shift Engine
# =========================================================
with tab3:
    st.markdown("### ⚡ Market-Shift Detection & Adaptive Retraining")
    st.markdown("""
    E-commerce demand patterns can shift abruptly due to seasonality, fashion turnover, or macro disruptions.
    The agent actively monitors:
    - **Rule A:** Forecast deviation ratio > 30% on at least 3 of the last 7 days.
    - **Rule B:** Demand percentage change between the recent 14-day average and preceding 30-day average > 35%.
    """)
    
    if is_shift:
        st.error(f"""
        ### 🚨 MARKET SHIFT DETECTED FOR {selected_product_name.upper()}
        **Detection Reason:** {shift_result['detection_reason']}  
        **Recommended Operational Action:** {shift_result['recommended_action']}
        """)
        
        col_sh1, col_sh2, col_sh3, col_sh4 = st.columns(4)
        col_sh1.metric("Recent 14-Day Avg", f"{shift_result['recent_average']:.1f} units/day")
        col_sh2.metric("Prior 30-Day Avg", f"{shift_result['previous_average']:.1f} units/day")
        col_sh3.metric("Structural Demand Shift", f"{shift_result['percentage_change']:+.1f}%", delta_color="inverse")
        col_sh4.metric("Rule A Breach Days", f"{shift_result['deviation_breach_count']} / 7 days")
        
        st.markdown("---")
        
        # Adaptive Retraining Comparison: Full History vs 45-Day Refreshed Model
        st.markdown("#### 🔄 Adaptive Re-forecasting Impact Analysis")
        st.markdown("""
        When a shift is detected, the agent automatically retrains a dedicated model on the **most recent 45 days**
        to prevent obsolete historical peaks from causing massive over-ordering or dangerous stockouts.
        """)
        
        std_qty = compute_inventory_recommendation(
            selected_product_id, selected_product_name, avg_daily_demand,
            standard_forecast["total_7_day_forecast"],
            saved_settings["current_stock"], saved_settings["incoming_stock"],
            saved_settings["lead_time_days"], saved_settings["safety_stock"],
            saved_settings["moq"], saved_settings["unit_cost"]
        )["suggested_order_quantity"]
        
        adapt_qty = inv_rec["suggested_order_quantity"]
        diff_qty = adapt_qty - std_qty
        
        comp1, comp2, comp3 = st.columns(3)
        with comp1:
            st.metric("Standard Model 7d Forecast", f"{standard_forecast['total_7_day_forecast']:.0f} units")
            st.caption(f"Unadjusted Order: **{std_qty} units**")
        with comp2:
            st.metric("Adaptive 45d Model 7d Forecast", f"{active_forecast['total_7_day_forecast']:.0f} units", delta=f"{active_forecast['total_7_day_forecast'] - standard_forecast['total_7_day_forecast']:.0f} units")
            st.caption(f"Adaptive Order: **{adapt_qty} units**")
        with comp3:
            st.metric("Order Adjustment", f"{diff_qty:+d} units", delta=f"{diff_qty} units")
            direction = "decreased" if diff_qty < 0 else ("increased" if diff_qty > 0 else "remained unchanged")
            st.caption(f"By retraining on recent data, recommended order **{direction}**, avoiding capital lockup or stockouts!")
            
        # Recent deviation details table
        with st.expander("🔎 View Rule A Day-by-Day Forecast Deviations (Last 7 Days)", expanded=True):
            dev_df = pd.DataFrame(shift_result["deviation_details"])
            st.dataframe(dev_df, use_container_width=True, hide_index=True)
            
    else:
        st.success(f"""
        ### ✅ Demand Regime Stable: {selected_product_name}
        **Status:** No market shift detected. Deviations remain within normal statistical bounds.  
        **Detection Reason:** {shift_result['detection_reason']}  
        **Recommended Action:** {shift_result['recommended_action']}
        """)
        
        col_ok1, col_ok2, col_ok3 = st.columns(3)
        col_ok1.metric("Recent 14-Day Demand Avg", f"{shift_result['recent_average']:.1f} units/day")
        col_ok2.metric("Preceding 30-Day Demand Avg", f"{shift_result['previous_average']:.1f} units/day")
        col_ok3.metric("Demand Variance", f"{shift_result['percentage_change']:+.1f}%")


# =========================================================
# TAB 4: Multi-Product Catalog Overview
# =========================================================
with tab4:
    st.markdown("### 📋 Multi-Product Inventory Health & Reorder Matrix")
    
    # Filter controls
    risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"], index=0)
    
    cat_df = pd.DataFrame(catalog_records)
    if risk_filter != "All":
        filtered_cat_df = cat_df[cat_df["risk"] == risk_filter]
    else:
        filtered_cat_df = cat_df
        
    st.dataframe(
        filtered_cat_df.rename(columns={
            "product_id": "Product ID",
            "product_name": "Product Name",
            "7d_forecast": "7-Day Forecast",
            "current_stock": "Current Stock",
            "incoming_stock": "Incoming Stock",
            "reorder_point": "Reorder Point",
            "suggested_order": "Suggested Order",
            "risk": "Inventory Risk",
            "shift_detected": "Market Shift",
            "confidence": "Confidence"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Cross-product demand comparison chart (Cream + Burgundy Palette)
    st.markdown("#### 📊 Product Demand Comparison")
    fig_prod_comp = px.bar(
        cat_df,
        x="product_name",
        y="7d_forecast",
        color="risk",
        color_discrete_map={
            "High": "#873B46",    # Primary burgundy
            "Medium": "#C58B91",  # Soft accent
            "Low": "#78666A"      # Secondary text / muted wine
        },
        title="7-Day Forecasted Demand by Product & Risk Category"
    )
    fig_prod_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFCF7",
        font=dict(color="#332426"),
        xaxis=dict(
            title="Product",
            gridcolor="#E7D8CE",
            tickfont=dict(color="#78666A"),
            title_font=dict(color="#78666A")
        ),
        yaxis=dict(
            title="Forecasted Units",
            gridcolor="#E7D8CE",
            tickfont=dict(color="#78666A"),
            title_font=dict(color="#78666A")
        ),
        legend=dict(font=dict(color="#332426")),
        height=350
    )
    st.plotly_chart(fig_prod_comp, use_container_width=True)


# =========================================================
# TAB 5: Purchase Order Draft Workflow
# =========================================================
with tab5:
    st.markdown(f"### 📝 Purchase Order Generation & Review Workflow")
    st.markdown("""
    **Core Safety Rule:** The system **never automatically submits orders to suppliers**.
    Every purchase order is initiated as a **Draft** requiring explicit human inspection, approval, or rejection.
    """)
    
    # Section A: Generate Draft for Selected Product
    st.markdown(f"#### 1. Generate Replenishment Draft for: **{selected_product_name}** (`{selected_product_id}`)")
    
    po_c1, po_c2 = st.columns([1, 1])
    with po_c1:
        st.write(f"**Supplier:** {saved_settings['supplier_name']}")
        st.write(f"**Lead Time:** {saved_settings['lead_time_days']} days (Required by approx. {(datetime.now() + timedelta(days=saved_settings['lead_time_days'])).strftime('%Y-%m-%d')})")
        st.write(f"**Current Stock:** {saved_settings['current_stock']} | **Incoming:** {saved_settings['incoming_stock']}")
    with po_c2:
        st.write(f"**Suggested Order Quantity:** **{inv_rec['suggested_order_quantity']} units**")
        st.write(f"**Unit Cost:** ${saved_settings['unit_cost']:.2f}")
        st.write(f"**Estimated Total Cost:** **${inv_rec['estimated_total_cost']:.2f}**")
        
    if st.button("📄 Generate Purchase Order Draft", key="btn_gen_po"):
        shift_str = f"Market Shift Detected ({shift_result['percentage_change']:+.1f}%)" if is_shift else "Normal"
        model_info_str = f"RandomForestRegressor ({active_forecast_mode})"
        
        success, po_record, msg = create_purchase_order_draft(
            product_id=selected_product_id,
            product_name=selected_product_name,
            supplier_name=saved_settings["supplier_name"],
            suggested_quantity=inv_rec["suggested_order_quantity"],
            lead_time_days=saved_settings["lead_time_days"],
            unit_cost=saved_settings["unit_cost"],
            reason=f"Risk: {inv_rec['inventory_risk']}; Stock {saved_settings['current_stock']} vs ROP {inv_rec['reorder_point']}",
            forecast_model_info=model_info_str,
            inventory_risk=inv_rec["inventory_risk"],
            market_shift_status=shift_str,
            db_path=DEFAULT_DB_PATH
        )
        
        if success:
            st.success(f"✅ {msg}")
            st.rerun()
        else:
            st.warning(f"ℹ️ {msg}")
            
    st.markdown("---")
    
    # Section B: Review Saved Purchase Orders
    st.markdown("#### 2. Saved Purchase Orders & Review Queue")
    
    po_status_filter = st.selectbox("Filter PO Queue by Status", ["All", "Draft — Requires Approval", "Approved", "Rejected"], index=0)
    saved_pos = get_all_purchase_orders(status_filter=po_status_filter, db_path=DEFAULT_DB_PATH)
    
    if not saved_pos:
        st.info("No purchase orders found matching the filter.")
    else:
        for po in saved_pos:
            is_draft = "Draft" in po["status"]
            is_app = po["status"] == "Approved"
            badge_bg = "#F4E2E3" if is_draft else ("#EAF5EE" if is_app else "#F9EBEA")
            badge_color = "#873B46" if is_draft else ("#276749" if is_app else "#A93226")
            badge_border = "#C58B91" if is_draft else ("#C2E2CC" if is_app else "#E6B0AA")

            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #E7D8CE; border-radius: 8px; padding: 1.1rem; margin-bottom: 0.8rem; background: #FFFCF7; box-shadow: 0 2px 8px rgba(135, 59, 70, 0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: #873B46; font-family: monospace;">{po['po_id']}</span>
                        <span style="font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 6px; 
                            background: {badge_bg};
                            color: {badge_color};
                            border: 1px solid {badge_border}; font-size: 0.8rem;">
                            {po['status']}
                        </span>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #332426;">
                        <strong>Product:</strong> {po['product_name']} ({po['product_id']}) | 
                        <strong>Supplier:</strong> {po['supplier_name']} | 
                        <strong>Quantity:</strong> <span style="color:#873B46; font-weight:700;">{po['quantity']} units</span> | 
                        <strong>Total:</strong> <span style="color:#873B46; font-weight:700;">${po['estimated_total_cost']:.2f}</span> | 
                        <strong>Required By:</strong> {po['required_by_date']}
                    </div>
                    <div style="margin-top: 0.35rem; font-size: 0.82rem; color: #78666A;">
                        <strong>Reason:</strong> {po['reason']} | 
                        <strong>Model:</strong> {po['forecast_model_info']} | 
                        <strong>Created:</strong> {po['created_at'][:19]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons for Drafts
                if po["status"] == "Draft — Requires Approval":
                    col_b1, col_b2, col_b3, _ = st.columns([1, 1, 1.5, 3])
                    with col_b1:
                        if st.button("✅ Approve", key=f"app_{po['po_id']}"):
                            approve_purchase_order(po["po_id"], notes="Approved via Dashboard", db_path=DEFAULT_DB_PATH)
                            st.success(f"PO {po['po_id']} Approved!")
                            st.rerun()
                    with col_b2:
                        if st.button("❌ Reject", key=f"rej_{po['po_id']}"):
                            reject_purchase_order(po["po_id"], notes="Rejected by manager", db_path=DEFAULT_DB_PATH)
                            st.info(f"PO {po['po_id']} Rejected.")
                            st.rerun()
                    with col_b3:
                        po_csv = export_po_to_csv_string(po)
                        st.download_button(
                            label="📥 Download Draft CSV",
                            data=po_csv,
                            file_name=f"{po['po_id']}.csv",
                            mime="text/csv",
                            key=f"dl_{po['po_id']}"
                        )
                else:
                    col_b1, _ = st.columns([1.5, 5])
                    with col_b1:
                        po_csv = export_po_to_csv_string(po)
                        st.download_button(
                            label="📥 Download PO CSV",
                            data=po_csv,
                            file_name=f"{po['po_id']}.csv",
                            mime="text/csv",
                            key=f"dl_{po['po_id']}"
                        )
                        
        # Download all POs
        st.markdown("---")
        all_pos_csv = export_po_list_to_csv_string(saved_pos)
        st.download_button(
            label="📥 Export All Purchase Orders (CSV)",
            data=all_pos_csv,
            file_name="all_purchase_orders.csv",
            mime="text/csv",
            key="dl_all_pos"
        )
