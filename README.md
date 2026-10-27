# AI Inventory Management Agent for E-Commerce with Adaptive Demand Forecasting

## 🎓 Academic Project Information
- **Project Title:** AI Inventory Management Agent for E-Commerce with Adaptive Demand Forecasting
- **Academic Domain:** Computer Science and Engineering — Artificial Intelligence and Machine Learning
- **Institution:** Greater Noida Institute of Technology (GNIOT), Greater Noida
- **Faculty Mentor:** Ms. Pragya Agarwal
- **Project Developers:**
  - **Sushil Kumar**
  - **Vivek Tripathi**
  - **Yashraj Singh**

---

## 📌 Executive Summary & Problem Statement

Small and medium-sized e-commerce sellers face two persistent inventory risks:
1. **Stockouts & Lost Revenue:** Depleting stock before replenishments arrive, resulting in missed customer sales and damaged search ranking algorithms.
2. **Overstocking & Capital Lockup:** Ordering excessive quantities of slowing or obsolete products, leading to dead capital and warehouse holding fees.

Static reorder points and naive spreadsheet averages fail when consumer trends, promotions, or seasons shift. Traditional deep-learning or cloud ERP systems are prohibitively complex, non-transparent, and often operate as "black boxes" that attempt to trigger unauthorized real-world purchases.

**The Solution:** An explainable, local AI-assisted inventory decision-support system. It converts daily sales transactions into trustworthy, machine-learning-driven demand forecasts, monitors demand for structural market shifts, recalibrates replenishment quantities using an adaptive 45-day window, and generates human-audited purchase-order drafts with strict human-in-the-loop control.

---

## 🎯 Primary Objectives

1. **Answer 7 Core Retail Questions:**
   - *Which products need replenishment right now?*
   - *How much stock should be ordered?*
   - *When should replenishment be initiated (Reorder Point)?*
   - *Is demand increasing, stable, or decreasing?*
   - *Has an abrupt market shift made the standard forecast unreliable?*
   - *How confident should the inventory manager be in the current forecast?*
   - *What draft purchase order should be prepared for executive review?*
2. **Prevent Automated Submissions:** Ensure zero automatic transmissions to suppliers; every order remains a draft until a human explicitly approves or rejects it.
3. **Transparent Explainability:** Expose the exact formulaic and algorithmic rationale behind every prediction, risk score, and order recommendation.
4. **Resilience to Structural Shifts:** Detect sudden demand collapses (e.g. post-season fashion drops) and recalibrate forecasts using recent 45-day observations.

---

## 🏗️ System Architecture & Workflow

```
Historical Sales Data (CSV)
         │
         ▼
[ Data Loader & Validator ] ──> Schema validation, zero-sales imputation
         │
         ▼
[ Feature Engineering ] ──────> Strict leak-free lags (lag_1, lag_7, rolling_mean_7, calendar)
         │
         ▼
[ RandomForest Forecasting ] ─> Recursive 7-day multi-step demand predictions
         │
         ▼
[ Market-Shift Detector ] ────> Rule A: Forecast deviation > 30% on >= 3 of last 7 days
         │                      Rule B: Recent 14d vs Prior 30d demand change > 35%
         │
     ┌───┴────────────────────────┐
     │ Has Market Shift Occurred? │
     └───┬────────────────────────┘
         │
    YES  ├────────────────────────────────────────┐ NO
         ▼                                        ▼
[ Adaptive 45-Day Retraining ]       [ Standard Full-History Model ]
- Train model on latest 45 days      - Retain standard forecast
- Lower confidence rating            - High/Medium confidence
- Revise forecast & reorder qty                   │
         │                                        │
         └─────────────────┬──────────────────────┘
                           │
                           ▼
              [ Inventory Calculation Engine ]
              - Lead-Time Demand = Avg Daily Demand × Lead Time
              - Reorder Point (ROP) = Lead-Time Demand + Safety Stock
              - Target Stock = 7-Day Forecast + Safety Stock
              - Raw Order = max(0, Target - Current - Incoming)
              - MOQ Rounding = ceil(Raw / MOQ) × MOQ
              - Inventory Risk = High / Medium / Low
                           │
                           ▼
          [ Interactive Streamlit Dashboard ]
          - Real-time Plotly charts & KPIs
          - Parameter tuning & SQLite persistence
                           │
                           ▼
          [ Purchase Order Draft Workflow ]
          - Create unique PO ID (PO-YYYYMMDD-PRODUCT-XXXX)
          - Set status: "Draft — Requires Approval"
          - Human review: Explicit [Approve] or [Reject]
          - Export to audit-ready CSV
```

---

## 💻 Technology Stack

- **Language:** Python 3.10+
- **Interactive UI:** Streamlit
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** scikit-learn (`RandomForestRegressor`)
- **Data Visualization:** Plotly (`plotly.graph_objects`, `plotly.express`)
- **Persistence:** SQLite (`sqlite3` with row factory)
- **Automated Testing:** pytest

*No cloud dependencies, paid APIs, microservices, or deep learning black boxes required. Runs 100% locally on Windows.*

---

## 📂 Project Directory Structure

```
PROJECT/
├── app.py                     # Main interactive Streamlit operations dashboard
├── generate_data.py           # Reproducible synthetic dataset generator (~18 months, 5 products)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive documentation and run guide
├── .gitignore                 # Version control exclusion rules
│
├── data/
│   ├── sales_data.csv         # Historical daily sales dataset (~2,700 rows)
│   └── inventory.db           # Persistent SQLite database for settings and PO drafts
│
├── models/
│   └── README.md              # Model architecture, lag engineering, and feature descriptions
│
├── src/
│   ├── __init__.py            # Package initialization and academic metadata
│   ├── data_loader.py         # CSV loading, validation, cleaning, and daily aggregation
│   ├── forecasting.py         # RandomForest demand pipeline, recursive multi-step, MA baseline
│   ├── market_shift.py        # Rule A & Rule B market shift detection and adaptive trigger
│   ├── inventory.py           # Lead-time demand, ROP, target stock, MOQ rounding, risk scoring
│   ├── evaluation.py          # Chronological holdout evaluation (MAE, RMSE, bias) & confidence
│   ├── database.py            # SQLite schema initialization, settings CRUD, and PO storage
│   └── purchase_orders.py     # PO draft creation, unique PO IDs, local approval/rejection, CSV export
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py    # Unit tests for loading, validation, and zero-sales imputation
    ├── test_forecasting.py    # Unit tests for leak-free features, 7-day horizon, and fallbacks
    ├── test_market_shift.py   # Unit tests for Rule A, Rule B, and Winter Jacket collapse
    ├── test_inventory.py      # Unit tests for inventory formulas, MOQ rounding, and risk levels
    └── test_purchase_orders.py# Unit tests for PO creation, unique IDs, SQLite persistence, approvals
```

---

## 📊 Dataset Description

The application includes a reproducible synthetic dataset generator (`generate_data.py`) spanning approximately 18 months (540 consecutive days) across 5 distinct e-commerce products (~2,700 total rows):

| Product ID | Product Name | Demand Behavior Pattern | Base Demand | Unit Price |
| :--- | :--- | :--- | :--- | :--- |
| **P001** | Wireless Mouse | Stable demand with predictable weekend lifts | ~22 units/day | $24.99 |
| **P002** | Bluetooth Speaker | Moderate demand with weekly variance | ~14 units/day | $49.99 |
| **P003** | USB-C Cable | High-volume, steady everyday necessity | ~45 units/day | $9.99 |
| **P004** | Phone Case | Moderate demand with periodic promotional spikes | ~28 units/day | $14.99 |
| **P005** | Winter Jacket | Strong seasonal demand with a sharp post-season collapse | ~32 units/day -> ~3 units/day | $89.99 |

### Schema:
- `date`: `YYYY-MM-DD` calendar date
- `product_id`: Unique product code (`P001` - `P005`)
- `product_name`: Descriptive product title
- `units_sold`: Non-negative daily transaction volume (integer)
- `promotion`: Binary promotion flag (`0` = normal day, `1` = promotional campaign)
- `selling_price`: Retail price in USD/INR (positive float)

### 🧥 Winter Jacket Market-Shift Scenario
During the first ~510 days, `P005` (Winter Jacket) maintains high winter sales (~32 units/day). In the final 30 days, spring arrives and sales plummet to ~3.5 units/day (an ~85% drop). This legitimate data pattern triggers the market-shift detector without any hardcoded alerts.

---

## ⚙️ Installation & Setup (Windows)

Follow these step-by-step terminal commands to set up and run the system locally:

### Step 1: Open Terminal in Project Directory
Open PowerShell or Command Prompt and ensure you are in the project folder:
```powershell
cd c:\Users\DELL\OneDrive\Desktop\PROJECT
```

### Step 2: Create a Virtual Environment
Isolates dependencies from system-wide Python packages:
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
Activates the isolated environment:
```powershell
.venv\Scripts\activate
```

### Step 4: Install Required Packages
Installs Streamlit, Pandas, NumPy, scikit-learn, Plotly, and pytest:
```powershell
python -m pip install -r requirements.txt
```

### Step 5: Generate the Synthetic Dataset
Generates the 540-day sales history across all 5 products into `data/sales_data.csv`:
```powershell
python generate_data.py
```

### Step 6: Launch the Streamlit Dashboard
Launches the interactive web application in your default browser:
```powershell
python -m streamlit run app.py
```
*(The dashboard will open automatically at `http://localhost:8501`)*

### Step 7: Run the Automated Test Suite
Executes all automated unit and integration tests:
```powershell
python -m pytest -q
```

---

## 🧮 Core Algorithms & Formulas

### 1. Leak-Free Time-Series Features
To prevent future data leakage, rolling metrics and lags are shifted by at least 1 day:
- `lag_1 = units_sold.shift(1)`
- `lag_7 = units_sold.shift(7)`
- `rolling_mean_7 = units_sold.shift(1).rolling(7).mean()`
- `prev_promotion = promotion.shift(1)`
- Calendar signals: `day_of_week`, `day_of_year`, `month`, `is_weekend`, `time_index`

### 2. Multi-Step Recursive Forecasting
To project demand 7 days into the future:
- Day $t+1$ is predicted using actual observations up to day $t$.
- Day $t+2$ uses the prediction of day $t+1$ as its `lag_1`, advances the rolling window, and predicts day $t+2$.
- Continues recursively through day $t+7$.
- Output is bounded: $\hat{y}_{t} = \max(0, \text{prediction})$.

### 3. Chronological Model Evaluation
Evaluates predictions against a 14-day holdout test set:
- **MAE (Mean Absolute Error):** $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **Forecast Bias:** $\frac{1}{N} \sum (\hat{y}_i - y_i)$ *(Positive = over-forecasting, negative = under-forecasting)*
- **Baseline Comparison:** Compares Random Forest against a 7-day Moving Average baseline.

### 4. Forecast Confidence Rating
Transparent empirical heuristic combining:
- Normalized error ratio ($MAE / \mu_{\text{demand}}$): $<20\%$ = High, $20-40\%$ = Medium, $>40\%$ = Low.
- Historical data volume: Penalty if $<60$ days available.
- Structural stability: 25-point penalty if an active market shift is detected.

### 5. Market-Shift Detection Rules
- **Rule A (Forecast Deviation):**
  $$\text{deviation}_t = \frac{|y_t - \hat{y}_t|}{\max(\hat{y}_t, 1)}$$
  Triggered if $\text{deviation}_t \ge 30\%$ on at least 3 of the last 7 eligible days.
- **Rule B (Structural Demand Shift):**
  $$\text{change} = \frac{\mu_{\text{recent 14d}} - \mu_{\text{prior 30d}}}{\max(\mu_{\text{prior 30d}}, 1)} \times 100\%$$
  Triggered if $|\text{change}| \ge 35\%$.
- **Adaptive Action:** Triggers a dedicated `RandomForestRegressor` trained exclusively on the **recent 45 days**, recalibrating the forecast to current realities.

### 6. Inventory Replenishment Engine
- **Lead-Time Demand:** $\text{LTD} = \text{Average Daily Demand} \times \text{Lead Time Days}$
- **Reorder Point (ROP):** $\text{ROP} = \text{LTD} + \text{Safety Stock}$
- **Target Stock:** $\text{Target} = \text{Forecasted 7-Day Demand} + \text{Safety Stock}$
- **Raw Requirement:** $\text{Raw} = \max(0, \text{Target} - \text{Current Stock} - \text{Incoming Stock})$
- **MOQ Rounding:**
  $$\text{Suggested Order} = \begin{cases} 0 & \text{if Raw} = 0 \\ \lceil \text{Raw} / \text{MOQ} \rceil \times \text{MOQ} & \text{if Raw} > 0 \end{cases}$$
- **Inventory Risk Classification:**
  - **High:** $\text{Current Stock} \le \text{ROP}$ (Imminent stockout risk during replenishment lead time)
  - **Medium:** $\text{Current Stock} \le \text{Target Stock}$ (Buffer partially depleted)
  - **Low:** $\text{Current Stock} > \text{Target Stock}$ (Well-stocked)

---

## 🎯 Demonstration Sequence: Winter Jacket (`P005`)

To see the agent in action during a college project presentation:
1. Open the dashboard in your browser.
2. In the sidebar dropdown, select **`[P005] Winter Jacket`**.
3. Observe the **⚡ Adaptive Market-Shift Engine** tab:
   - Notice the prominent red alert: **"MARKET SHIFT DETECTED FOR WINTER JACKET"**.
   - Note the real calculated metrics: 14-day average (~3.5 units/day) vs prior 30-day average (~28 units/day) showing a **~-87% collapse**.
   - Observe the comparison: The standard full-history model would have predicted ~220 units (causing a disastrous over-order). The adaptive 45-day model slashes the 7-day forecast to ~24 units!
4. Check the **Inventory Recommendation**:
   - Because current stock on hand is 85 units, the adaptive model recognizes that current stock is already ample for post-season demand and recommends **0 units**!
5. Now switch the product selector to **`[P001] Wireless Mouse`**:
   - Notice that **NO market shift is triggered** (green status banner).
   - The system recognizes steady demand (~22 units/day), calculates a High Risk due to current stock (35) being below ROP (165), and recommends an order of **140 units**!
6. Click **"Generate Purchase Order Draft"**:
   - Notice the generated draft with unique ID `PO-YYYYMMDD-P001-XXXX` stored in SQLite with status `"Draft — Requires Approval"`.
   - Click **"✅ Approve"** or **"❌ Reject"** to observe the persistent human audit trail.

---

## 🛡️ Error Handling & Operational Safeguards

- **Schema Guards:** Missing columns, negative sales quantities, and invalid dates are rejected with explicit user explanations.
- **Zero-Sales Imputation:** Missing calendar dates are safely filled with 0 units sold rather than creating broken time indexes.
- **Data Leakage Immunity:** All rolling averages and lags strictly reference prior days ($t-1$ or earlier).
- **Zero Order Guard:** Prevents generating unnecessary purchase order drafts when stock on hand already satisfies demand.
- **No External Transmission:** Drafts remain strictly local records in SQLite; no external supplier orders can be accidentally placed.

---

## 🔮 Limitations & Future Enhancements

- **Current Limitations:**
  - SQLite database is single-node/local, ideal for small e-commerce merchants or single-store operations.
  - Forecasting model focuses on 7-day operational horizons; long-term strategic forecasts (>90 days) require macroeconomic indicators.
- **Future Enhancements:**
  - Multi-warehouse inventory routing and transfer balancing.
  - Automated supplier price break tiering and discount optimization.
  - Direct integration with open-source e-commerce APIs (WooCommerce / Shopify webhooks).
