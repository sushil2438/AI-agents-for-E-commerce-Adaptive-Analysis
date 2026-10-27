# Models Directory

This directory contains metadata and persistence utilities for demand forecasting models.

## Forecasting Architectures Used:
1. **RandomForestRegressor (scikit-learn)**
   - Ensemble decision trees with recursive multi-step lag generation.
   - Time-series aware features: day of week, day of year, month, time index, lag_1, lag_7, rolling_mean_7 (preventing data leakage), previous day promotion.
   - Non-negative prediction clipping.

2. **Adaptive Recent-Data Model**
   - Automatically triggered upon Market-Shift Detection.
   - Re-trained strictly on the most recent 45-day window to adapt swiftly to structural shifts, demand collapses, or trend changes.

3. **7-Day Moving Average Baseline**
   - Naive chronological holdout baseline against which the Machine Learning models are evaluated.
