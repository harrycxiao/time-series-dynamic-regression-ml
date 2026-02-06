# Dynamic Regression (ARIMA + Residual ML) for Stock Return Forecasting

## Overview
This project explores a dynamic regression approach for short-horizon stock return forecasting.

The workflow combines:
1) ARIMA modeling to capture the core time-series structure
2) Residual modeling using technical indicators (e.g., RSI, MACD)
3) Machine learning models (Gradient Boosting / Random Forest) to learn nonlinear relationships in ARIMA residuals

The goal is to test whether ARIMA + residual-based ML can improve predictive performance compared to ARIMA alone.

---

## Data
- Price data downloaded from Yahoo Finance via `yfinance`
- Derived features include common technical indicators (RSI, MACD)
- Target: return (or next-period price change) over the chosen horizon

---

## Method
### 1. Baseline time-series model
- Fit an ARIMA model on the target series
- Compute residuals to measure unexplained variation

### 2. Feature engineering on residuals
- Compute indicators such as RSI and MACD
- Align indicators with residuals for supervised learning

### 3. Residual prediction with ML
- Train ML models (e.g., Gradient Boosting, Random Forest) on engineered features
- Compare predicted residuals vs actual residuals

### 4. Evaluation
- Compare error metrics (e.g., MSE) across:
  - ARIMA baseline
  - ARIMA + residual ML

---

## Results (high-level)
The residual-ML approach can improve fit in certain regimes, but performance depends on:
- feature quality
- horizon choice
- market regime (volatility)

See the notebook for plots, diagnostics, and model comparisons.

---

## Files
- `ARIMA_Graphs.ipynb` – main analysis notebook (plots + results)
- `report.pdf` – supporting writeup/notes
- `requirements.txt` – Python dependencies

---

## How to Run
1) Install dependencies:
```
pip install -r requirements.txt
```
2) Open the notebook:
```
jupyter notebook Time-Series-Dynamic-Regression.ipynb
```
---

## Future Improvements
- Add walk-forward validation (time-series CV)
- Include more assets and time periods
- Add stronger baselines (e.g., GARCH, simple momentum)
- Report metrics across multiple random seeds and regimes

## Example Outputs
![ARIMA fit](images/arima_fit.png)
![Residual ML](images/residual_ml.png)
