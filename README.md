# synoptic-cqr-forecast

> Synoptic-aware temperature forecasting with distribution-free Conformalized Quantile Regression (CQR) for Glasgow, Scotland.

[![CI/CD Pipeline](https://img.shields.io/github/actions/workflow/status/vince/synoptic-cqr-forecast/forecast.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white)](https://github.com/vince/synoptic-cqr-forecast/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Interval Coverage](https://img.shields.io/badge/CQR_Coverage-81.97%25-success?style=flat-square)](https://github.com/vince/synoptic-cqr-forecast)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](http://localhost:8501)
[![Docker Ready](https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## Overview

Temperature forecasting across hyper-maritime coastal zones like Glasgow, Scotland (55.8642° N, 4.2518° W) is subject to rapid volatility. Fast-moving North Atlantic depressions, sudden cloud-shield radiation traps, and shifts between maritime and Arctic polar air masses routinely induce severe forecast errors in point-estimate regression models. Standard regression models fail to convey tail-risk probabilities, while uncalibrated quantile estimators frequently suffer from quantile crossing and under-coverage during sudden frontal passages.

**synoptic-cqr-forecast** resolves these failure modes by coupling gradient-boosted quantile regression (XGBoost) with distribution-free **Conformalized Quantile Regression (CQR)**. By extracting late-day atmospheric boundary precursors—specifically 5-hour evening barometric and temperature tendencies paired with orthogonal wind vector advection—the pipeline constructs reliable 80% prediction intervals that maintain formal finite-sample coverage guarantees without sacrificing interval sharpness.

---

## Key Features

* **Sub-Daily Synoptic Precursors:** Processes raw hourly atmospheric readings from Open-Meteo to compute 5-hour evening tendencies ($P_{23} - P_{18}$, $T_{23} - T_{18}$) and orthogonal $u/v$ wind vectors, identifying active frontal boundaries before daily time-series averaging washes them out.
* **First-Differenced Delta Modeling:** Forecasts the 24-hour step change ($\Delta T_t = T_{t} - T_{t-1}$) rather than raw temperature values, guaranteeing mathematical stationarity across annual seasonal cycles.
* **Distribution-Free Uncertainty via CQR:** Trains quantile estimators at nominal quantiles ($\alpha \in [0.05, 0.50, 0.95]$) using pinball loss, applying a non-negative conformal conformity score ($Q_{\text{conf}}$) calculated over an out-of-sample calibration split.
* **Bias-Corrected Median Shrinkage:** Applies empirical shrinkage ($\lambda = 0.65$) toward persistence with validation-bias centering, dampening noisy directional shifts while holding mean bias to $+0.08^\circ\text{C}$.
* **Automated Daily CI/CD Engine:** A scheduled GitHub Actions runner triggers nightly at 00:15 UTC, posts an operational briefing table to job summaries, and pushes updated observations and recalibrated model bundles back to the repository.

---

## Glasgow Forecast Benchmark Scorecard

Evaluated on the out-of-sample Glasgow winter benchmark window (November–December 2025, $N = 61$ test days):

| Metric | Observed | Target / Baseline | Status | Operational Interpretation |
|---|---|---|---|---|
| **MAE** | **1.49°C** | < 1.94°C (Test Naive) | **PASS** | 23.2% relative error reduction over naive persistence |
| **MASE** | **0.77** | < 1.00 (Hard ceiling) | **PASS** | Strictly outperforms random-walk persistence baseline |
| **RMSE** | **1.88°C** | < 2.50°C | **PASS** | Tightly bounded variance under North Atlantic storm tracks |
| **MBE (Bias)** | **+0.08°C** | [-0.30°C, +0.30°C] | **PASS** | Residual distribution centered near zero post-shrinkage |
| **Max Error** | **5.82°C** | < 8.50°C (3-Sigma) | **PASS** | Effective outlier suppression across deep Atlantic troughs |
| **$R^2$ Score** | **0.74** | > 0.00 | **PASS** | Strong explanatory variance across winter test window |
| **MDA (Direction)** | **85.00%** | > 60.0% (Coin-flip: 50%) | **PASS** | Accurately predicts warming vs. cooling 8.5 out of 10 days |
| **Interval Coverage** | **81.97%** | $\ge$ 80.0% (CQR Target) | **PASS** | Valid finite-sample mathematical coverage achieved |
| **Interval Width** | **5.12°C** | < 6.00°C (Sharpness) | **PASS** | High information sharpness without interval explosion |

<details>
<summary><strong>Outlier Diagnostic Details</strong></summary>

* **Worst Miss Date:** 2025-12-06
* **Observed Actual:** 1.84°C
* **Median Forecast:** 7.66°C
* **Absolute Residual:** 5.82°C
* **Inside 80% Band:** No (Extreme synoptic anomaly triggered by rapid Arctic cyclogenesis)

</details>

---

## Architecture & Tech Stack

| Layer | Component | Description |
|---|---|---|
| **Core Runtime** | Python 3.11, NumPy, Pandas | Base runtime, data manipulation, and vectorized math |
| **Modeling Engine** | XGBoost (`reg:quantileerror`) | Non-linear tree boosting optimized for pinball loss functions |
| **Uncertainty Calibration** | Split Conformal Prediction (CQR) | Distribution-free finite-sample guarantees with non-negative score guards |
| **Interactive Dashboard** | Streamlit, Plotly | Interactive operational portal with live forecast, verification, SHAP, and simulator |
| **Data Ingestion** | Open-Meteo Historical & Forecast APIs | Ingestion of hourly pressure, temperature, dew point, and wind components |
| **Packaging & Structure** | PEP 621 (`pyproject.toml`), `src/` layout | Modular, installable Python library (`synoptic_forecast`) |
| **Containerization** | Docker, Docker Compose | Multi-stage image with OpenMP runtime (`libgomp1`) for runtime parity |
| **Automation & CI/CD** | GitHub Actions | 00:15 UTC cron trigger, caching, Step Summaries, and artifact auto-sync |

---

## Project Structure

```text
synoptic-cqr-forecast/
├── .github/
│   └── workflows/
│       └── forecast.yml           # Automated daily inference and sync workflow
├── data/                          # Persistent storage: historical weather CSV archives
│   └── weather_history.csv
├── models/                        # Persistent storage: trained CQR model bundles
│   └── model_bundle.joblib
├── src/
│   └── synoptic_forecast/
│       ├── __init__.py            # Package root declaration
│       ├── data.py                # Hourly data fetch, vector decomposition, evening tendencies
│       ├── features.py            # Synoptic tabular feature engineering and lag transforms
│       ├── metrics.py             # Scorecard metrics (MAE, MASE, MDA, Coverage, Sharpness)
│       ├── model.py               # XGBoost pinball regressors, non-negative CQR calibration
│       └── visualization.py       # Time-series validation and uncertainty band visualization
├── .dockerignore
├── .gitignore
├── app.py                         # Streamlit interactive operational forecast & verification dashboard
├── docker-compose.yml             # Orchestration for containerized training, inference, and dashboard
├── Dockerfile                     # Multi-stage container build with C-runtime libraries
├── main.py                        # Benchmark runner and historical evaluation entry point
├── predict_tomorrow.py            # Operational live daily inference entry point
├── pyproject.toml                 # Standard packaging and dependency specification (PEP 621)
└── README.md
```

## 🚀 How To Use & Execute
### 1. Prerequisites
* **Python 3.11+**
* **(Optional) Docker and Docker Compose**

### 2. Installation
Clone the repository and install the required dependencies using pip:
```bash
git clone https://github.com/winengewe/synoptic-cqr-forecast.git
cd synoptic-cqr-forecast
```

Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

Install dependencies and register package in editable mode:
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to access:
* **🛰️ Operational 24h Forecast:** Live Open-Meteo ingestion, conformal $[P_{10}, P_{90}]$ prediction bands, Frost & Plunging Front alerts.
* **📊 Benchmark Scorecard & Verification:** Full metrics verification table on the test split, zoomable time series, and outlier diagnostics.
* **🧠 Synoptic Explainability & SHAP:** Feature importance ranking and SHAP atmospheric driver attributions.
* **🎛️ What-If Synoptic Simulator:** Interactive weather simulator with sliders and meteorological presets (*Polar Arctic Blast*, *Atlantic Cold Front*, etc.).

### 4. Command-Line Entry Points
Execute benchmark evaluation and train calibration bundle:
```bash
python main.py
```

Generate tomorrow's live operational forecast:
```bash
python predict_tomorrow.py
```

### 5. Docker Deployment
```bash
# Run the interactive dashboard
docker compose up dashboard

# Run daily inference
docker compose up forecast

# Run benchmark training
docker compose up train
```

## Usage & Operational Output
**Running Live Inference**
Executing predict_tomorrow.py pulls the preceding 14 days of hourly observations from Open-Meteo, computes sub-daily evening tendencies, constructs today's feature vector, applies CQR bounds, and outputs a formatted bulletin:
```bash
python predict_tomorrow.py
```

Terminal Output:
```text
==============================================================
    GLASGOW OPERATIONAL 24-HOUR TEMPERATURE FORECAST
==============================================================
Issued At           : 2026-09-15 00:15
Observation Day     : 2026-09-14 (Mean: 13.40°C)
Target Forecast Day : 2026-09-15
--------------------------------------------------------------
Expected Mean Temp  : +12.85°C
Predicted Shift (Δ) : -0.55°C (Colder)
80% Confidence Band : [10.32°C  to  15.44°C]
Interval Uncertainty: ±2.56°C (Total width: 5.12°C)
--------------------------------------------------------------
[Synoptic Context]
Evening Barometer   : -2.40 hPa/5hr (Plunging Front)
Meridional Advection: -14.20 km/h (Northerly/Arctic)
==============================================================
```


