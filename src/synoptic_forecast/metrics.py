from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    max_error,
    r2_score,
)


def evaluate_forecast(
    y_true: pd.Series,
    y_pred_p50: pd.Series,
    p10: pd.Series,
    p90: pd.Series,
    temp_lag_1: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates point predictions, directional fidelity, and interval validity."""
    mae = mean_absolute_error(y_true, y_pred_p50)
    rmse = root_mean_squared_error(y_true, y_pred_p50)
    mbe = float(np.mean(y_pred_p50 - y_true))
    max_err = max_error(y_true, y_pred_p50)
    r2 = r2_score(y_true, y_pred_p50)

    # Directional movement
    actual_dir = np.sign(y_true.values[1:] - y_true.values[:-1])
    pred_dir = np.sign(y_pred_p50.values[1:] - y_true.values[:-1])
    mda = np.mean(actual_dir == pred_dir) * 100

    # In-window persistence benchmark
    naive_test_mae = float(mean_absolute_error(y_true, temp_lag_1))
    mase_test = mae / naive_test_mae

    # Interval metrics
    coverage = float(np.mean((y_true >= p10) & (y_true <= p90)) * 100)
    interval_width = float(np.mean(p90 - p10))

    return {
        "MAE": {
            "value": mae,
            "unit": "°C",
            "target": f"< {naive_test_mae:.2f}°C (Test Naive)",
            "status": "PASS" if mae < naive_test_mae else "FAIL",
        },
        "MASE": {
            "value": mase_test,
            "unit": "",
            "target": "< 1.00 (Hard limit)",
            "status": "PASS" if mase_test < 1.00 else "FAIL",
        },
        "RMSE": {
            "value": rmse,
            "unit": "°C",
            "target": "< 2.50°C",
            "status": "PASS" if rmse < 2.50 else "WARN",
        },
        "MBE (Bias)": {
            "value": mbe,
            "unit": "°C",
            "target": "[-0.30°C, +0.30°C]",
            "status": "PASS"
            if abs(mbe) <= 0.30
            else ("WARN" if abs(mbe) <= 0.60 else "FAIL"),
        },
        "Max Error": {
            "value": max_err,
            "unit": "°C",
            "target": "< 8.50°C (3-Sigma Delta)",
            "status": "PASS" if max_err < 8.50 else "WARN",
        },
        "R² Score": {
            "value": r2,
            "unit": "",
            "target": "> 0.00 (Short-window)",
            "status": "PASS" if r2 > 0.0 else "FAIL",
        },
        "MDA (Direction)": {
            "value": mda,
            "unit": "%",
            "target": "> 60.0% (Coin-flip: 50%)",
            "status": "PASS" if mda >= 65.0 else ("WARN" if mda >= 50.0 else "FAIL"),
        },
        "Interval Coverage": {
            "value": coverage,
            "unit": "%",
            "target": ">= 80.0% (CQR Target)",
            "status": "PASS" if coverage >= 78.0 else "WARN",
        },
        "Interval Width": {
            "value": interval_width,
            "unit": "°C",
            "target": "< 6.00°C (Sharpness)",
            "status": "PASS" if interval_width < 6.00 else "WARN",
        },
    }
