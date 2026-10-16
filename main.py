import os
import sys

# Ensure src/ directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from synoptic_forecast.data import load_weather_data
from synoptic_forecast.features import build_features
from synoptic_forecast.model import train_quantile_regressors, reconstruct_forecasts
from synoptic_forecast.metrics import evaluate_forecast
from synoptic_forecast.visualization import plot_forecast_results


def main():
    # 1. Ingest Glasgow Historical Weather & Build Features
    raw_df = load_weather_data(
        latitude=55.8642,
        longitude=-4.2518,
        start_date="2023-01-01",
        end_date="2025-12-31",
        cache_path="data/weather_history.csv",
    )
    df = build_features(raw_df)

    # 2. Chronological 3-Way Split
    val_cutoff = "2025-09-01"
    test_cutoff = "2025-11-01"

    feature_cols = [c for c in df.columns if c not in ["temperature", "target_delta"]]
    X, y_delta, y_actual = df[feature_cols], df["target_delta"], df["temperature"]

    X_train = X.loc[X.index < val_cutoff]
    y_train_delta = y_delta.loc[y_delta.index < val_cutoff]

    X_val = X.loc[(X.index >= val_cutoff) & (X.index < test_cutoff)]
    y_val_delta = y_delta.loc[
        (y_delta.index >= val_cutoff) & (y_delta.index < test_cutoff)
    ]
    y_val_actual = y_actual.loc[
        (y_actual.index >= val_cutoff) & (y_actual.index < test_cutoff)
    ]

    X_test = X.loc[X.index >= test_cutoff]
    y_test_actual = y_actual.loc[y_actual.index >= test_cutoff]

    # 3. Fit Quantile Regressors (P05, P50 Median, P95)
    models = train_quantile_regressors(
        X_train, y_train_delta, X_val, y_val_delta, quantiles=[0.05, 0.50, 0.95]
    )

    # 4. Reconstruct Temperatures with Non-Negative CQR & Bias Centering
    forecasts, q_conf, val_bias = reconstruct_forecasts(
        models=models,
        X_test=X_test,
        test_index=y_test_actual.index,
        X_val=X_val,
        y_val_actual=y_val_actual,
        delta_shrinkage=0.65,
        coverage_target=0.80,
    )

    print(
        f"\n[Calibration] Q_conf expansion: ±{q_conf:.2f}°C | Validation bias offset: {val_bias:+.2f}°C"
    )

    # 5. Evaluate Against Operational Targets
    report = evaluate_forecast(
        y_true=y_test_actual,
        y_pred_p50=forecasts[0.50],
        p10=forecasts[0.10],
        p90=forecasts[0.90],
        temp_lag_1=X_test["temp_lag_1"],
    )

    # Display Benchmark Scorecard
    header = (
        f"{'Metric':<20} | {'Observed':<12} | {'Target / Baseline':<28} | {'Status'}"
    )
    print("\n" + "=" * len(header))
    print("           GLASGOW FORECAST BENCHMARK SCORECARD")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, data in report.items():
        val_str = (
            f"{data['value']:+.2f}{data['unit']}"
            if "Bias" in name
            else f"{data['value']:.2f}{data['unit']}"
        )
        print(f"{name:<20} | {val_str:<12} | {data['target']:<28} | [{data['status']}]")

    print("=" * len(header))

    # 6. Outlier Diagnostics
    abs_residuals = (y_test_actual - forecasts[0.50]).abs()
    worst_date = abs_residuals.idxmax()
    worst_val = abs_residuals.max()
    actual_on_worst = y_test_actual.loc[worst_date]
    pred_on_worst = forecasts[0.50].loc[worst_date]
    p10_on_worst = forecasts[0.10].loc[worst_date]
    p90_on_worst = forecasts[0.90].loc[worst_date]

    print("\n[Outlier Diagnostic]")
    print(f"Worst Miss Date  : {worst_date.strftime('%Y-%m-%d')}")
    print(f"Observed Actual  : {actual_on_worst:.2f}°C")
    print(f"Median Forecast  : {pred_on_worst:.2f}°C")
    print(f"Absolute Error   : {worst_val:.2f}°C")
    print(
        f"Inside 80% Band  : {'Yes' if p10_on_worst <= actual_on_worst <= p90_on_worst else 'No (Extreme Anomaly)'}\n"
    )

    # 7. Visualization
    y_train_actual = y_actual.loc[y_actual.index < test_cutoff]
    plot_forecast_results(
        y_train=y_train_actual,
        y_test=y_test_actual,
        p10=forecasts[0.10],
        p50=forecasts[0.50],
        p90=forecasts[0.90],
        split_cutoff=test_cutoff,
    )


if __name__ == "__main__":
    main()
