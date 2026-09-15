import datetime as np_datetime
import os
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import requests

from synoptic_forecast.data import load_weather_data
from synoptic_forecast.features import build_features
from synoptic_forecast.model import train_quantile_regressors, reconstruct_forecasts

MODEL_BUNDLE_PATH = "models/model_bundle.joblib"
LATITUDE = 55.8642  # Glasgow
LONGITUDE = -4.2518


def get_or_train_pipeline() -> Tuple[dict, float, float]:
    """Loads cached models and calibration parameters or trains them from historical data."""
    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        print(f"[Model] Loaded trained bundle from '{MODEL_BUNDLE_PATH}'.")
        return bundle["models"], bundle["q_conf"], bundle["val_bias"]
    except FileNotFoundError:
        print("[Model] Bundle not found. Training on Glasgow historical archive...")

        raw_df = load_weather_data(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            start_date="2023-01-01",
            end_date="2025-12-31",
            cache_path="data/weather_history.csv",
        )
        df = build_features(raw_df)

        val_cutoff = "2025-09-01"
        test_cutoff = "2025-11-01"

        feature_cols = [
            c for c in df.columns if c not in ["temperature", "target_delta"]
        ]
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

        models = train_quantile_regressors(
            X_train, y_train_delta, X_val, y_val_delta, quantiles=[0.05, 0.50, 0.95]
        )

        _, q_conf, val_bias = reconstruct_forecasts(
            models=models,
            X_test=X_test,
            test_index=y_test_actual.index,
            X_val=X_val,
            y_val_actual=y_val_actual,
            delta_shrinkage=0.65,
            coverage_target=0.80,
        )

        os.makedirs(os.path.dirname(MODEL_BUNDLE_PATH), exist_ok=True)
        joblib.dump(
            {"models": models, "q_conf": q_conf, "val_bias": val_bias},
            MODEL_BUNDLE_PATH,
        )
        print(f"[Model] Saved pipeline bundle to '{MODEL_BUNDLE_PATH}'.")
        return models, q_conf, val_bias


def fetch_live_recent_window(days: int = 14) -> pd.DataFrame:
    """Fetches recent hourly weather through today to compute historical lags and evening tendencies."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_days": days,
        "forecast_days": 1,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        "timezone": "auto",
    }

    print(f"[Live API] Querying Glasgow observations (past {days} days + today)...")
    res = requests.get(url, params=params, timeout=20)
    res.raise_for_status()
    hourly = res.json()["hourly"]

    speed = np.array(hourly["wind_speed_10m"])
    direction_rad = np.radians(hourly["wind_direction_10m"])

    u_hourly = -speed * np.sin(direction_rad)
    v_hourly = -speed * np.cos(direction_rad)

    df_hourly = pd.DataFrame(
        {
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "dew_point": hourly["dew_point_2m"],
            "pressure": hourly["surface_pressure"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_u": u_hourly,
            "wind_v": v_hourly,
        },
        index=pd.to_datetime(hourly["time"]),
    )

    # Daily aggregation
    df_daily = pd.DataFrame(
        {
            "temperature": df_hourly["temperature"].resample("D").mean(),
            "humidity": df_hourly["humidity"].resample("D").mean(),
            "dew_point": df_hourly["dew_point"].resample("D").mean(),
            "pressure": df_hourly["pressure"].resample("D").mean(),
            "wind_speed_max": df_hourly["wind_speed"].resample("D").max(),
            "wind_u": df_hourly["wind_u"].resample("D").mean(),
            "wind_v": df_hourly["wind_v"].resample("D").mean(),
        }
    )

    # Late-day synoptic snapshots (18:00 and 23:00)
    h18 = df_hourly.loc[df_hourly.index.hour == 18].copy()
    h18.index = h18.index.normalize()

    h23 = df_hourly.loc[df_hourly.index.hour == 23].copy()
    h23.index = h23.index.normalize()

    df_daily["pressure_evening_tendency"] = h23["pressure"] - h18["pressure"]
    df_daily["temp_evening_tendency"] = h23["temperature"] - h18["temperature"]
    df_daily["wind_v_evening_tendency"] = h23["wind_v"] - h18["wind_v"]
    df_daily["pressure_23h"] = h23["pressure"]
    df_daily["temp_23h"] = h23["temperature"]

    return df_daily.ffill().bfill().dropna()


def append_github_summary(
    today_date: str,
    tomorrow_date: str,
    t_anchor: float,
    pred_p50: float,
    expected_delta: float,
    pred_p10: float,
    pred_p90: float,
    p_tend: float,
    v_wind: float,
) -> None:
    """Renders formatted Markdown tables to $GITHUB_STEP_SUMMARY when executed in CI/CD."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    trend_icon = (
        "🔺 Warmer"
        if expected_delta > 0.2
        else ("🔻 Colder" if expected_delta < -0.2 else "➡️ Steady")
    )
    advisory = (
        "> ⚠️ **Frost Alert:** Sub-zero temperatures expected within 80% boundary.\n"
        if pred_p10 <= 0
        else ""
    )

    markdown = f"""
## 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Glasgow Daily Temperature Bulletin
**Target Date:** `{tomorrow_date}` | **Baseline Date:** `{today_date}` (Mean: `{t_anchor:.2f}°C`)

| Metric | Prediction |
|---|---|
| **Expected Mean Temp (P50)** | **`{pred_p50:+.2f}°C`** |
| **Projected Shift (Δ)** | `{expected_delta:+.2f}°C` ({trend_icon}) |
| **80% Prediction Interval** | **`{pred_p10:.2f}°C` to `{pred_p90:.2f}°C`** |
| **Uncertainty Margin** | `±{(pred_p90 - pred_p10) / 2:.2f}°C` |
| **Evening Pressure Trend** | `{p_tend:+.2f} hPa / 5hr` |
| **Meridional Wind ($v$)** | `{v_wind:+.2f} km/h` |

{advisory}
"""
    with open(summary_file, "a") as f:
        f.write(markdown)


def main():
    # 1. Load pipeline bundle
    models, q_conf, val_bias = get_or_train_pipeline()

    # 2. Fetch live atmospheric observations for Glasgow
    recent_df = fetch_live_recent_window(days=14)
    features_df = build_features(recent_df)

    # 3. Isolate feature vector for today (t)
    X_today = features_df.iloc[[-1]].copy()
    feature_cols = [
        c for c in X_today.columns if c not in ["temperature", "target_delta"]
    ]
    X_today = X_today[feature_cols]

    today_date = features_df.index[-1].strftime("%Y-%m-%d")
    tomorrow_date = (features_df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t_anchor = float(recent_df["temperature"].iloc[-1])

    # 4. Predict raw quantile deltas
    raw_p05 = float(models[0.05].predict(X_today)[0])
    raw_p50 = float(models[0.50].predict(X_today)[0])
    raw_p95 = float(models[0.95].predict(X_today)[0])

    p05_d, p50_d, p95_d = np.sort([raw_p05, raw_p50, raw_p95])

    # 5. Apply CQR calibration and bias-corrected median shrinkage
    pred_p10 = (t_anchor + p05_d) - q_conf
    pred_p90 = (t_anchor + p95_d) + q_conf
    pred_p50 = (t_anchor + (p50_d * 0.65)) - val_bias
    pred_p50 = float(np.clip(pred_p50, pred_p10, pred_p90))

    expected_delta = pred_p50 - t_anchor
    spread = pred_p90 - pred_p10

    # 6. Terminal Bulletin
    print("\n" + "=" * 62)
    print("    GLASGOW OPERATIONAL 24-HOUR TEMPERATURE FORECAST")
    print("=" * 62)
    print(
        f"Issued At           : {np_datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"Observation Day     : {today_date} (Mean: {t_anchor:.2f}°C)")
    print(f"Target Forecast Day : {tomorrow_date}")
    print("-" * 62)
    print(f"Expected Mean Temp  : {pred_p50:+.2f}°C")
    print(
        f"Predicted Shift (Δ) : {expected_delta:+.2f}°C ({'Warmer' if expected_delta > 0.2 else ('Colder' if expected_delta < -0.2 else 'Steady')})"
    )
    print(f"80% Confidence Band : [{pred_p10:.2f}°C  to  {pred_p90:.2f}°C]")
    print(f"Interval Uncertainty: ±{spread / 2:.2f}°C (Total width: {spread:.2f}°C)")
    print("-" * 62)

    p_tend = float(X_today["pressure_tendency_eve"].iloc[0])
    v_wind = float(X_today["wind_v_lag_1"].iloc[0])
    print("[Synoptic Context]")
    print(
        f"Evening Barometer   : {p_tend:+.2f} hPa/5hr ({'Plunging Front' if p_tend < -2.0 else 'Stable'})"
    )
    print(
        f"Meridional Advection: {v_wind:+.2f} km/h ({'Southerly/Maritime' if v_wind > 0 else 'Northerly/Arctic'})"
    )

    if pred_p10 <= 0.0:
        print("\nADVISORY: Frost/sub-zero conditions fall within the 80% boundary.")
    print("=" * 62 + "\n")

    # 7. CI/CD Step Summary hook
    append_github_summary(
        today_date=today_date,
        tomorrow_date=tomorrow_date,
        t_anchor=t_anchor,
        pred_p50=pred_p50,
        expected_delta=expected_delta,
        pred_p10=pred_p10,
        pred_p90=pred_p90,
        p_tend=p_tend,
        v_wind=v_wind,
    )


if __name__ == "__main__":
    main()
