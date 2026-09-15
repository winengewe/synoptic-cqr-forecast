import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms atmospheric state, wind vectors, and evening tendencies into tabular features."""
    data = df.copy()

    # 1. Target (Daily Temperature Delta)
    data["target_delta"] = data["temperature"].diff(1)

    # 2. Temperature Lags & Autoregressive Memory
    for lag in [1, 2, 7]:
        data[f"temp_lag_{lag}"] = data["temperature"].shift(lag)
    data["delta_lag_1"] = data["target_delta"].shift(1)
    data["delta_lag_2"] = data["target_delta"].shift(2)
    data["temp_ewma_2d"] = data["temperature"].shift(1).ewm(span=2).mean()

    # 3. Barometric and Moisture Precursors
    data["pressure_lag_1"] = data["pressure"].shift(1)
    data["pressure_delta_24h"] = data["pressure"].shift(1) - data["pressure"].shift(2)
    data["dew_depression_lag_1"] = data["temperature"].shift(1) - data[
        "dew_point"
    ].shift(1)
    data["humidity_lag_1"] = data["humidity"].shift(1)
    data["wind_speed_lag_1"] = data["wind_speed_max"].shift(1)

    # 4. Wind Vector Advection
    data["wind_u_lag_1"] = data["wind_u"].shift(1)
    data["wind_v_lag_1"] = data["wind_v"].shift(1)
    data["wind_u_delta_24h"] = data["wind_u"].shift(1) - data["wind_u"].shift(2)
    data["wind_v_delta_24h"] = data["wind_v"].shift(1) - data["wind_v"].shift(2)

    # 5. Late-Day Synoptic Tendencies (Shifted by 1 day to represent the eve of forecast day)
    # 5-hour rate of change right before midnight
    data["pressure_tendency_eve"] = data["pressure_evening_tendency"].shift(1)
    data["temp_tendency_eve"] = data["temp_evening_tendency"].shift(1)
    data["wind_v_tendency_eve"] = data["wind_v_evening_tendency"].shift(1)

    # Evening departure from daily baseline (detects late-night warm sectors and pressure plunges)
    data["evening_pressure_departure"] = data["pressure_23h"].shift(1) - data[
        "pressure"
    ].shift(1)
    data["evening_temp_departure"] = data["temp_23h"].shift(1) - data[
        "temperature"
    ].shift(1)

    # 6. Drop unlagged ground truth columns
    raw_columns_to_drop = [
        "humidity",
        "dew_point",
        "pressure",
        "wind_speed_max",
        "wind_u",
        "wind_v",
        "pressure_evening_tendency",
        "temp_evening_tendency",
        "wind_v_evening_tendency",
        "pressure_23h",
        "temp_23h",
    ]
    data = data.drop(columns=raw_columns_to_drop)

    return data.dropna()
