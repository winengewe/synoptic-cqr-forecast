import os
from typing import Optional
import numpy as np
import pandas as pd
import requests


def load_weather_data(
    latitude: float = 55.8642,  # Glasgow
    longitude: float = -4.2518,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    cache_path: Optional[str] = "data/weather_history.csv",
) -> pd.DataFrame:
    """
    Fetches real weather data from Open-Meteo, computes hourly u/v wind vectors,
    extracts late-day synoptic tendencies, and aggregates daily metrics.
    """
    if cache_path:
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if "pressure_evening_tendency" in cached.columns:
                print(
                    f"[Cache] Loaded {len(cached)} records with evening tendencies from '{cache_path}'."
                )
                return cached
            print("[Cache] Existing cache lacks evening tendencies. Re-fetching...")
        except FileNotFoundError:
            pass

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
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

    print(
        f"[API] Querying Open-Meteo for ({latitude}, {longitude}) from {start_date} to {end_date}..."
    )
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    speed = np.array(hourly["wind_speed_10m"])
    direction_rad = np.radians(hourly["wind_direction_10m"])

    # Decomposition into zonal (u) and meridional (v) wind vectors
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

    # 1. Base 24-hour mean aggregations
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

    # 2. Extract late-day boundary snapshots (18:00 and 23:00)
    h18 = df_hourly.loc[df_hourly.index.hour == 18].copy()
    h18.index = h18.index.normalize()

    h23 = df_hourly.loc[df_hourly.index.hour == 23].copy()
    h23.index = h23.index.normalize()

    # 3. Sub-daily evening dynamics
    df_daily["pressure_evening_tendency"] = h23["pressure"] - h18["pressure"]
    df_daily["temp_evening_tendency"] = h23["temperature"] - h18["temperature"]
    df_daily["wind_v_evening_tendency"] = h23["wind_v"] - h18["wind_v"]
    df_daily["pressure_23h"] = h23["pressure"]
    df_daily["temp_23h"] = h23["temperature"]

    df_daily = df_daily.ffill().bfill().dropna()

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        df_daily.to_csv(cache_path)
        print(
            f"[Cache] Saved dataset with evening tendency features to '{cache_path}'."
        )

    return df_daily
