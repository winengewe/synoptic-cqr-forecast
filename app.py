import os
import sys

# Ensure src/ directory is on sys.path for Streamlit Cloud and container environments
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st

from synoptic_forecast.data import load_weather_data
from synoptic_forecast.features import build_features
from synoptic_forecast.metrics import evaluate_forecast
from synoptic_forecast.model import reconstruct_forecasts
import predict_tomorrow

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Glasgow Synoptic-CQR Weather Forecast",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.25);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700;
    }
    /* Banner Callouts */
    .synoptic-alert {
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-weight: 500;
    }
    .synoptic-alert-frost {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
    }
    .synoptic-alert-front {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        color: #92400e;
    }
    .synoptic-alert-stable {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        color: #065f46;
    }
    /* Custom Badge */
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: center;
    }
    .badge-pass { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
    .badge-warn { background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
    .badge-fail { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Resource & Data Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline_bundle():
    """Loads or trains the calibrated XGBoost CQR model bundle."""
    return predict_tomorrow.get_or_train_pipeline()


@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_dataset():
    """Loads cached historical Glasgow weather dataset and computes synoptic features."""
    raw_df = load_weather_data(
        latitude=predict_tomorrow.LATITUDE,
        longitude=predict_tomorrow.LONGITUDE,
        start_date="2023-01-01",
        end_date="2025-12-31",
        cache_path="data/weather_history.csv",
    )
    df = build_features(raw_df)
    return raw_df, df


@st.cache_data(ttl=900, show_spinner=False)
def get_live_data(days: int = 14):
    """Fetches real-time recent atmospheric observations via Open-Meteo API."""
    return predict_tomorrow.fetch_live_recent_window(days=days)


@st.cache_data(show_spinner=False)
def compute_backtest_evaluation(
    coverage_target: float = 0.80, delta_shrinkage: float = 0.65
):
    """Executes chronological test partition evaluation with custom calibration parameters."""
    bundle_models, _, _ = load_pipeline_bundle()
    _, df = get_historical_dataset()

    val_cutoff = "2025-09-01"
    test_cutoff = "2025-11-01"

    feature_cols = [c for c in df.columns if c not in ["temperature", "target_delta"]]
    X = df[feature_cols]
    y_delta = df["target_delta"]
    y_actual = df["temperature"]

    X_val = X.loc[(X.index >= val_cutoff) & (X.index < test_cutoff)]
    y_val_actual = y_actual.loc[
        (y_actual.index >= val_cutoff) & (y_actual.index < test_cutoff)
    ]

    X_test = X.loc[X.index >= test_cutoff]
    y_test_actual = y_actual.loc[y_actual.index >= test_cutoff]

    forecasts, q_conf, val_bias = reconstruct_forecasts(
        models=bundle_models,
        X_test=X_test,
        test_index=y_test_actual.index,
        X_val=X_val,
        y_val_actual=y_val_actual,
        delta_shrinkage=delta_shrinkage,
        coverage_target=coverage_target,
    )

    scorecard = evaluate_forecast(
        y_true=y_test_actual,
        y_pred_p50=forecasts[0.50],
        p10=forecasts[0.10],
        p90=forecasts[0.90],
        temp_lag_1=X_test["temp_lag_1"],
    )

    y_train_actual = y_actual.loc[y_actual.index < test_cutoff]

    return {
        "scorecard": scorecard,
        "forecasts": forecasts,
        "y_test_actual": y_test_actual,
        "y_train_actual": y_train_actual,
        "X_test": X_test,
        "q_conf": q_conf,
        "val_bias": val_bias,
        "test_cutoff": test_cutoff,
    }


# ---------------------------------------------------------
# Sidebar Configuration & Metadata
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=600&auto=format&fit=crop&q=60",
        caption="Synoptic Meteorology & CQR Engine",
        use_container_width=True,
    )
    st.title("⚙️ Engine Controls")

    st.markdown("### 📍 Operational Station")
    st.info(
        f"**Station:** Glasgow, Scotland\n\n"
        f"**Coordinates:** {predict_tomorrow.LATITUDE}°N, {abs(predict_tomorrow.LONGITUDE)}°W\n\n"
        f"**Target:** Daily Mean Temp (24h Ahead)\n\n"
        f"**Quantiles:** P05, P50, P95 + CQR"
    )

    st.markdown("### 🎯 Calibration Hyperparameters")
    coverage_slider = st.slider(
        "Conformal Coverage Target (1 - α)",
        min_value=0.70,
        max_value=0.95,
        value=0.80,
        step=0.05,
        help="Nominal coverage probability guaranteed by Non-Negative Conformalized Quantile Regression.",
    )
    shrinkage_slider = st.slider(
        "Delta Shrinkage (Persistence Bias)",
        min_value=0.30,
        max_value=1.00,
        value=0.65,
        step=0.05,
        help="Dampens raw predicted median delta toward persistence before applying validation bias offset.",
    )

    st.markdown("### 🛰️ Live API Fetch Window")
    history_days = st.slider(
        "Recent Observation Window (Days)",
        min_value=7,
        max_value=30,
        value=14,
        step=1,
    )

    if st.button("🔄 Force Refresh Live Observations", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    models, q_conf_bundle, val_bias_bundle = load_pipeline_bundle()
    st.divider()
    st.caption(
        f"**Active Bundle:** `models/model_bundle.joblib`\n\n"
        f"• Conformal $q_{{conf}}$: `±{q_conf_bundle:.2f}°C`\n\n"
        f"• Validation Bias: `{val_bias_bundle:+.2f}°C`"
    )

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.title("🌦️ Glasgow Synoptic Weather Forecast & CQR Scorecard")
st.markdown(
    "Operational 24-hour probabilistic temperature forecasts powered by **XGBoost Pinball Loss**, "
    "**Conformalized Quantile Regression (CQR)**, and **Late-Day Barometric & Wind Tendencies**."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🛰️ Operational 24h Forecast",
        "📊 Benchmark Scorecard & Verification",
        "🧠 Synoptic Explainability & SHAP",
        "🎛️ What-If Synoptic Simulator",
    ]
)


# =========================================================
# TAB 1: OPERATIONAL 24-HOUR FORECAST
# =========================================================
with tab1:
    st.subheader("Operational 24-Hour Forecast Bulletin")

    with st.spinner("Querying live Open-Meteo observations for Glasgow..."):
        try:
            recent_df = get_live_data(days=history_days)
            features_df = build_features(recent_df)

            X_today = features_df.iloc[[-1]].copy()
            feature_cols = [
                c for c in X_today.columns if c not in ["temperature", "target_delta"]
            ]
            X_today = X_today[feature_cols]

            today_date = features_df.index[-1].strftime("%Y-%m-%d")
            tomorrow_date = (features_df.index[-1] + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            t_anchor = float(recent_df["temperature"].iloc[-1])

            # Model inference
            raw_p05 = float(models[0.05].predict(X_today)[0])
            raw_p50 = float(models[0.50].predict(X_today)[0])
            raw_p95 = float(models[0.95].predict(X_today)[0])

            p05_d, p50_d, p95_d = np.sort([raw_p05, raw_p50, raw_p95])

            pred_p10 = (t_anchor + p05_d) - q_conf_bundle
            pred_p90 = (t_anchor + p95_d) + q_conf_bundle
            pred_p50 = (t_anchor + (p50_d * shrinkage_slider)) - val_bias_bundle
            pred_p50 = float(np.clip(pred_p50, pred_p10, pred_p90))

            expected_delta = pred_p50 - t_anchor
            spread = pred_p90 - pred_p10
            uncertainty_margin = spread / 2.0

            p_tend = float(X_today["pressure_tendency_eve"].iloc[0])
            v_wind = float(X_today["wind_v_lag_1"].iloc[0])
            u_wind = float(X_today["wind_u_lag_1"].iloc[0])
            p_dep = float(X_today["evening_pressure_departure"].iloc[0])
            t_dep = float(X_today["evening_temp_departure"].iloc[0])

            # Alerts & Synoptic Conditions
            alert_items = []
            if pred_p10 <= 0.0:
                alert_items.append(
                    (
                        "synoptic-alert-frost",
                        f"⚠️ <b>Frost / Sub-Zero Alert:</b> Lower 80% boundary reaches {pred_p10:.1f}°C. Ground frost or freezing conditions possible.",
                    )
                )
            if p_tend < -2.0:
                alert_items.append(
                    (
                        "synoptic-alert-front",
                        f"📉 <b>Plunging Front Detected:</b> 5-hour evening barometric plunge of {p_tend:+.1f} hPa indicates an active incoming low-pressure frontal system.",
                    )
                )
            elif p_tend > 1.5:
                alert_items.append(
                    (
                        "synoptic-alert-stable",
                        f"📈 <b>Anticyclonic High Pressure:</b> Evening pressure rose by {p_tend:+.1f} hPa. Stable airmass expected.",
                    )
                )

            if not alert_items:
                alert_items.append(
                    (
                        "synoptic-alert-stable",
                        "✅ <b>Stable Synoptic Setup:</b> Barometric and wind conditions indicate steady atmospheric evolution.",
                    )
                )

            for alert_cls, alert_msg in alert_items:
                st.markdown(
                    f'<div class="synoptic-alert {alert_cls}">{alert_msg}</div>',
                    unsafe_allow_html=True,
                )

            # Hero KPI Cards
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.metric(
                    label=f"Baseline Today ({today_date})",
                    value=f"{t_anchor:.2f} °C",
                    help="Mean 24-hour temperature observed today.",
                )
            with kpi_col2:
                trend_str = (
                    "Warmer 🔺"
                    if expected_delta > 0.2
                    else ("Colder 🔻" if expected_delta < -0.2 else "Steady ➡️")
                )
                st.metric(
                    label=f"Forecast Tomorrow ({tomorrow_date})",
                    value=f"{pred_p50:.2f} °C",
                    delta=f"{expected_delta:+.2f} °C ({trend_str})",
                    help="Median point prediction (P50) with shrinkage and validation bias centering.",
                )
            with kpi_col3:
                st.metric(
                    label=f"{int(coverage_slider * 100)}% Prediction Interval",
                    value=f"[{pred_p10:.1f}°, {pred_p90:.1f}°C]",
                    help=f"Conformalized prediction band guaranteeing {int(coverage_slider * 100)}% coverage.",
                )
            with kpi_col4:
                st.metric(
                    label="Uncertainty Margin (±)",
                    value=f"±{uncertainty_margin:.2f} °C",
                    delta=f"Band Width: {spread:.2f} °C",
                    delta_color="off",
                    help="Half-width of the conformal prediction interval.",
                )

            # Synoptic State Cards
            st.markdown("#### 🧭 Evening Atmospheric Boundary Conditions")
            syn_c1, syn_c2, syn_c3, syn_c4 = st.columns(4)
            with syn_c1:
                p_label = "Plunging ⚠️" if p_tend < -2.0 else ("Rising" if p_tend > 1.0 else "Stable")
                st.metric(
                    label="Evening Barometer (18h→23h)",
                    value=f"{p_tend:+.2f} hPa",
                    delta=p_label,
                    delta_color="normal" if p_tend >= -2.0 else "inverse",
                )
            with syn_c2:
                v_label = "Southerly (Maritime Atlantic)" if v_wind > 0 else "Northerly (Polar Arctic)"
                st.metric(
                    label="Meridional Wind (v)",
                    value=f"{v_wind:+.1f} km/h",
                    delta=v_label,
                    delta_color="off",
                )
            with syn_c3:
                u_label = "Westerly" if u_wind > 0 else "Easterly"
                st.metric(
                    label="Zonal Wind (u)",
                    value=f"{u_wind:+.1f} km/h",
                    delta=u_label,
                    delta_color="off",
                )
            with syn_c4:
                st.metric(
                    label="Late-Night Temp Shift (23h)",
                    value=f"{t_dep:+.2f} °C",
                    delta="vs Daily Mean",
                    delta_color="off",
                )

            # Interactive Forecast Fan Plotly Chart
            st.markdown("#### 📈 Atmospheric Trajectory & Prediction Fan")

            # Build forecast time series
            hist_series = recent_df["temperature"]
            fig = go.Figure()

            # Plot historical observed line
            fig.add_trace(
                go.Scatter(
                    x=hist_series.index,
                    y=hist_series.values,
                    mode="lines+markers",
                    name="Observed Mean Temp",
                    line=dict(color="#2563eb", width=2.5),
                    marker=dict(size=6),
                )
            )

            # Forecast connection segment (Today -> Tomorrow)
            t_last_date = hist_series.index[-1]
            t_next_date = pd.to_datetime(tomorrow_date)
            forecast_dates = [t_last_date, t_next_date]

            # 80% Upper band
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[t_anchor, pred_p90],
                    mode="lines",
                    name=f"P90 Upper Bound ({pred_p90:.1f}°C)",
                    line=dict(color="#10b981", width=1.5, dash="dash"),
                    showlegend=True,
                )
            )

            # 80% Lower band with fill
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[t_anchor, pred_p10],
                    mode="lines",
                    name=f"P10 Lower Bound ({pred_p10:.1f}°C)",
                    line=dict(color="#10b981", width=1.5, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(16, 185, 129, 0.20)",
                    showlegend=True,
                )
            )

            # Median forecast point
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=[t_anchor, pred_p50],
                    mode="lines+markers",
                    name=f"P50 Forecast ({pred_p50:.2f}°C)",
                    line=dict(color="#059669", width=3),
                    marker=dict(size=[0, 10], symbol="diamond"),
                )
            )

            # Freezing Reference line (0°C)
            fig.add_hline(
                y=0,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="0°C Freezing Line",
                annotation_position="bottom right",
            )

            fig.update_layout(
                title=f"Glasgow 24-Hour Temperature Trajectory & CQR Forecast Interval ({int(coverage_slider*100)}% Coverage)",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                hovermode="x unified",
                template="plotly_white",
                height=480,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 View Recent Atmospheric Data Table"):
                st.dataframe(
                    recent_df.tail(10).style.format("{:.2f}"),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error executing operational live forecast: {e}")
            st.info(
                "Ensure internet access to `api.open-meteo.com` is active or try refreshing the window."
            )


# =========================================================
# TAB 2: BENCHMARK SCORECARD & VERIFICATION
# =========================================================
with tab2:
    st.subheader("Model Benchmark Scorecard & Chronological Verification")
    st.markdown(
        "Evaluation across the held-out test partition (**November 1, 2025 to December 31, 2025**). "
        "Targets define operational requirements for deployment."
    )

    with st.spinner("Computing test partition metrics..."):
        eval_res = compute_backtest_evaluation(
            coverage_target=coverage_slider, delta_shrinkage=shrinkage_slider
        )
        scorecard = eval_res["scorecard"]
        y_test_actual = eval_res["y_test_actual"]
        forecasts = eval_res["forecasts"]
        y_train_actual = eval_res["y_train_actual"]
        q_conf_eval = eval_res["q_conf"]
        val_bias_eval = eval_res["val_bias"]

    # Render Scorecard Table
    sc_rows = []
    for metric_name, m_data in scorecard.items():
        val = m_data["value"]
        unit = m_data["unit"]
        target = m_data["target"]
        status = m_data["status"]

        if "Bias" in metric_name:
            val_str = f"{val:+.2f} {unit}".strip()
        elif unit == "%":
            val_str = f"{val:.1f} %"
        else:
            val_str = f"{val:.2f} {unit}".strip()

        badge_class = (
            "badge-pass"
            if status == "PASS"
            else ("badge-warn" if status == "WARN" else "badge-fail")
        )
        status_html = f'<span class="metric-badge {badge_class}">{status}</span>'

        sc_rows.append(
            {
                "Metric": metric_name,
                "Observed Score": val_str,
                "Target / Baseline Benchmark": target,
                "Status": status_html,
            }
        )

    sc_df = pd.DataFrame(sc_rows)
    st.write(sc_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.markdown("---")

    # Time-Series Interactive Verification Plot
    st.subheader("Chronological Backtest Time-Series")
    plot_window = st.radio(
        "Display Range:",
        ["Test Partition Only (61 Days)", "Include Preceding 60 Days History"],
        horizontal=True,
    )

    verif_fig = go.Figure()

    if "Include" in plot_window:
        hist_window = y_train_actual.iloc[-60:]
        verif_fig.add_trace(
            go.Scatter(
                x=hist_window.index,
                y=hist_window.values,
                mode="lines",
                name="Historical Observed (Train/Val)",
                line=dict(color="#94a3b8", width=1.5),
            )
        )

    # Actual Test Observed
    verif_fig.add_trace(
        go.Scatter(
            x=y_test_actual.index,
            y=y_test_actual.values,
            mode="lines+markers",
            name="Ground Truth Actual",
            line=dict(color="#0f172a", width=2),
            marker=dict(size=4),
        )
    )

    # Upper 80% Bound
    verif_fig.add_trace(
        go.Scatter(
            x=y_test_actual.index,
            y=forecasts[0.90].values,
            mode="lines",
            name="P90 Upper Interval",
            line=dict(color="#10b981", width=1, dash="dot"),
        )
    )

    # Lower 80% Bound with Shading
    verif_fig.add_trace(
        go.Scatter(
            x=y_test_actual.index,
            y=forecasts[0.10].values,
            mode="lines",
            name="P10 Lower Interval",
            line=dict(color="#10b981", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.20)",
        )
    )

    # Median Forecast
    verif_fig.add_trace(
        go.Scatter(
            x=y_test_actual.index,
            y=forecasts[0.50].values,
            mode="lines",
            name="Median Forecast (P50)",
            line=dict(color="#2563eb", width=2, dash="dash"),
        )
    )

    # Flag Anomalies (Outside 80% interval)
    outside_mask = (y_test_actual < forecasts[0.10]) | (y_test_actual > forecasts[0.90])
    if outside_mask.sum() > 0:
        verif_fig.add_trace(
            go.Scatter(
                x=y_test_actual[outside_mask].index,
                y=y_test_actual[outside_mask].values,
                mode="markers",
                name=f"Interval Outliers ({outside_mask.sum()})",
                marker=dict(size=9, color="#ef4444", symbol="x"),
            )
        )

    verif_fig.update_layout(
        title="Glasgow Test Period: Observed Actual vs CQR Median & 80% Confidence Band",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(verif_fig, use_container_width=True)

    # Outlier Diagnostic & Residual Distribution
    col_out1, col_out2 = st.columns([1, 1])

    abs_residuals = (y_test_actual - forecasts[0.50]).abs()
    worst_date = abs_residuals.idxmax()
    worst_val = abs_residuals.max()
    actual_on_worst = y_test_actual.loc[worst_date]
    pred_on_worst = forecasts[0.50].loc[worst_date]
    p10_on_worst = forecasts[0.10].loc[worst_date]
    p90_on_worst = forecasts[0.90].loc[worst_date]
    inside_worst = p10_on_worst <= actual_on_worst <= p90_on_worst

    with col_out1:
        st.subheader("🔍 Worst-Case Outlier Diagnostic")
        st.markdown(
            f"""
            <div style="background-color: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.3); padding: 16px; border-radius: 8px;">
                <p><b>Worst Miss Date:</b> <code>{worst_date.strftime('%Y-%m-%d')}</code></p>
                <p><b>Observed Actual:</b> <code>{actual_on_worst:.2f}°C</code></p>
                <p><b>Median Forecast:</b> <code>{pred_on_worst:.2f}°C</code></p>
                <p><b>Absolute Error:</b> <code>{worst_val:.2f}°C</code></p>
                <p><b>80% Prediction Band:</b> <code>[{p10_on_worst:.2f}°C to {p90_on_worst:.2f}°C]</code></p>
                <p><b>Captured inside 80% band?</b> <b>{'Yes ✅' if inside_worst else 'No ❌ (Extreme Weather Anomaly)'}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_out2:
        st.subheader("📊 Forecast Error (Residual) Distribution")
        residuals = forecasts[0.50] - y_test_actual
        hist_fig = px.histogram(
            residuals,
            nbins=20,
            labels={"value": "Residual (P50 - Actual) [°C]"},
            title="Residual Error Distribution (Median Error)",
            color_discrete_sequence=["#3b82f6"],
        )
        hist_fig.add_vline(x=0, line_dash="dash", line_color="#ef4444")
        hist_fig.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(hist_fig, use_container_width=True)


# =========================================================
# TAB 3: SYNOPTIC EXPLAINABILITY & SHAP
# =========================================================
with tab3:
    st.subheader("Synoptic Atmospheric Drivers & Explainability")
    st.markdown(
        "Tree-based SHAP (SHapley Additive exPlanations) and feature importances reveal "
        "how sub-daily atmospheric boundaries (18:00 to 23:00 tendencies) outperform standard lag persistence."
    )

    with st.spinner("Calculating SHAP explanations for XGBoost P50 model..."):
        model_p50 = models[0.50]
        feature_names = model_p50.feature_names_in_
        importances = model_p50.feature_importances_

        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .reset_index(drop=True)
        )

        # Categorize features
        def categorize_feat(f):
            if "tendency" in f or "departure" in f:
                return "Evening Synoptic Precursor"
            elif "wind" in f:
                return "Wind Vector Advection"
            elif "pressure" in f:
                return "Barometric State"
            else:
                return "Temperature Autoregressive Lag"

        fi_df["Category"] = fi_df["Feature"].apply(categorize_feat)

    col_fi, col_shap = st.columns([1, 1])

    with col_fi:
        st.markdown("#### Feature Importance Ranking")
        fi_fig = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            color="Category",
            orientation="h",
            title="XGBoost Median Regressor Split Weight",
            color_discrete_map={
                "Evening Synoptic Precursor": "#ef4444",
                "Wind Vector Advection": "#3b82f6",
                "Barometric State": "#f59e0b",
                "Temperature Autoregressive Lag": "#10b981",
            },
        )
        fi_fig.update_layout(
            template="plotly_white",
            height=580,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fi_fig, use_container_width=True)

    with col_shap:
        st.markdown("#### Top Synoptic Explanations (Recent Test Window)")
        _, raw_features_df = get_historical_dataset()
        feat_cols = [c for c in raw_features_df.columns if c not in ["temperature", "target_delta"]]
        sample_X = raw_features_df[feat_cols].iloc[-60:]

        explainer = shap.TreeExplainer(model_p50)
        shap_values = explainer(sample_X)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_df = (
            pd.DataFrame({"Feature": feat_cols, "Mean_Abs_SHAP": mean_abs_shap})
            .sort_values("Mean_Abs_SHAP", ascending=True)
            .reset_index(drop=True)
        )

        shap_fig = px.bar(
            shap_df.tail(10),
            x="Mean_Abs_SHAP",
            y="Feature",
            orientation="h",
            title="Top 10 Drivers by Mean Absolute SHAP Impact (°C)",
            color="Mean_Abs_SHAP",
            color_continuous_scale="Viridis",
        )
        shap_fig.update_layout(
            template="plotly_white",
            height=580,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(shap_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📚 Meteorological & Conformal Mathematical Framework")
    meth_col1, meth_col2 = st.columns(2)

    with meth_col1:
        st.markdown(
            r"""
            #### 1. Why Sub-Daily Evening Tendencies Work
            Standard weather forecasts rely on 24-hour mean aggregations or midnight lags. However, in maritime climates like Glasgow:
            - **Evening Barometer (18:00 to 23:00):** A steep pressure drop ($\Delta P < -2$ hPa/5h) indicates rapid cold front encroachment before midnight.
            - **Meridional Advection ($v$):** South-to-north Atlantic airmass transport ($v > 0$) brings mild maritime moisture; north-to-south transport ($v < 0$) advects Arctic polar air.
            - **Evening Departure:** If late-night temperatures remain elevated relative to the daily mean, a warm sector has overridden the region.
            """
        )

    with meth_col2:
        st.markdown(
            r"""
            #### 2. Non-Negative Conformalized Quantile Regression (CQR)
            XGBoost predicts conditional quantiles under the asymmetric pinball loss:
            $$\mathcal{L}_q(y, \hat{y}) = \max\{q(y - \hat{y}), (1-q)(\hat{y} - y)\}$$
            CQR calibrates empirical coverage on the validation partition via non-conformity scores:
            $$s_i = \max(\hat{q}_{0.05}(x_i) - y_i,\; y_i - \hat{q}_{0.95}(x_i))$$
            $$q_{\text{conf}} = \max(0, \text{Quantile}_{1-\alpha}(s_i))$$
            This mathematically guarantees that $P(Y \in [\hat{q}_{0.05} - q_{\text{conf}}, \hat{q}_{0.95} + q_{\text{conf}}]) \ge 1 - \alpha$.
            """
        )


# =========================================================
# TAB 4: WHAT-IF SYNOPTIC SIMULATOR
# =========================================================
with tab4:
    st.subheader("🎛️ Interactive Synoptic Scenario Simulator")
    st.markdown(
        "Simulate hypothetical synoptic patterns (e.g., severe Atlantic cold front passages, "
        "Arctic blasts, or high-pressure blocking) and observe real-time model response."
    )

    # Preset selection
    st.markdown("##### ⚡ Quick Atmospheric Presets")
    preset_cols = st.columns(4)

    preset_name = None
    if preset_cols[0].button("❄️ Polar Arctic Blast", use_container_width=True):
        preset_name = "arctic"
    if preset_cols[1].button("🌪️ Rapid Atlantic Cold Front", use_container_width=True):
        preset_name = "cold_front"
    if preset_cols[2].button("☀️ Summer Heat / High Pressure", use_container_width=True):
        preset_name = "heatwave"
    if preset_cols[3].button("🔄 Reset to Neutral Baseline", use_container_width=True):
        preset_name = "neutral"

    # Default values based on preset
    def_temp = 9.0
    def_ptend = 0.0
    def_pdelta = 0.0
    def_vwind = 0.0
    def_uwind = 10.0
    def_tdep = 0.0
    def_humidity = 80.0

    if preset_name == "arctic":
        def_temp = 3.0
        def_ptend = 1.8
        def_pdelta = 8.0
        def_vwind = -25.0
        def_uwind = -5.0
        def_tdep = -3.5
        def_humidity = 65.0
    elif preset_name == "cold_front":
        def_temp = 12.0
        def_ptend = -4.2
        def_pdelta = -12.0
        def_vwind = 22.0
        def_uwind = 28.0
        def_tdep = -2.0
        def_humidity = 92.0
    elif preset_name == "heatwave":
        def_temp = 20.0
        def_ptend = 0.5
        def_pdelta = 2.0
        def_vwind = 15.0
        def_uwind = 5.0
        def_tdep = 2.5
        def_humidity = 60.0

    sim_c1, sim_c2, sim_c3 = st.columns(3)

    with sim_c1:
        st.markdown("**Temperature State**")
        sim_temp = st.slider("Today's Mean Temp (°C)", -5.0, 25.0, def_temp, 0.5)
        sim_tdep = st.slider("23h Temp Departure (°C)", -5.0, 5.0, def_tdep, 0.2)
        sim_humidity = st.slider("Relative Humidity (%)", 40.0, 100.0, def_humidity, 1.0)

    with sim_c2:
        st.markdown("**Barometric Pressure**")
        sim_ptend = st.slider("Evening Tendency 18h→23h (hPa/5h)", -6.0, 4.0, def_ptend, 0.2)
        sim_pdelta = st.slider("24h Barometer Delta (hPa)", -20.0, 20.0, def_pdelta, 0.5)
        sim_pdep = st.slider("Evening Pressure Departure (hPa)", -5.0, 5.0, 0.0, 0.2)

    with sim_c3:
        st.markdown("**Wind Vector Advection**")
        sim_vwind = st.slider("Meridional Wind v (km/h, +South / -North)", -35.0, 35.0, def_vwind, 1.0)
        sim_uwind = st.slider("Zonal Wind u (km/h, +West / -East)", -35.0, 35.0, def_uwind, 1.0)
        sim_speed = st.slider("Max Wind Speed (km/h)", 0.0, 70.0, 25.0, 1.0)

    # Construct feature vector
    sim_vector = pd.DataFrame(
        [
            {
                "temp_lag_1": sim_temp,
                "temp_lag_2": sim_temp - 0.2,
                "temp_lag_7": sim_temp + 0.5,
                "delta_lag_1": 0.2,
                "delta_lag_2": -0.1,
                "temp_ewma_2d": sim_temp,
                "pressure_lag_1": 1013.25,
                "pressure_delta_24h": sim_pdelta,
                "dew_depression_lag_1": (100.0 - sim_humidity) / 5.0,
                "humidity_lag_1": sim_humidity,
                "wind_speed_lag_1": sim_speed,
                "wind_u_lag_1": sim_uwind,
                "wind_v_lag_1": sim_vwind,
                "wind_u_delta_24h": 0.0,
                "wind_v_delta_24h": 0.0,
                "pressure_tendency_eve": sim_ptend,
                "temp_tendency_eve": sim_tdep,
                "wind_v_tendency_eve": 0.0,
                "evening_pressure_departure": sim_pdep,
                "evening_temp_departure": sim_tdep,
            }
        ]
    )

    # Predict with active models
    raw_s05 = float(models[0.05].predict(sim_vector)[0])
    raw_s50 = float(models[0.50].predict(sim_vector)[0])
    raw_s95 = float(models[0.95].predict(sim_vector)[0])
    s05_d, s50_d, s95_d = np.sort([raw_s05, raw_s50, raw_s95])

    sim_p10 = (sim_temp + s05_d) - q_conf_bundle
    sim_p90 = (sim_temp + s95_d) + q_conf_bundle
    sim_p50 = (sim_temp + (s50_d * shrinkage_slider)) - val_bias_bundle
    sim_p50 = float(np.clip(sim_p50, sim_p10, sim_p90))
    sim_delta = sim_p50 - sim_temp

    st.markdown("---")
    st.markdown("#### 🔮 Simulated Output Response")
    out_c1, out_c2, out_c3, out_c4 = st.columns(4)
    with out_c1:
        st.metric("Baseline Anchor", f"{sim_temp:.2f} °C")
    with out_c2:
        st.metric(
            "Predicted Mean (P50)",
            f"{sim_p50:.2f} °C",
            delta=f"{sim_delta:+.2f} °C ({'Warmer' if sim_delta > 0.2 else ('Colder' if sim_delta < -0.2 else 'Steady')})",
        )
    with out_c3:
        st.metric(f"{int(coverage_slider*100)}% Interval", f"[{sim_p10:.1f}°, {sim_p90:.1f}°C]")
    with out_c4:
        st.metric("Uncertainty Margin (±)", f"±{(sim_p90 - sim_p10)/2:.2f} °C")

    # Interactive Bullet / Gauge representation
    gauge_fig = go.Figure()
    gauge_fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=sim_p50,
            delta={"reference": sim_temp, "valueformat": "+.2f", "suffix": " °C"},
            title={"text": "Tomorrow's Projected Mean Temperature (°C)"},
            gauge={
                "axis": {"range": [-10, 30], "tickwidth": 1},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [-10, 0], "color": "#fee2e2"},
                    {"range": [0, 10], "color": "#dbeafe"},
                    {"range": [10, 20], "color": "#d1fae5"},
                    {"range": [20, 30], "color": "#fef3c7"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 4},
                    "thickness": 0.75,
                    "value": sim_p50,
                },
            },
        )
    )
    gauge_fig.update_layout(height=300, template="plotly_white")
    st.plotly_chart(gauge_fig, use_container_width=True)
