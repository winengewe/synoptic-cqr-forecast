import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_results(
    y_train: pd.Series,
    y_test: pd.Series,
    p10: pd.Series,
    p50: pd.Series,
    p90: pd.Series,
    split_cutoff: str,
    history_window_days: int = 60,
):
    """Draws a continuous line plot across split boundaries with an 80% shaded prediction interval."""
    boundary = y_train.iloc[-1:]

    plot_actual = pd.concat([boundary, y_test])
    plot_p50 = pd.concat([boundary, p50])
    plot_p10 = pd.concat([boundary, p10])
    plot_p90 = pd.concat([boundary, p90])

    plt.figure(figsize=(13, 6))
    plt.plot(
        y_train.index[-history_window_days:],
        y_train.iloc[-history_window_days:],
        label=f"Historical Observed (Last {history_window_days} Days)",
        color="#1f77b4",
        linewidth=1.5,
    )
    plt.plot(
        plot_actual.index,
        plot_actual,
        label="Actual Observed",
        color="black",
        linewidth=1.8,
    )
    plt.plot(
        plot_p50.index,
        plot_p50,
        label="Median Forecast (P50)",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
    )
    plt.fill_between(
        plot_p10.index,
        plot_p10,
        plot_p90,
        color="#2ca02c",
        alpha=0.2,
        label="80% Prediction Interval (P10–P90)",
    )

    plt.axvline(
        x=pd.to_datetime(split_cutoff),
        color="gray",
        linestyle=":",
        label="Split Boundary",
    )
    plt.title(
        "XGBoost Probabilistic Weather Forecast with Uncertainty Bands", fontsize=12
    )
    plt.ylabel("Temperature (°C)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
