from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb


def train_quantile_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantiles: List[float] = [0.05, 0.50, 0.95],
    random_state: int = 42,
) -> Dict[float, xgb.XGBRegressor]:
    """Trains separate XGBoost models under pinball loss with L2 regularization."""
    models = {}
    for q in quantiles:
        reg = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=6,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            random_state=random_state,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models[q] = reg
    return models


def reconstruct_forecasts(
    models: Dict[float, xgb.XGBRegressor],
    X_test: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    X_val: pd.DataFrame,
    y_val_actual: pd.Series,
    delta_shrinkage: float = 0.65,
    coverage_target: float = 0.80,
) -> Tuple[Dict[float, pd.Series], float, float]:
    """
    Reconstructs temperature levels with:
      1. Non-negative CQR interval calibration on P05/P95.
      2. Post-shrinkage validation bias correction on P50.
    """
    # 1. Validation Calibration: Conformal Intervals & Empirical Bias
    val_deltas = {q: models[q].predict(X_val) for q in [0.05, 0.50, 0.95]}
    val_matrix = np.sort(
        np.vstack([val_deltas[0.05], val_deltas[0.50], val_deltas[0.95]]), axis=0
    )
    val_p05 = X_val["temp_lag_1"] + val_matrix[0]
    val_p95 = X_val["temp_lag_1"] + val_matrix[2]

    # Non-conformity scores
    nonconformity_scores = np.maximum(
        val_p05 - y_val_actual.values, y_val_actual.values - val_p95
    )
    n_val = len(y_val_actual)
    conf_level = min(1.0, np.ceil((n_val + 1) * coverage_target) / n_val)
    raw_q_conf = float(np.quantile(nonconformity_scores, conf_level, method="higher"))
    q_conf = max(0.0, raw_q_conf)

    # Compute validation bias caused by delta shrinkage to recenter test forecasts
    val_p50_shrunk = X_val["temp_lag_1"] + (val_matrix[1] * delta_shrinkage)
    val_bias = float(np.mean(val_p50_shrunk - y_val_actual.values))

    # 2. Test Reconstruction
    test_deltas = {q: models[q].predict(X_test) for q in [0.05, 0.50, 0.95]}
    test_matrix = np.sort(
        np.vstack([test_deltas[0.05], test_deltas[0.50], test_deltas[0.95]]), axis=0
    )

    # Calibrate outer bounds
    p10_calibrated = (X_test["temp_lag_1"] + test_matrix[0]) - q_conf
    p90_calibrated = (X_test["temp_lag_1"] + test_matrix[2]) + q_conf

    # Dampen median point forecast toward persistence and remove calibration drift
    p50_shrunk = (X_test["temp_lag_1"] + (test_matrix[1] * delta_shrinkage)) - val_bias
    p50_shrunk = np.clip(p50_shrunk, p10_calibrated, p90_calibrated)

    reconstructed = {
        0.10: pd.Series(p10_calibrated, index=test_index),
        0.50: pd.Series(p50_shrunk, index=test_index),
        0.90: pd.Series(p90_calibrated, index=test_index),
    }

    return reconstructed, q_conf, val_bias
