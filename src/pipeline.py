"""
Multi-reservoir ARIMA pipeline.

Translated from the experiment notebook (functions `pipeline_arima` and
`_processar_reservatorio`). Selects ARIMA orders by all four methods,
fits each on the full training set, and computes metrics on the test set.

The "val" suffix in the result keys refers to Rep-holdout VALidation.
"""

import gc

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from src.selection import (
    compute_metrics,
    search_aic_bic,
    search_rep_holdout,
    search_time_series_cv,
)


def arima_pipeline(
    series,
    range_p,
    range_d,
    range_q,
    n_splits_cv=10,
    n_reps_holdout=10,
):
    """Full pipeline: 85/15 split, grid search, forecast and metrics."""

    n = len(series)
    n_test = round(n * 0.15)

    train_pipe = series.iloc[:-n_test]
    test_pipe = series.iloc[-n_test:]

    # --- approach 1: AIC/BIC ---
    results_1 = search_aic_bic(
        train_pipe,
        range_p,
        range_d,
        range_q,
        verbose=False,
        keep_model=False,
    )
    if len(results_1) == 0:
        return None

    best_aic = min(results_1, key=lambda x: x["aic"])
    best_bic = min(results_1, key=lambda x: x["bic"])
    order_aic = best_aic["order"]
    order_bic = best_bic["order"]

    del results_1
    gc.collect()

    # AIC forecasts
    try:
        model_aic = ARIMA(train_pipe, order=order_aic).fit(low_memory=True)
        ext_aic = model_aic.append(test_pipe, refit=False)
        pred_aic = np.array(ext_aic.fittedvalues[-len(test_pipe) :])
        met_aic = compute_metrics(test_pipe.values, pred_aic)
        del model_aic, ext_aic, pred_aic
    except Exception:
        met_aic = None

    # BIC forecasts
    if order_bic != order_aic:
        try:
            model_bic = ARIMA(train_pipe, order=order_bic).fit(low_memory=True)
            ext_bic = model_bic.append(test_pipe, refit=False)
            pred_bic = np.array(ext_bic.fittedvalues[-len(test_pipe) :])
            met_bic = compute_metrics(test_pipe.values, pred_bic)
            del model_bic, ext_bic, pred_bic
        except Exception:
            met_bic = met_aic
    else:
        met_bic = met_aic

    # --- approach 2: repeated holdout (Cerqueira et al., 2020) ---
    results_val = search_rep_holdout(
        train_pipe,
        range_p,
        range_d,
        range_q,
        n_reps=n_reps_holdout,
        verbose=False,
    )

    if len(results_val) == 0:
        return None

    order_val = results_val[0]["order"]

    del results_val
    gc.collect()

    # --- approach 3: time series cross-validation (Preq-Bls in Cerqueira et al., 2020) ---
    results_cv = search_time_series_cv(
        train_pipe,
        range_p,
        range_d,
        range_q,
        n_splits=n_splits_cv,
        verbose=False,
    )

    if len(results_cv) == 0:
        return None

    order_cv = results_cv[0]["order"]

    del results_cv
    gc.collect()

    # Rep-Holdout forecasts
    try:
        model_val = ARIMA(train_pipe, order=order_val).fit(low_memory=True)
        ext_val = model_val.append(test_pipe, refit=False)
        pred_val = np.array(ext_val.fittedvalues[-len(test_pipe) :])
        met_val = compute_metrics(test_pipe.values, pred_val)
        del model_val, ext_val, pred_val
    except Exception:
        met_val = None

    # TSCV forecasts
    try:
        model_cv = ARIMA(train_pipe, order=order_cv).fit(low_memory=True)
        ext_cv = model_cv.append(test_pipe, refit=False)
        pred_cv = np.array(ext_cv.fittedvalues[-len(test_pipe) :])
        met_cv = compute_metrics(test_pipe.values, pred_cv)
        del model_cv, ext_cv, pred_cv
    except Exception:
        met_cv = None

    if met_aic is None or met_cv is None or met_val is None:
        return None

    return {
        "ordem_aic": str(order_aic),
        "ordem_bic": str(order_bic),
        "ordem_val": str(order_val),
        "ordem_cv": str(order_cv),
        "mae_aic": met_aic["MAE"],
        "rmse_aic": met_aic["RMSE"],
        "pbias_aic": met_aic["PBIAS"],
        "mae_bic": met_bic["MAE"],
        "rmse_bic": met_bic["RMSE"],
        "pbias_bic": met_bic["PBIAS"],
        "mae_val": met_val["MAE"],
        "rmse_val": met_val["RMSE"],
        "pbias_val": met_val["PBIAS"],
        "mae_cv": met_cv["MAE"],
        "rmse_cv": met_cv["RMSE"],
        "pbias_cv": met_cv["PBIAS"],
        "n_obs": n,
    }


def process_reservoir(name, series, range_p, range_d, range_q):
    """Worker function for parallel execution via joblib."""
    result = arima_pipeline(
        series, range_p, range_d, range_q
    )
    if result is not None:
        result["reservatorio"] = name
    return result
