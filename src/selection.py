"""
Evaluation metrics and ARIMA order selection methods.

"""

import gc
from itertools import product

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def compute_metrics(actual, predicted):
    """Compute MAE, RMSE, and PBIAS."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    pbias = np.sum(predicted - actual) / np.sum(actual) * 100

    return {"MAE": mae, "RMSE": rmse, "PBIAS": pbias}


def search_aic_bic(
    train_series,
    range_p,
    range_d,
    range_q,
    verbose=True,
    keep_model=True,
):
    """Grid search returning AIC and BIC for each (p, d, q) combination."""
    results = []
    total = sum(1 for p, d, q in product(range_p, range_d, range_q) if not (p == 0 and q == 0))

    counter = 0
    for p, d, q in product(range_p, range_d, range_q):

        if p == 0 and q == 0:
            continue

        counter += 1
        try:
            model = ARIMA(train_series, order=(p, d, q))
            fitted = model.fit(low_memory=True)

            results.append(
                {
                    "order": (p, d, q),
                    "aic": fitted.aic,
                    "bic": fitted.bic,
                    "fitted_model": fitted if keep_model else None,
                }
            )

            if not keep_model:
                del fitted
            del model

        except Exception:
            continue
        finally:
            if counter % 10 == 0:
                gc.collect()

        if verbose and counter % 20 == 0:
            print(f"  Progress: {counter}/{total} combinations tested...")

    if verbose:
        print(f"  Done: {len(results)} models fitted out of {total} attempted.")

    return results


def search_rep_holdout(
    train_series,
    range_p,
    range_d,
    range_q,
    n_reps=10,
    use_fraction=0.70,
    inner_train_fraction=6 / 7,
    random_state=10,
    verbose=True
):
    """
    Rep-Holdout (Cerqueira et al., 2020) or repeated Monte Carlo holdout.

    In each repetition, a random starting point is drawn and a window of
    `use_fraction` (70%) of the data is taken, split into training
    (~60% of t) and validation (~10% of t). The final score is the mean
    RMSE across the `n_reps` repetitions.
    """
    n = len(train_series)
    window_size = round(n * use_fraction)
    n_train_rep = round(window_size * inner_train_fraction)
    n_val_rep = window_size - n_train_rep
    max_start = n - window_size

    if max_start < 0:
        if verbose:
            print("  Rep-Holdout: series too short for the chosen parameters.")
        return []

    # Draw all starting points before the candidate loop so that every
    # candidate is evaluated on the same windows.
    rng = np.random.default_rng(random_state)
    starts = rng.integers(0, max_start + 1, size=n_reps)

    if verbose:
        print(
            f"  Rep-Holdout: {n_reps} reps, window={window_size} obs "
            f"(train={n_train_rep}, val={n_val_rep})"
        )

    combinations = [
        (p, d, q)
        for p, d, q in product(range_p, range_d, range_q)
        if not (p == 0 and q == 0)
    ]
    total = len(combinations)
    results = []
    best_rmse = float("inf")

    for idx, (p, d, q) in enumerate(combinations, 1):
        rep_rmses = []

        for a in starts:
            train_rep = train_series.iloc[a : a + n_train_rep]
            val_rep = train_series.iloc[a + n_train_rep : a + window_size]

            try:
                model = ARIMA(train_rep, order=(p, d, q)).fit(low_memory=True)
                ext = model.append(val_rep, refit=False)
                pred = np.array(ext.fittedvalues[-len(val_rep) :])
                rmse_rep = np.sqrt(mean_squared_error(val_rep.values, pred))
                rep_rmses.append(rmse_rep)
                del ext, pred, model
            except Exception:
                break

        if len(rep_rmses) == n_reps:
            r = {
                "order": (p, d, q),
                "rmse_val": np.mean(rep_rmses),
                "rmse_std": np.std(rep_rmses),
            }
            results.append(r)
            best_rmse = min(best_rmse, r["rmse_val"])

        if idx % 10 == 0:
            gc.collect()

        if verbose and idx % 20 == 0:
            print(f"  Progress: {idx}/{total} combinations tested...")

    results.sort(key=lambda x: x["rmse_val"])

    if verbose:
        print(f"  Done: {len(results)} models evaluated out of {total} attempted.")

    return results


def search_time_series_cv(
    train_series,
    range_p,
    range_d,
    range_q,
    n_splits=10,
    verbose=True
):
    """
    Time series cross-validation with expanding window (Preq-Bls in
    Cerqueira et al., 2020).

    The data are divided into `n_splits` contiguous blocks. For each
    fold k (k=1..K-1), training uses blocks 0..k-1 (expanding window)
    and validation uses block k.
    """
    n = len(train_series)
    block_size = n // n_splits

    # Define contiguous blocks
    blocks = []
    for k in range(n_splits):
        start = k * block_size
        end = start + block_size if k < n_splits - 1 else n
        blocks.append((start, end))

    # Generate folds with expanding window
    folds = []
    for k in range(1, n_splits):
        train_start = blocks[0][0]
        train_end = blocks[k - 1][1]
        test_start = blocks[k][0]
        test_end = blocks[k][1]

        train_idx = list(range(train_start, train_end))
        val_idx = list(range(test_start, test_end))
        folds.append((train_idx, val_idx))

    n_folds = len(folds)

    if verbose:
        print(
            f"  Time series CV: block_size={block_size} obs, "
            f"{n_folds} folds (expanding window)"
        )

    combinations = [
        (p, d, q)
        for p, d, q in product(range_p, range_d, range_q)
        if not (p == 0 and q == 0)
    ]
    total = len(combinations)

    results = []
    best_rmse = float("inf")

    for idx, (p, d, q) in enumerate(combinations, 1):
        fold_rmses = []

        for train_idx, val_idx in folds:
            train_fold = train_series.iloc[train_idx]
            val_fold = train_series.iloc[val_idx]

            try:
                model = ARIMA(train_fold, order=(p, d, q)).fit(low_memory=True)
                ext = model.append(val_fold, refit=False)
                pred = np.array(ext.fittedvalues[-len(val_fold) :])
                rmse_fold = np.sqrt(mean_squared_error(val_fold.values, pred))
                fold_rmses.append(rmse_fold)
                del ext, pred, model
            except Exception:
                break

        if len(fold_rmses) == n_folds:
            r = {
                "order": (p, d, q),
                "rmse_cv": np.mean(fold_rmses),
                "rmse_std": np.std(fold_rmses),
                "rmses_folds": fold_rmses,
            }
            results.append(r)
            best_rmse = min(best_rmse, r["rmse_cv"])

        if idx % 10 == 0:
            gc.collect()

        if verbose and idx % 20 == 0:
            print(f"  Progress: {idx}/{total} combinations tested...")

    results.sort(key=lambda x: x["rmse_cv"])

    if verbose:
        print(f"  Done: {len(results)} models evaluated out of {total} attempted.")

    return results
