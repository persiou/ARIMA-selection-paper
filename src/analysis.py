"""
Aggregation, agreement and statistical tests for comparing multiple models across many series. 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, studentized_range

# Display labels and column suffixes
METHOD_LABELS = ["AIC", "BIC", "Rep-holdout", "TSCV"]
METHOD_SUFFIXES = ["aic", "bic", "val", "cv"]
LABEL_TO_SUFFIX = dict(zip(METHOD_LABELS, METHOD_SUFFIXES))
ORDER_COLUMNS = [f"ordem_{s}" for s in METHOD_SUFFIXES]


def normalize_metrics(df_results, prepared_series, test_fraction=0.15):
    """
    Add NMAE and NRMSE columns by dividing MAE and RMSE by the mean of
    the test set observed values for each reservoir.
    """
    df = df_results.copy()

    mean_test_map = {}
    for name, series in prepared_series.items():
        n = len(series)
        n_test = round(n * test_fraction)
        test = series.iloc[-n_test:]
        mean_test_map[name] = test.mean()

    df["mean_teste"] = df["reservatorio"].map(mean_test_map)

    for suffix in METHOD_SUFFIXES:
        df[f"nmae_{suffix}"] = df[f"mae_{suffix}"] / df["mean_teste"]
        df[f"nrmse_{suffix}"] = df[f"rmse_{suffix}"] / df["mean_teste"]

    return df


def remove_outliers(df, threshold=2.0):
    """
    Detect optimization failures: any approach whose MAE exceeds
    `threshold` times the minimum MAE across approaches.
    """
    mae_cols = [f"mae_{s}" for s in METHOD_SUFFIXES]
    min_mae = df[mae_cols].min(axis=1)
    mask = pd.Series(False, index=df.index)
    for col in mae_cols:
        mask |= df[col] > threshold * min_mae
    return df[~mask].copy(), df[mask].copy()


def count_wins(df, metrics=("nmae", "nrmse", "pbias")):
    """
    Count wins per approach per metric. Ties are credited to all tied
    approaches. For PBIAS, the absolute value is used.

    Returns
    -------
    dict
        {metric: {suffix: count}}
    """
    wins = {m: {s: 0 for s in METHOD_SUFFIXES} for m in metrics}

    for _, row in df.iterrows():
        for metric in metrics:
            values = {s: row[f"{metric}_{s}"] for s in METHOD_SUFFIXES}
            if metric == "pbias":
                lowest = min(abs(v) for v in values.values())
                for s, v in values.items():
                    if abs(v) == lowest:
                        wins[metric][s] += 1
            else:
                lowest = min(values.values())
                for s, v in values.items():
                    if v == lowest:
                        wins[metric][s] += 1

    return wins


def agreement_matrix(df):
    """Pairwise agreement matrix on the selected (p, d, q) orders."""
    matrix = np.zeros((4, 4), dtype=int)
    for i, ci in enumerate(ORDER_COLUMNS):
        for j, cj in enumerate(ORDER_COLUMNS):
            matrix[i, j] = (df[ci] == df[cj]).sum()
    return pd.DataFrame(matrix, index=METHOD_LABELS, columns=METHOD_LABELS)


def metric_frame(df, metric):
    """
    Build a wide dataframe (rows = reservoirs, columns = method labels)
    for a single metric. For PBIAS, the absolute value is used.
    """
    data = pd.DataFrame(
        {
            METHOD_LABELS[i]: df[f"{metric}_{METHOD_SUFFIXES[i]}"]
            for i in range(4)
        }
    )
    if metric == "pbias":
        data = data.abs()
    return data


def friedman_nemenyi_tables(df_sub):
    """
    Run Friedman + Nemenyi for NMAE, NRMSE, and |PBIAS| and return:
        - df_friedman: summary table of Friedman test
        - df_nemenyi: table of pairwise p-values from Nemenyi

    """
    import scikit_posthocs as sp

    friedman_rows = []
    nemenyi_rows = []

    for metric_name, col_base in [
        ("NMAE", "nmae"),
        ("NRMSE", "nrmse"),
        ("|PBIAS|", "pbias"),
    ]:
        data = pd.DataFrame(
            {
                m: df_sub[f"{col_base}_{LABEL_TO_SUFFIX[m]}"]
                for m in METHOD_LABELS
            }
        )
        if metric_name == "|PBIAS|":
            data = data.abs()

        cols = data.columns.tolist()
        stat, p = friedmanchisquare(*[data[c].values for c in cols])
        friedman_rows.append(
            {"Metric": metric_name, "chi2": f"{stat:.2f}", "p-value": f"{p:.4e}"}
        )

        # Nemenyi post-hoc: all pairs
        nemenyi = sp.posthoc_nemenyi_friedman(data.values)
        nemenyi.index = cols
        nemenyi.columns = cols

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pv = nemenyi.iloc[i, j]
                sig = (
                    "***" if pv < 0.001
                    else "**" if pv < 0.01
                    else "*" if pv < 0.05
                    else "ns"
                )
                nemenyi_rows.append(
                    {
                        "Metric": metric_name,
                        "Pair": f"{cols[i]} vs {cols[j]}",
                        "p-value": f"{pv:.4f}",
                        "Sig.": sig,
                    }
                )

    return pd.DataFrame(friedman_rows), pd.DataFrame(nemenyi_rows)


def mean_ranks(data):
    """Mean rank per method, with rank 1 = best."""
    matrix = data.values
    ranks = np.apply_along_axis(rankdata, 1, matrix)
    return pd.Series(ranks.mean(axis=0), index=data.columns)
