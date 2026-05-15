"""
Plotting utilities for the paper figures.

Figures produced here:
    - histograms of selected p, d, q (Figs. 5, 6, 7)
    - boxplots of NMAE, NRMSE, PBIAS (Figs. 8, 9, 10)
    - heatmaps of wins on all series and on non-stationary subset
      (Figs. 11, 12)
    - example for the Capivara reservoir: series, ACF/PACF, forecasts
      (Figs. 13, 14, 15, 16)
"""

import ast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.analysis import (
    METHOD_LABELS,
    METHOD_SUFFIXES,
    ORDER_COLUMNS,
    count_wins,
)

# Conversion factor used in figsize tuples
CM = 1 / 2.54

# Color palette used in the paper figures
BOXPLOT_FILL = "#5e81a5"
MEDIAN_GREEN = "#2CA751"
PBIAS_ZERO_LINE = "#BE2C2C"
LINE_HISTORICAL = "#9cbcdb"
FORECAST_COLORS = {
    "AIC": "#6e8cce",
    "BIC": "#ce9f62",
    "Rep-holdout": "#59b86e",
    "TSCV": "#b64747",
    "Traditional": "#bf77d4",
}


# ---------------------------------------------------------------------------
# rcParams overrides per figure
#
# Each function applies its dict together with seaborn's ``whitegrid`` style
# via stacked context managers, so every plotting function is self-contained
# and doesn't depend on any global configuration being set beforehand.
# ---------------------------------------------------------------------------

# Histograms (Figs. 5, 6, 7).
_RC_HISTOGRAM = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

# Boxplots (Figs. 8, 9, 10): ticks bumped to 10pt.
_RC_BOXPLOT = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}

# Wins heatmaps (Figs. 11, 12).
_RC_HEATMAP = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# Capivara series and forecasts (Figs. 13, 15, 16): smaller labels/legend.
_RC_CAPIVARA = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}

# ACF / PACF (Fig. 14): small title and ticks.
_RC_ACF_PACF = {
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


# ---------------------------------------------------------------------------
# Histograms of selected components (Figs. 5, 6, 7)
# ---------------------------------------------------------------------------


def _extract_component(order_str, idx):
    """Extract the p (idx=0), d (idx=1), or q (idx=2) component."""
    return ast.literal_eval(order_str)[idx]


def plot_component_histogram(df, component_idx, values, out_path):
    """Histogram of selected component values across the four methods."""
    with sns.axes_style("whitegrid"), mpl.rc_context(_RC_HISTOGRAM):
        fig, ax = plt.subplots(figsize=(17.4 * CM, 6.5 * CM),
                               layout="constrained")
        x = np.arange(len(METHOD_LABELS))
        n_vals = len(values)
        width = 0.8 / n_vals

        for j, val in enumerate(values):
            counts = []
            for col in ORDER_COLUMNS:
                n = (
                    df[col].apply(
                        lambda s, i=component_idx: _extract_component(s, i)
                    )
                    == val
                ).sum()
                counts.append(n)
            ax.bar(
                x + j * width,
                counts,
                width,
                label=str(val),
                edgecolor="black",
                alpha=0.85,
                color=plt.cm.PuBuGn(j / n_vals),
            )

        ax.set_xlabel("Selection method")
        ax.set_ylabel("Number of selections")
        ax.set_xticks(x + (n_vals - 1) * width / 2)
        ax.set_xticklabels(METHOD_LABELS)
        ax.legend()

        for container in ax.containers:
            ax.bar_label(container, fontsize=8)

        fig.savefig(out_path, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Boxplots (Figs. 8, 9, 10)
# ---------------------------------------------------------------------------


def plot_boxplot(df, metric, ylabel, out_path,
                 draw_zero_line=False):
    """Single boxplot of one metric across the four methods.
    """
    data = pd.DataFrame(
        {
            METHOD_LABELS[i]: df[f"{metric}_{METHOD_SUFFIXES[i]}"]
            for i in range(4)
        }
    )

    with sns.axes_style("whitegrid"), mpl.rc_context(_RC_BOXPLOT):
        fig, ax = plt.subplots(figsize=(17.4 * CM, 6 * CM),
                               layout="constrained")
        sns.boxplot(
            data=data,
            ax=ax,
            color=BOXPLOT_FILL,
            linecolor="black",
            boxprops={"edgecolor": "black", "linewidth": 1},
            whiskerprops={"color": "gray"},
            capprops={"color": "gray"},
            medianprops={"color": MEDIAN_GREEN, "linewidth": 1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "gray",
                "markersize": 3,
            },
        )
        ax.set_ylabel(ylabel)
        ax.grid(False)
        if draw_zero_line:
            ax.axhline(y=0, color=PBIAS_ZERO_LINE, linestyle="--", alpha=0.7)

        fig.savefig(out_path, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Wins heatmap (Figs. 11, 12)
# ---------------------------------------------------------------------------


def plot_wins_heatmap(df, out_path):
    """Heatmap of wins per metric per method."""
    wins = count_wins(df)
    wins_df = pd.DataFrame(wins).rename(
        index={"aic": "AIC", "bic": "BIC", "val": "Rep-holdout", "cv": "TSCV"},
        columns={"nmae": "NMAE", "nrmse": "NRMSE", "pbias": "PBIAS"},
    )

    n_total = len(df)
    annot = wins_df.apply(
        lambda col: col.map(lambda v: f"{v}\n({v / n_total * 100:.0f}%)")
    )

    with sns.axes_style("whitegrid"), mpl.rc_context(_RC_HEATMAP):
        fig, ax = plt.subplots(figsize=(17.4 * CM, 8 * CM),
                               layout="constrained")
        sns.heatmap(
            wins_df,
            annot=annot,
            fmt="",
            cmap="PuBuGn",
            annot_kws={"fontsize": 10},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "Wins"},
        )
        ax.set_ylabel("")
        ax.set_xlabel("")

        fig.savefig(out_path, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Capivara example (Figs. 13, 14, 15, 16)
# ---------------------------------------------------------------------------


def plot_capivara_series(series, out_path):
    """Plot the full historical streamflow series of the Capivara reservoir."""
    with sns.axes_style("whitegrid"), mpl.rc_context(_RC_CAPIVARA):
        fig, ax = plt.subplots(figsize=(17.4 * CM, 6 * CM),
                               layout="constrained")

        ax.plot(series.index, series.values, color=LINE_HISTORICAL,
                linewidth=0.4, alpha=0.85)
        ax.set_ylabel("Natural inflow (m³/s)")
        ax.grid(True, linewidth=0.4, color="gray", linestyle="--", alpha=0.25)
        for spine in ["top", "bottom", "right", "left"]:
            ax.spines[spine].set_visible(False)

        fig.savefig(out_path, dpi=300)
    return fig


def plot_acf_pacf(series, out_path, title_acf="ACF", title_pacf="PACF"):
    """Plot ACF and PACF side by side for one series."""
    with sns.axes_style("whitegrid"), mpl.rc_context(_RC_ACF_PACF):
        fig, axes = plt.subplots(1, 2, figsize=(17.4 * CM, 4.5 * CM),
                                 layout="constrained")

        plot_acf(
            series,
            lags=40,
            ax=axes[0],
            title=title_acf,
            markersize=2,
            vlines_kwargs={"linewidth": 0.6, "colors": BOXPLOT_FILL},
            markerfacecolor=BOXPLOT_FILL,
            markeredgecolor=BOXPLOT_FILL,
        )
        plot_pacf(
            series,
            lags=40,
            ax=axes[1],
            title=title_pacf,
            method="ywm",
            markersize=2,
            vlines_kwargs={"linewidth": 0.6, "colors": BOXPLOT_FILL},
            markerfacecolor=BOXPLOT_FILL,
            markeredgecolor=BOXPLOT_FILL,
        )

        for ax in axes:
            ax.grid(False)
            ax.grid(True, linewidth=0.4, color="gray", linestyle="--", alpha=0.4)
            for coll in ax.collections:
                coll.set_facecolor(BOXPLOT_FILL)
                coll.set_alpha(0.2)
            for line in ax.lines:
                if list(line.get_ydata()) == [0, 0]:
                    line.set_color(BOXPLOT_FILL)
                    line.set_linewidth(0.5)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.savefig(out_path, dpi=300)
    return fig


def plot_capivara_forecasts(test_series, forecasts, orders, out_path, n_zoom=180):
    """Plot the observed test series vs. forecasts from each approach."""
    with sns.axes_style("whitegrid"), mpl.rc_context(_RC_CAPIVARA):
        fig, ax = plt.subplots(figsize=(17.4 * CM, 6 * CM),
                               layout="constrained")

        ax.plot(
            test_series.index[:n_zoom],
            test_series.values[:n_zoom],
            color="black",
            linewidth=0.4,
            label="Observed",
        )
        for label, pred in forecasts.items():
            ax.plot(
                test_series.index[:n_zoom],
                pred[:n_zoom],
                color=FORECAST_COLORS.get(label, "gray"),
                linewidth=0.4,
                alpha=1,
                label=f"{label} {orders[label]}",
            )
        ax.set_ylabel("Natural inflow (m³/s)")
        ax.legend(loc="upper right")
        ax.grid(True, linewidth=0.4, color="gray", linestyle="--", alpha=0.25)
        for spine in ["top", "bottom", "right", "left"]:
            ax.spines[spine].set_visible(False)

        fig.savefig(out_path, dpi=300)
    return fig
