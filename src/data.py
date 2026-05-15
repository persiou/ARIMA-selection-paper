"""
ONS streamflow series.

"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def prepare_daily_series(df, reservoir_name):
    """Prepare daily streamflow series for one reservoir."""
    df_res = df[df["nom_reservatorio"] == reservoir_name].copy()
    df_res["din_instante"] = pd.to_datetime(df_res["din_instante"])
    df_res["val_vazaonatural"] = pd.to_numeric(
        df_res["val_vazaonatural"], errors="coerce"
    )
    df_res = df_res.dropna(subset=["val_vazaonatural"])

    if len(df_res) == 0:
        return None

    df_res = df_res.set_index("din_instante").sort_index()
    series = df_res["val_vazaonatural"].copy()
    series = series[~series.index.duplicated(keep="first")]
    series = series.asfreq("D")
    if series.isna().any():
        return None
    return series


def classify_stationarity(prepared_series, alpha=0.05):
    """
    Classify each series jointly using the ADF and KPSS tests.
    """
    results = []
    for name, series in prepared_series.items():
        # ADF: H0 = unit root (non-stationary)
        adf_res = adfuller(series, autolag="AIC")
        adf_stat, adf_p = adf_res[0], adf_res[1]
        adf_rejects = adf_p < alpha

        # KPSS: H0 = stationary
        kpss_res = kpss(series, regression="c", nlags="auto")
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
        kpss_rejects = kpss_p < alpha

        if adf_rejects and not kpss_rejects:
            classification = "Estacionaria"
        elif not adf_rejects and kpss_rejects:
            classification = "Nao estacionaria"
        elif adf_rejects and kpss_rejects:
            classification = "Nao estacionaria no kpss"
        elif not adf_rejects and not kpss_rejects:
            classification = "Nao estacionaria no adf"


        results.append(
            {
                "reservatorio": name,
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "adf_rejeita": adf_rejects,
                "kpss_stat": kpss_stat,
                "kpss_p": kpss_p,
                "kpss_rejeita": kpss_rejects,
                "classificacao": classification,
            }
        )

    return pd.DataFrame(results).sort_values("classificacao")
