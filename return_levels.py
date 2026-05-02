import numpy as np
import pandas as pd

GEV_EPS = 1e-6

def gev_return_level(T, mu, sigma, xi, eps=GEV_EPS):
    """
    Deterministic GEV return level (annual maxima / block maxima).

    Parameters
    ----------
    T : float or array_like
        Return period in years (T > 1).
    mu, sigma, xi : float or array_like
        GEV parameters:
          mu    : location
          sigma : scale (must be > 0)
          xi    : shape
    eps : float
        Threshold for using the xi -> 0 (Gumbel) limit.

    Returns
    -------
    x_T : ndarray
        Return level corresponding to return period T.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T <= 1.0):
        raise ValueError("GEV return period T must be > 1.")

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be > 0.")

    p = 1.0 - 1.0 / T
    p = np.clip(p, 1e-12, 1.0 - 1e-12)

    term = -np.log(p)

    # xi != 0
    x_xi = mu + sigma * ((term ** (-xi) - 1.0) / xi)

    # xi -> 0 (Gumbel)
    x_gumbel = mu - sigma * np.log(term)

    return np.where(np.abs(xi) < eps, x_gumbel, x_xi)

def exceedance_rate_per_year(series, u):
    """
    series : pd.Series with datetime-like index (daily rainfall)
    u : float threshold

    Returns
    -------
    lam : float
        exceedances per year above u
    """
    s = series.dropna()
    if s.empty:
        return np.nan

    years = pd.to_datetime(s.index).year
    n_years = years.nunique()
    if n_years <= 0:
        return np.nan

    n_exc = int((s.to_numpy(float) > float(u)).sum())
    return n_exc / n_years

GPD_EPS = 1e-6

def gpd_return_level(T, u, sigma, xi, lam_per_year, eps=GPD_EPS):
    """
    Deterministic POT (GPD) return level.

    Parameters
    ----------
    T : float or array_like
        Return period in years (T > 0).
    u : float or array_like
        Threshold used for POT (same units as rainfall).
    sigma, xi : float or array_like
        GPD parameters for exceedances y = x - u >= 0.
        sigma must be > 0.
    lam_per_year : float
        Exceedance rate lambda = expected exceedances per year above u.
        (e.g., n_exceed / n_years)
    eps : float
        Threshold for using the xi -> 0 (exponential) limit.

    Returns
    -------
    z_T : ndarray
        Return level in the original rainfall units.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("POT return period T must be > 0.")

    u = np.asarray(u, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    xi = np.asarray(xi, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be > 0.")
    if not np.isfinite(lam_per_year) or lam_per_year <= 0.0:
        raise ValueError("lam_per_year must be > 0.")

    a = lam_per_year * T  # expected exceedances in T years

    # xi != 0
    z_xi = u + (sigma / xi) * (a ** xi - 1.0)

    # xi -> 0 (exponential)
    z_exp = u + sigma * np.log(a)

    return np.where(np.abs(xi) < eps, z_exp, z_xi)


def gev_return_levels_summary(T, mu_q, sigma_q, xi_q):
    """
    mu_q/sigma_q/xi_q: dict like {"mean":..., "p05":..., "p95":...}
    Returns dict of return levels for each key.
    """
    out = {}
    for k in mu_q.keys():
        out[k] = gev_return_level(T, mu_q[k], sigma_q[k], xi_q[k])
    return out

def gpd_return_levels_summary(T, u, sigma_q, xi_q, lam_per_year):
    """
    sigma_q/xi_q: dict like {"mean":..., "p05":..., "p95":...}
    Returns dict of return levels for each key.
    """
    out = {}
    for k in sigma_q.keys():
        out[k] = gpd_return_level(T, u, sigma_q[k], xi_q[k], lam_per_year)
    return out

def gev_design_from_summary(param_summary, station_id, T, quant="mean"):
    """
    quant: "mean", "5", or "95"
    """
    row = param_summary.loc[param_summary["station_id"] == station_id].iloc[0]
    if quant == "mean":
        mu, sigma, xi = row["gev_mu"], row["gev_sigma"], row["gev_xi"]
    elif quant == "5":
        mu, sigma, xi = row["gev_mu_5"], row["gev_sigma_5"], row["gev_xi_5"]
    elif quant == "95":
        mu, sigma, xi = row["gev_mu_95"], row["gev_sigma_95"], row["gev_xi_95"]
    else:
        raise ValueError("quant must be 'mean', '5', or '95'")

    return gev_return_level(T, mu, sigma, xi)

def gpd_design_from_summary(param_summary, station_id, T, u, lam_per_year, quant="mean"):
    """
    quant: "mean", "5", or "95"
    """
    row = param_summary.loc[param_summary["station_id"] == station_id].iloc[0]
    if quant == "mean":
        sigma, xi = row["gpd_sigma"], row["gpd_xi"]
    elif quant == "5":
        sigma, xi = row["gpd_sigma_5"], row["gpd_xi_5"]
    elif quant == "95":
        sigma, xi = row["gpd_sigma_95"], row["gpd_xi_95"]
    else:
        raise ValueError("quant must be 'mean', '5', or '95'")

    return gpd_return_level(T, u=u, sigma=sigma, xi=xi, lam_per_year=lam_per_year)

def gev_return_levels_from_posterior(
    T,
    posterior,
    *,
    mu_col="mu",
    log_sigma_col="log_sigma",
    xi_col="xi",
    quantiles=(0.05, 0.5, 0.95),
    return_draws=False,
):
    """
    Compute GEV return levels for each posterior draw, then summarize across draws.

    Parameters
    ----------
    T : float or array_like
        Return period(s) in years (T > 1).
    posterior : pandas.DataFrame or dict-like
        Must provide arrays for mu, log_sigma, xi.
        Each row = one posterior draw.
    mu_col, log_sigma_col, xi_col : str
        Column names (if posterior is a DataFrame).
    quantiles : tuple
        Quantiles to report (e.g., (0.05, 0.95)).
    return_draws : bool
        If True, also return the full matrix of per-draw return levels.

    Returns
    -------
    summary : pandas.DataFrame
        Rows = return periods T
        Columns = ["mean", "median", "q05", "q95", ...] depending on quantiles.
    draws (optional) : ndarray
        Shape (n_draws, n_T), per-draw return levels.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T <= 1.0):
        raise ValueError("GEV return period T must be > 1.")

    if isinstance(posterior, pd.DataFrame):
        mu = posterior[mu_col].to_numpy(float)
        sigma = np.exp(posterior[log_sigma_col].to_numpy(float))
        xi = posterior[xi_col].to_numpy(float)
    else:
        mu = np.asarray(posterior[mu_col], dtype=float)
        sigma = np.exp(np.asarray(posterior[log_sigma_col], dtype=float))
        xi = np.asarray(posterior[xi_col], dtype=float)

    xT = gev_return_level(T[None, :], mu[:, None], sigma[:, None], xi[:, None])  # (n_draws, n_T)

    out = {
        "T": T,
        "mean": np.mean(xT, axis=0),
        "std":  np.std(xT, axis=0, ddof=1),
        "median": np.quantile(xT, 0.5, axis=0),
    }
    for q in quantiles:
        out[f"q{int(round(100*q)):02d}"] = np.quantile(xT, q, axis=0)

    summary = pd.DataFrame(out)
    return (summary, xT) if return_draws else summary

def gpd_return_levels_from_posterior(
    T,
    posterior,
    *,
    u,
    lam_per_year,
    log_sigma_col="log_sigma",
    xi_col="xi",
    quantiles=(0.05, 0.5, 0.95),
    return_draws=False,
):
    """
    Compute POT/GPD return levels for each posterior draw, then summarize across draws.

    Parameters
    ----------
    T : float or array_like
        Return period(s) in years (T > 0).
    posterior : pandas.DataFrame or dict-like
        Must provide arrays for log_sigma and xi.
        Each row = one posterior draw.
    u : float
        POT threshold used for exceedances.
    lam_per_year : float
        Exceedance rate above u: expected exceedances per year.
    log_sigma_col, xi_col : str
        Column names (if posterior is a DataFrame).
    quantiles : tuple
        Quantiles to report.
    return_draws : bool
        If True, also return the full matrix of per-draw return levels.

    Returns
    -------
    summary : pandas.DataFrame
        Rows = return periods T
        Columns = ["mean", "median", "q05", "q95", ...]
    draws (optional) : ndarray
        Shape (n_draws, n_T), per-draw return levels.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("POT return period T must be > 0.")
    if not np.isfinite(lam_per_year) or lam_per_year <= 0.0:
        raise ValueError("lam_per_year must be > 0.")

    if isinstance(posterior, pd.DataFrame):
        sigma = np.exp(posterior[log_sigma_col].to_numpy(float))
        xi = posterior[xi_col].to_numpy(float)
    else:
        sigma = np.exp(np.asarray(posterior[log_sigma_col], dtype=float))
        xi = np.asarray(posterior[xi_col], dtype=float)

    zT = gpd_return_level(
        T[None, :], u=float(u), sigma=sigma[:, None], xi=xi[:, None], lam_per_year=float(lam_per_year)
    )  # (n_draws, n_T)

    out = {
        "T": T,
        "mean": np.mean(zT, axis=0),
        "std":  np.std(zT, axis=0, ddof=1),
        "median": np.quantile(zT, 0.5, axis=0),
    }
    for q in quantiles:
        out[f"q{int(round(100*q)):02d}"] = np.quantile(zT, q, axis=0)

    summary = pd.DataFrame(out)
    return (summary, zT) if return_draws else summary