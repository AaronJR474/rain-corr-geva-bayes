import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pyproj import Geod
from sklearn.neighbors import BallTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel

# compute azimuth and distance to reference station/point
def LongLatToPolar(Xst, Xeq):
    geod = Geod(ellps='WGS84')
    r = geod.inv(lons1 = Xeq[:, 0], lats1 = Xeq[:, 1],
        lons2 = Xst[:, 0], lats2 = Xst[:, 1])
    repi = r[2]/1000
    az = r[0]*np.pi/180
    az[az<0] += 2*np.pi
    Xp = np.vstack([repi, az]).T
    return Xp

# Function to calculate yearly completeness per station
def calculate_yearly_completeness(df, date_col='date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year

    stations = df.drop(columns=[date_col, 'year']).columns
    results = []

    for year, group in df.groupby('year'):
        total_days = group.shape[0]
        for station in stations:
            available_days = group[station].notna().sum()
            completeness = available_days / total_days * 100

            # Check rainy season coverage (June to December/January for tropical regions)
            rainy_season = group[group[date_col].dt.month.isin([1,6,7,8,9,10,11,12])]
            rainy_available = rainy_season[station].notna().sum()
            rainy_days = rainy_season.shape[0]
            rainy_coverage = rainy_available / rainy_days * 100 if rainy_days >= 0 else np.nan

            # Annual maximum
            annual_max = group[station].max(skipna=True)

            results.append({
                'year': year,
                'station': station,
                'total_days': total_days,
                'available_days': available_days,
                'completeness_%': round(completeness, 1),
                'rainy_coverage_%': round(rainy_coverage, 1),
                'annual_max': round(annual_max, 2) if pd.notna(annual_max) else np.nan
            })

    return pd.DataFrame(results)

# Circular Median for angular distance
def circmedian(angles, period = 2*np.pi):
    """
    Compute the circular median of a set of angles.

    The circular median is the angle (wrapped to [0, period)) that minimizes
    the total circular deviation to all other angles. Distances are computed
    on the circle, so opposite‑side wraparound is handled correctly.

    Parameters
    ----------
    angles : array-like
        Input angles in radians. Values may lie outside [0, period); they
        are wrapped internally.
    period : float, default 2π
        Circular period. Use 2π for radians or 360 for degrees.

    Returns
    -------
    median_angle : float
        The circular median in [0, period). Returns NaN if `angles` is empty.

    Notes
    -----
    The circular median is defined as:

        argmin_a  Σ_i  min(|a - θ_i|, period - |a - θ_i|)

    where θ_i are the wrapped input angles.
    """
    if len(angles) == 0:
        return np.nan

    # Wrap angles into [0, 2π)
    angles = np.mod(angles, period)
    sorted_angles = np.sort(angles)

    # Total circular deviation for each candidate
    def angular_distance(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, period - diff)

    total_deviation = np.array([
        np.sum(angular_distance(angle, sorted_angles)) for angle in sorted_angles
    ])

    return sorted_angles[np.argmin(total_deviation)]

# angular dissimilarity
def getAngDist(theta1,theta2):
    cos_angle = np.cos(np.abs(theta1-theta2))
    distA = np.arccos(np.clip(cos_angle,-1,1))
    return distA

# euclidean dissimilarity
def getDiss(diss_1,diss_2):
    distS = np.abs(diss_1-diss_2)
    return distS

EARTH_R_KM = 6371.0088
# source: https://docs.openquake.org/oq-engine/2.7/_modules/openquake/hmtk/seismicity/utils.html#haversine
def haversine_oq(lon1, lat1, lon2, lat2, radians=False, earth_rad=EARTH_R_KM):
    """
    Allows to calculate geographical distance
    using the haversine formula.

    :param lon1: longitude of the first set of locations
    :type lon1: numpy.ndarray
    :param lat1: latitude of the frist set of locations
    :type lat1: numpy.ndarray
    :param lon2: longitude of the second set of locations
    :type lon2: numpy.float64
    :param lat2: latitude of the second set of locations
    :type lat2: numpy.float64
    :keyword radians: states if locations are given in terms of radians
    :type radians: bool
    :keyword earth_rad: radius of the earth in km
    :type earth_rad: float
    :returns: geographical distance in km
    :rtype: numpy.ndarray
    """
    if not radians:
        cfact = np.pi / 180.
        lon1 = cfact * lon1
        lat1 = cfact * lat1
        lon2 = cfact * lon2
        lat2 = cfact * lat2

    # Number of locations in each set of points
    if not np.shape(lon1):
        nlocs1 = 1
        lon1 = np.array([lon1])
        lat1 = np.array([lat1])
    else:
        nlocs1 = np.max(np.shape(lon1))
    if not np.shape(lon2):
        nlocs2 = 1
        lon2 = np.array([lon2])
        lat2 = np.array([lat2])
    else:
        nlocs2 = np.max(np.shape(lon2))
    # Pre-allocate array
    distance = np.zeros((nlocs1, nlocs2))
    i = 0
    while i < nlocs2:
        # Perform distance calculation
        dlat = lat1 - lat2[i]
        dlon = lon1 - lon2[i]
        aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
                                            (np.sin(dlon / 2.) ** 2.))
        distance[:, i] = (2. * earth_rad * np.arctan2(np.sqrt(aval),
                                                      np.sqrt(1 - aval))).T
        i += 1
    return distance

# non-stationary correlations
def cossim(im_sim_all, nj_lim, hmax=None, ref_sta=None, sta_i=None, sta_j=None):
    """
    Correlation cloud builder for new schema.

    Required columns in im_sim_all:
      ['station_id','date_id','rainfall_norm','Lat','Lon','Elevation','r_azi','r_ref']

    Modes:
      1) Full network (default): all unique station pairs
      2) Reference station only: pairs involving ref_sta
      3) Cross-subsets: all pairs (s in sta_i) x (t in sta_j)

    Returns DataFrame with:
      ['p_hat','z_p_hat','h',
       'sta_j','sta_k','n_j',
       'Elev_j','Elev_k','dElev',
       'rRef_j','rRef_k','dRref',
       'theta_j','theta_k','dA',
       'Lon_j','Lat_j','Lon_k','Lat_k']
    """
    EARTH_R_KM = 6371.0088
    finite_hmax = (hmax is not None) and np.isfinite(hmax)

    # --- mode checks ---
    use_subset_pairs = (sta_i is not None) and (sta_j is not None)
    if (ref_sta is not None) and use_subset_pairs:
        raise ValueError("Specify either ref_sta or sta_i/sta_j, not both.")

    # enforce schema & types
    needed_cols = ['station_id','date_id','rainfall_norm','Lat','Lon','Elevation','r_azi','r_ref']
    missing = [c for c in needed_cols if c not in im_sim_all.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_all = im_sim_all[needed_cols].copy()
    df_all['station_id'] = df_all['station_id'].astype(str).str.strip()
    df_all = df_all.dropna(subset=['station_id','date_id','rainfall_norm','Lat','Lon'])

    # dense station x date arrays
    df_key = df_all[['station_id','date_id','rainfall_norm','r_azi']].copy()
    sta_codes, stations = pd.factorize(df_key['station_id'], sort=False)
    dt_codes, dates      = pd.factorize(df_key['date_id'],    sort=False)
    S, T = len(stations), len(dates)

    y_vals  = df_key['rainfall_norm'].to_numpy(float)
    az_vals = df_key['r_azi'].to_numpy(float)   # assume radians; convert earlier if degrees

    Y  = np.full((S, T), np.nan, float)
    AZ = np.full((S, T), np.nan, float)
    Y[sta_codes, dt_codes]  = y_vals
    AZ[sta_codes, dt_codes] = az_vals

    sta_index = {s: i for i, s in enumerate(stations)}

    # resolve modes (ref / subsets)
    ref_idx = None
    if ref_sta is not None:
        ref_key = str(ref_sta).strip()
        if ref_key not in sta_index:
            raise ValueError(f"ref_sta '{ref_key}' not found among station_id values.")
        ref_idx = sta_index[ref_key]

    if use_subset_pairs:
        sta_i_list = [str(s).strip() for s in np.atleast_1d(sta_i)]
        sta_j_list = [str(s).strip() for s in np.atleast_1d(sta_j)]
        miss_i = [s for s in sta_i_list if s not in sta_index]
        miss_j = [s for s in sta_j_list if s not in sta_index]
        if miss_i or miss_j:
            msgs = []
            if miss_i: msgs.append(f"sta_i not found: {miss_i}")
            if miss_j: msgs.append(f"sta_j not found: {miss_j}")
            raise ValueError("; ".join(msgs))
        idx_i_list = [sta_index[s] for s in sta_i_list]
        idx_j_list = [sta_index[s] for s in sta_j_list]
    else:
        idx_i_list = idx_j_list = None

    # station metadata aligned
    meta = (
        df_all.groupby('station_id', as_index=False)
              .agg({'Lat':'first','Lon':'first','Elevation':'median','r_ref':'median'})
              .set_index('station_id')
              .reindex(stations)
    )
    if meta[['Lat','Lon']].isna().any().any():
        bad = meta[meta['Lat'].isna() | meta['Lon'].isna()].index.tolist()
        raise ValueError(f"Missing Lat/Lon for stations: {bad}")

    lat_arr = meta['Lat'].to_numpy(float)
    lon_arr = meta['Lon'].to_numpy(float)
    elv_arr = meta['Elevation'].to_numpy(float)
    rref_arr = meta['r_ref'].to_numpy(float)

    # optional neighbor pruning
    if finite_hmax and not use_subset_pairs:
        coords_rad = np.deg2rad(np.c_[lat_arr, lon_arr])
        tree = BallTree(coords_rad, metric='haversine')
        rad = hmax / EARTH_R_KM
        nbrs = tree.query_radius(coords_rad, r=rad, return_distance=False)
    else:
        nbrs = None

    # pair iterator & size
    if use_subset_pairs:
        pair_iter = ((i, j) for i in idx_i_list for j in idx_j_list)
        total_pairs = len(idx_i_list) * len(idx_j_list)
    else:
        if nbrs is None:
            if ref_idx is None:
                pair_iter = ((i, j) for i in range(S) for j in range(i+1, S))
                total_pairs = S * (S - 1) // 2
            else:
                pair_iter = ((ref_idx, j) for j in range(S) if j != ref_idx)
                total_pairs = S - 1
        else:
            if ref_idx is None:
                pair_iter = ((i, j) for i in range(S) for j in nbrs[i] if j > i)
                total_pairs = sum((np.asarray(neigh) > i).sum() for i, neigh in enumerate(nbrs))
            else:
                neigh = np.asarray(nbrs[ref_idx])
                pair_iter = ((ref_idx, j) for j in neigh if j != ref_idx)
                total_pairs = int((neigh != ref_idx).sum())

    pair_iter = tqdm(pair_iter, total=total_pairs, desc="Station pairs")

    # storage lists
    p_hat, h_km = [], []
    sta_j_id, sta_k_id, n_j = [], [], []
    elev_j, elev_k, dElev = [], [], []
    rref_j, rref_k, dRref = [], [], []
    theta_j, theta_k, dA = [], [], []
    lon_j, lat_j, lon_k, lat_k = [], [], [], []

    # main loop
    for i_idx, j_idx in pair_iter:
        yi, yj = Y[i_idx], Y[j_idx]
        valid = np.isfinite(yi) & np.isfinite(yj)
        n_overlap = int(valid.sum())
        if n_overlap < nj_lim:
            continue

        # pearsonr uncentered/cosine similarity
        xi, xj = yi[valid], yj[valid]
        den = float(np.sqrt(np.dot(xi, xi) * np.dot(xj, xj)))
        if den == 0.0:
            continue
        p = float(np.dot(xi, xj)) / den

        # angular medians over overlapping dates
        th_i = circmedian(AZ[i_idx, valid])
        th_j = circmedian(AZ[j_idx, valid])
        dA_ij = getAngDist(th_i, th_j) * 180.0 / np.pi  # degrees

        # great-circle distance (km)
        d = haversine_oq(np.array([lon_arr[j_idx]]), np.array([lat_arr[j_idx]]),
                         np.array([lon_arr[i_idx]]), np.array([lat_arr[i_idx]]))[0][0]
        if d <= 0.0:
            continue
        if finite_hmax and d > hmax:
            continue

        # stash
        p_hat.append(p); h_km.append(d); n_j.append(n_overlap)
        sta_k_id.append(stations[i_idx]); sta_j_id.append(stations[j_idx])

        e_i, e_j = elv_arr[i_idx], elv_arr[j_idx]
        elev_k.append(e_i); elev_j.append(e_j); dElev.append(abs(e_i - e_j))

        r_i, r_j = rref_arr[i_idx], rref_arr[j_idx]
        rref_k.append(r_i); rref_j.append(r_j); dRref.append(abs(r_i - r_j))

        theta_k.append(th_i); theta_j.append(th_j); dA.append(dA_ij)

        lon_k.append(lon_arr[i_idx]); lat_k.append(lat_arr[i_idx])
        lon_j.append(lon_arr[j_idx]); lat_j.append(lat_arr[j_idx])

    # assemble & finalize
    out = pd.DataFrame({
        'p_hat': p_hat,
        'h': h_km,
        'sta_j': sta_j_id,
        'sta_k': sta_k_id,
        'n_j': n_j,
        'Elev_j': elev_j,
        'Elev_k': elev_k,
        'dElev': dElev,
        'rRef_j': rref_j,
        'rRef_k': rref_k,
        'dRref': dRref,
        'theta_j': theta_j,
        'theta_k': theta_k,
        'dA': dA,
        'Lon_j': lon_j,
        'Lat_j': lat_j,
        'Lon_k': lon_k,
        'Lat_k': lat_k,
    })

    # Fisher z
    p = np.clip(out['p_hat'].to_numpy(), -1 + 1e-12, 1 - 1e-12)
    out['z_p_hat'] = 0.5 * np.log((1.0 + p) / (1.0 - p))

    # filters & order
    out = out[out['h'] > 0.0]
    if finite_hmax:
        out = out[out['h'] <= hmax]
    out = out[out['n_j'] >= nj_lim]
    out = out.sort_values('h').reset_index(drop=True)
    return out

# Ordinary kriging
# pairwise Euclidean distance between sites
def pairwiseEucDistanceFromPolar(r, theta):
    r_i = r[:, None]          # (N, 1)
    r_j = r[None, :]          # (1, N)
    dtheta = theta[:, None] - theta[None, :]   # (N, N)

    sq_dist = r_i**2 + r_j**2 - 2 * r_i * r_j * np.cos(np.abs(dtheta))
    sq_dist = np.clip(sq_dist, 0.0, np.inf)

    dist = np.sqrt(sq_dist)
    np.fill_diagonal(dist, 0.0)
    return dist

# pairwise Elevation difference/dissimilarity between sites
def pairwiseElevationDissimilarity(elev):
    diff = elev[:, None] - elev[None, :]
    return np.abs(diff)

# pairwise Angular difference/similarity between sites in DEGREES
def pairwiseAngDistanceFromPolarDegrees(theta):

    # Broadcast differences: shape (N, N)
    dtheta = theta[:, None] - theta[None, :]

    # Compute angular distance
    cos_angle = np.cos(np.abs(dtheta))
    dist = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(dist)

# correlation models
def pE(dh,lE,gE):
    p = np.exp(-1*((dh/lE)**gE))
    return p

def pEElev(dh, dElev, lE, gE, lElev):
    p = np.exp(-1*(dh/lE)**gE)*(np.exp(-dElev/lElev))
    return p

def pEAElev(dh, dElev, dA, lE, gE, lElev, lA, w):
    p = np.exp(-1*(dh/lE)**gE) * (w*(np.exp(-dElev/lElev)) + (1-w)*((1 + np.multiply(dA, 1.0/lA)) *
          np.power(1 - np.multiply(dA, 1.0/180), 180/lA)))
    return p

# Ordinary Kriging core
def OK_predict_one_target(C_oo_latent, c_uo_latent, z_o,
                          partial_sill, tau2, predict_observed=True):
    """
    Ordinary kriging predictor for a single target location.

    This function solves the ordinary kriging (OK) linear system

        [ C_oo + tau^2 I    1 ] [ w ]   [ c_uo ]
        [     1^T           0 ] [ λ ] = [  1   ]

    where:
        - C_oo_latent is the latent (process) covariance among observed sites.
        - c_uo_latent is the latent covariance between the target u and the
          observed sites.
        - tau2 is the nugget (measurement error) variance, added only to the
          observed–observed block.
        - z_o are the observed data values at the observed sites.

    The kriging estimate is

        ẑ_u = w^T z_o.

    The kriging variance of the *latent process* at u is

        Var_latent(ẑ_u) = partial_sill - c_uo^T w - λ.

    If the user requests the variance of a *noisy observation* at u, the
    measurement-error variance tau2 is added:

        Var_obs(ẑ_u) = Var_latent(ẑ_u) + tau2.

    This function returns either the latent variance or the observation
    variance depending on the `predict_observed` flag.

    Parameters
    ----------
    C_oo_latent : (n, n) ndarray
        Latent covariance matrix among observed sites (no nugget included).
    c_uo_latent : (n,) ndarray
        Latent covariance between the target location and observed sites.
    z_o : (n,) ndarray
        Observed data values at the observed sites.
    partial_sill : float
        Latent process variance at zero lag.
    tau2 : float
        Nugget (measurement error) variance.
    predict_observed : bool, default=True
        If True, return the variance of an observation at u (latent variance
        plus tau2). If False, return the variance of the latent process.

    Returns
    -------
    est : float
        Ordinary kriging estimate at the target location.
    var : float
        Kriging variance (latent or observation-level, depending on flag),
        clipped to be non-negative.
    """

    # Add nugget only to the observed-observed block
    C_oo = C_oo_latent + tau2 * np.eye(C_oo_latent.shape[0])

    n = C_oo.shape[0]
    A = np.empty((n+1, n+1), float)
    A[:n, :n] = C_oo
    A[:n, n]  = 1.0
    A[n, :n]  = 1.0
    A[n, n]   = 0.0

    b = np.empty(n+1, float)
    b[:n] = c_uo_latent
    b[n]  = 1.0

    wlam = np.linalg.solve(A, b)
    w, lam = wlam[:n], wlam[n]

    est = float(np.dot(w, z_o))

    # OK variance: C0 - c^T w - λ
    # Use process (latent) variance at zero lag; add tau2 only when predicting an observed site
    # C0 = partial_sill + (tau2 if predict_observed else 0.0)
    # var = float(C0 - np.dot(c_uo_latent, w) - lam)
    C0_latent = partial_sill
    var_latent = float(C0_latent - np.dot(c_uo_latent, w) - lam)
    var = var_latent + (tau2 if predict_observed else 0.0)
    return est, max(var, 0.0)

def build_covariance(day_df, model, params,
                     jitter_frac=1e-10, nugget_frac=None, sill_override=None):
    """
    Construct the latent covariance matrix for all stations on a given day,
    together with the nugget variance and partial sill used in kriging.

    This function computes:
      1. Pairwise dissimilarities among all stations:
         - dh     : Euclidean distance in polar coordinates
         - dElev  : elevation dissimilarity
         - dAdeg  : angular separation in degrees
      2. The model-specific correlation matrix P:
         - model = "E":
               P = pE(dh; lE, gE)
         - model = "EElev":
               P = pEElev(dh, dElev; lE, gE, lElev)
         - model = "EAElev":
               P = pEAElev(dh, dElev, dAdeg; lE, gE, lElev, lA_deg, w)
               (clipped to [0, 1] for numerical stability)
      3. The total daily variance (sill_total), either:
         - provided via `sill_override`, or
         - estimated from the sample variance of observed rainfall.
      4. The nugget fraction f (from params["nugget"], or `nugget_frac`, or 0),
         and the decomposition:
               tau2         = f * sill_total
               partial_sill = (1 - f) * sill_total
      5. The latent covariance:
               C_latent = partial_sill * P
         with a small diagonal jitter added for numerical stability.

    Parameters
    ----------
    day_df : pandas.DataFrame
        Must contain columns:
            "r_ref"      : radial distance (km)
            "r_azi"      : azimuth (radians)
            "Elevation"  : elevation (m)
            "rainfall"   : observed rainfall values
    model : {"E", "EElev", "EAElev"}
        Choice of spatial correlation model.
    params : dict
        Model parameters:
            E:       {"lE", "gE"}
            EElev:   {"lE", "gE", "lElev"}
            EAElev:  {"lE", "gE", "lElev", "lA_deg", "w"}
        Optionally may include "nugget".
    jitter_frac : float, default 1e-10
        Fraction of sill_total added to the diagonal of C_latent for stability.
    nugget_frac : float or None
        If provided, overrides params["nugget"].
    sill_override : float or None
        If provided, overrides the sample variance of observed rainfall.

    Returns
    -------
    C_latent : (S, S) ndarray
        Latent (process) covariance matrix for all stations.
    tau2 : float
        Nugget (measurement error) variance.
    partial_sill : float
        Process variance (sill minus nugget), used in kriging.
    """

    # pairwise dissimilarities
    r     = day_df["r_ref"].to_numpy(float)
    theta = day_df["r_azi"].to_numpy(float)     # radians
    elev  = day_df["Elevation"].to_numpy(float)

    dh    = pairwiseEucDistanceFromPolar(r, theta)       # (S,S)
    dElev = pairwiseElevationDissimilarity(elev)         # (S,S)
    dAdeg = pairwiseAngDistanceFromPolarDegrees(theta)   # (S,S)

    # correlation P
    if model == "E":
        P = pE(dh, params["lE"], params["gE"])
    elif model == "EElev":
        P = pEElev(dh, dElev, params["lE"], params["gE"], params["lElev"])
    elif model == "EAElev":
        # Ensure angular kernel stays within [0,1]
        P = pEAElev(dh, dElev, dAdeg, params["lE"], params["gE"],
                    params["lElev"], params["lA_deg"], params["w"])
        P = np.clip(P, 0.0, 1.0)
    else:
        raise ValueError("model must be 'E', 'EElev', or 'EAElev'")

    # total day variance from observed values (or override)
    z_obs = day_df["rainfall"].to_numpy(float)
    z_obs = z_obs[np.isfinite(z_obs)]
    SMALL_VAR = 0.1

    if sill_override is not None:
        sill_total = float(sill_override)
    else:
        if z_obs.size >= 2:
            v = float(np.var(z_obs, ddof=1))
            sill_total = v if v > 0.0 else SMALL_VAR
        else:
            sill_total = SMALL_VAR

    # nugget fraction f
    if "nugget" in (params or {}):
        f = float(params["nugget"])
    elif nugget_frac is not None:
        f = float(nugget_frac)
    else:
        f = 0.0
    f = max(0.0, min(f, 0.99))

    tau2         = f * sill_total
    partial_sill = (1.0 - f) * sill_total

    # latent covariance
    C_latent = partial_sill * P

    # tiny jitter for numerics (not a nugget)
    np.fill_diagonal(C_latent, np.diag(C_latent) + jitter_frac * sill_total)

    return C_latent, tau2, partial_sill

def OK_per_day(day_df, model="E", params=None, min_obs=3,
               nugget_frac=None, jitter_frac=1e-10,
               predict_observed=False):
    """
    Perform ordinary kriging (OK) for a single day's rainfall field.

    This function:
      1. Splits stations into observed and missing based on finite rainfall values.
      2. If there are too few observed stations (< min_obs) or no missing stations,
         returns the input with observed values unchanged and missing values left as NaN.
      3. Builds the latent covariance matrix C_latent and nugget tau2 using
         `build_covariance`, based on the chosen spatial model ("E", "EElev", "EAElev").
      4. Extracts the observed–observed block C_oo_latent and observed data z_o.
      5. For each missing station u, computes the OK estimate and variance via
         `OK_predict_one_target`, using:
             - latent covariance C_latent
             - nugget tau2 applied only to C_oo
             - partial_sill for the latent process variance
             - optional addition of tau2 to return observation-level variance
               if `predict_observed=True`.
      6. Returns a DataFrame with:
             "rainfall_est" : kriging estimates (observed values unchanged)
             "rainfall_var" : kriging variances, interpreted as:
                 - latent-field variance for missing stations and 0 for observed
                   if `predict_observed=False`;
                 - observation-level variance (latent + tau2) for missing stations
                   and tau2 for observed if `predict_observed=True`.

    Parameters
    ----------
    day_df : pandas.DataFrame
        Must contain columns:
            "rainfall"   : observed rainfall (NaN for missing)
            "r_ref"      : radial distance (km)
            "r_azi"      : azimuth (radians)
            "Elevation"  : elevation (m)
    model : {"E", "EElev", "EAElev"}, default "E"
        Spatial correlation model used to construct the covariance.
    params : dict or None
        Model parameters passed to `build_covariance`.
        May optionally include "nugget".
    min_obs : int, default 3
        Minimum number of observed stations required to perform kriging.
    nugget_frac : float or None
        If provided, overrides params["nugget"] when computing tau2.
    jitter_frac : float, default 1e-10
        Fraction of the sill added to the diagonal of C_latent for numerical stability.
    predict_observed : bool, default False
        If False, `rainfall_var` represents latent-field variance for missing
        stations and 0 for observed stations.
        If True, `rainfall_var` represents observation-level variance: latent
        variance + tau2 for missing stations, and tau2 for observed stations.

    Returns
    -------
    df_out : pandas.DataFrame
        Same as input but with two additional columns:
            "rainfall_est" : kriging estimate at each station
            "rainfall_var" : kriging variance at each station
    """

    df = day_df.reset_index(drop=True).copy()

    obs_mask = np.isfinite(df["rainfall"].to_numpy(float))
    idx_all = np.arange(len(df))
    idx_obs = idx_all[obs_mask]
    idx_mis = idx_all[~obs_mask]

    if len(idx_obs) < min_obs or len(idx_mis) == 0:
        df["rainfall_est"] = df["rainfall"].to_numpy(float)
        df["rainfall_var"] = np.where(obs_mask, 0.0, np.nan)
        return df

    C_latent, tau2, partial_sill = build_covariance(
        df, model, params, jitter_frac=jitter_frac, nugget_frac=nugget_frac
    )

    z_o  = df.iloc[idx_obs]["rainfall"].to_numpy(float)
    C_oo_latent = C_latent[np.ix_(idx_obs, idx_obs)]

    est = df["rainfall"].to_numpy(float).copy()
    var = np.zeros(len(df), float)
    var[idx_obs] = 0.0 if not predict_observed else tau2  # optional choice

    for u in idx_mis:
        c_uo_latent = C_latent[np.ix_([u], idx_obs)].ravel()
        xhat, vhat = OK_predict_one_target(
            C_oo_latent, c_uo_latent, z_o,
            partial_sill=partial_sill, tau2=tau2,
            predict_observed=predict_observed
        )
        est[u] = xhat
        var[u] = vhat

    df["rainfall_est"] = est
    df["rainfall_var"] = var
    return df

def run_OK_all_days(rainfall_long, model="E", params=None, min_obs=3,
                    nugget_frac=None, jitter_frac=1e-10,
                    predict_observed=False):
    """
    Apply ordinary kriging (OK) day‑by‑day across a long-format rainfall dataset.

    This function loops over all unique `date_id` values in `rainfall_long`,
    extracts the subset of stations for each day, and applies `OK_per_day`
    using the specified spatial model and parameters. The per‑day results are
    concatenated into a single DataFrame with kriging estimates and variances.

    Expected columns in `rainfall_long`:
        "date_id"     : day identifier
        "station_id"  : station identifier
        "rainfall"    : observed rainfall (NaN for missing)
        "Lat", "Lon"  : geographic coordinates
        "Elevation"   : elevation (m)
        "r_ref"       : radial distance (km)
        "r_azi"       : azimuth (radians)

    Parameters
    ----------
    rainfall_long : pandas.DataFrame
        Long-format dataset containing all stations across all days.
    model : {"E", "EElev", "EAElev"}, default "E"
        Spatial correlation model passed to `OK_per_day`.
    params : dict or None
        Model parameters passed to `OK_per_day` and `build_covariance`.
    min_obs : int, default 3
        Minimum number of observed stations required to perform kriging on a day.
    nugget_frac : float or None
        Nugget fraction passed to `OK_per_day` (overrides params["nugget"] if present).
    jitter_frac : float, default 1e-10
        Fraction of the sill added to the diagonal of the latent covariance for stability.
    predict_observed : bool, default False
        If True, kriging variance for observed stations is set to tau2 instead of 0.

    Returns
    -------
    df_out : pandas.DataFrame
        Concatenation of per‑day kriging results, including:
            "rainfall_est" : kriging estimate at each station
            "rainfall_var" : kriging variance at each station
    """
    out = []
    for d in tqdm(rainfall_long["date_id"].drop_duplicates().tolist(),
                  desc="Ordinary kriging by day"):
        grp = rainfall_long[rainfall_long["date_id"] == d]
        out.append(
            OK_per_day(grp, model=model, params=params, min_obs=min_obs,
                       nugget_frac=nugget_frac, jitter_frac=jitter_frac,
                       predict_observed=predict_observed)
        )
    return pd.concat(out, ignore_index=True)

# Mising At Random Creation and evaluation
def make_fake_missing(
    df_full: pd.DataFrame,
    min_obs: int = 3,
    mask_n: int | None = None,
    mask_frac: float | None = 0.3,
    random_state: int = 123
):
    """
    Create artificial missing rainfall values for imputation experiments.

    For each day (grouped by ``date_id``), a subset of stations is randomly
    selected and their rainfall values are replaced with NaN, while ensuring
    that at least ``min_obs`` observed stations remain. The true rainfall
    values of the masked rows are recorded in a separate log for later
    evaluation.

    Exactly one of ``mask_n`` or ``mask_frac`` must be provided:
        - ``mask_n``   : fixed number of stations to mask per day
        - ``mask_frac``: fraction of stations to mask per day

    Required columns in ``df_full``:
        "date_id", "station_id", "rainfall",
        "Lat", "Lon", "Elevation", "r_ref", "r_azi"

    Parameters
    ----------
    df_full : pandas.DataFrame
        Full dataset containing all stations across all days.
    min_obs : int, default 3
        Minimum number of observed stations that must remain after masking.
    mask_n : int or None
        Fixed number of stations to mask per day. Mutually exclusive with ``mask_frac``.
    mask_frac : float or None
        Fraction of stations to mask per day. Mutually exclusive with ``mask_n``.
    random_state : int, default 123
        Seed for the random number generator.

    Returns
    -------
    fake_df : pandas.DataFrame
        Copy of the input with NaNs inserted in the ``rainfall`` column for
        the masked rows. Includes a boolean column ``was_masked``.
    mask_log : pandas.DataFrame
        Table containing the masked rows and their true rainfall values,
        with columns:
            "date_id", "station_id", "rainfall_true"
    """

    if (mask_n is None) == (mask_frac is None):
        raise ValueError("Specify exactly one of mask_n or mask_frac.")

    rng = np.random.default_rng(random_state)
    out_parts = []
    mask_log_parts = []

    for date_id, g in df_full.groupby("date_id", sort=True):
        g = g.copy()

        n = len(g)
        if n <= min_obs:
            # nothing to mask
            g["was_masked"] = False
            out_parts.append(g)
            continue

        # choose how many to mask
        k = mask_n if mask_n is not None else int(np.floor(mask_frac * n))
        k = max(0, min(k, n - min_obs))  # clamp so we keep >= min_obs

        if k == 0:
            g["was_masked"] = False
            out_parts.append(g)
            continue

        idx_all = np.arange(n)
        idx_mask = rng.choice(idx_all, size=k, replace=False)

        # record which rows are masked and their ground truth
        masked_rows = g.iloc[idx_mask][["date_id","station_id","rainfall"]].copy()
        masked_rows.rename(columns={"rainfall": "rainfall_true"}, inplace=True)
        mask_log_parts.append(masked_rows)

        # apply masking
        g["was_masked"] = False
        g.iloc[idx_mask, g.columns.get_loc("rainfall")] = np.nan
        g.iloc[idx_mask, g.columns.get_loc("was_masked")] = True

        out_parts.append(g)

    fake_df = pd.concat(out_parts, ignore_index=True)
    mask_log = pd.concat(mask_log_parts, ignore_index=True) if mask_log_parts else pd.DataFrame(
        columns=["date_id","station_id","rainfall_true"]
    )
    return fake_df, mask_log

def evaluate_imputation(
    pred_df: pd.DataFrame,
    mask_log: pd.DataFrame,
    label_prefix: str = ""
):
    """
    Evaluate imputation accuracy on artificially masked rainfall values.

    This function merges the prediction DataFrame with the mask log
    (which stores the true rainfall values for the masked entries),
    computes pointwise errors, and returns standard accuracy metrics
    along with the merged evaluation table.

    Expected columns:
        pred_df :
            "date_id"        : day identifier
            "station_id"     : station identifier
            "rainfall_est"   : model-imputed rainfall
        mask_log :
            "date_id"        : day identifier
            "station_id"     : station identifier
            "rainfall_true"  : true rainfall before masking

    Metrics computed:
        - MAE   : mean absolute error
        - RMSE  : root mean squared error
        - Bias  : mean signed error
        - Corr  : Pearson correlation between true and estimated rainfall
        - N_masked : number of masked entries evaluated

    Parameters
    ----------
    pred_df : pandas.DataFrame
        DataFrame containing model predictions for all stations.
    mask_log : pandas.DataFrame
        DataFrame containing the true rainfall values for the masked entries.
    label_prefix : str, default ""
        Optional prefix added to metric names (useful when comparing models).

    Returns
    -------
    metrics : dict
        Dictionary of scalar evaluation metrics.
    eval_df : pandas.DataFrame
        Merged table containing:
            "date_id", "station_id",
            "rainfall_true", "rainfall_est",
            "error"
    """

    cols_needed = ["date_id","station_id","rainfall_est"]
    if not set(cols_needed).issubset(pred_df.columns):
        raise ValueError(f"pred_df must contain {cols_needed}")

    eval_df = mask_log.merge(
        pred_df[["date_id","station_id","rainfall_est"]],
        on=["date_id","station_id"], how="left"
    ).copy()

    eval_df["error"] = eval_df["rainfall_est"] - eval_df["rainfall_true"]
    mae  = float(np.mean(np.abs(eval_df["error"])))
    rmse = float(np.sqrt(np.mean(np.square(eval_df["error"]))))
    bias = float(np.mean(eval_df["error"]))
    corr = float(eval_df[["rainfall_true","rainfall_est"]].corr().iloc[0,1])

    metrics = {
        f"{label_prefix}MAE": mae,
        f"{label_prefix}RMSE": rmse,
        f"{label_prefix}Bias": bias,
        f"{label_prefix}N_masked": int(len(eval_df)),
        f"{label_prefix}Corr": corr

    }
    return metrics, eval_df

# Gaussian Process Regression
def make_gpr_model(kind="space", nu=1.5, n_restarts=5, random_state=0):
    """
    Construct a scikit-learn GaussianProcessRegressor for daily rainfall.

    Parameters
    ----------
    kind : {"space", "space+elev"}
        Kernel structure to use:

        - "space":
            Constant * Matern([x_km, y_km]) + White

        - "space+elev":
            (Constant * Matern([x_km, y_km])) +
            (Constant * RBF([Elevation])) +
            White

        Notes on implementation:
        - In scikit-learn, kernels act on the full feature vector passed to the
          regressor. If you use "space+elev", you should pass features in the
          order [x_km, y_km, Elevation] and (ideally) scale Elevation so the
          optimizer is well-conditioned (e.g., standardize Elevation globally).
        - The ConstantKernel sets the overall signal variance (amplitude) of the
          corresponding kernel term. Without it, the kernel variance is fixed.
        - The WhiteKernel represents iid observation noise ("nugget") and is
          added to the predictive *observation* variance when you want a
          predictive uncertainty for rainfall itself (not just the latent field).

    nu : float
        Smoothness of the Matern kernel (common choices: 0.5, 1.5, 2.5).

    n_restarts : int
        Number of optimizer restarts used by sklearn to fit kernel hyperparameters.

    random_state : int
        Random seed for optimizer restarts.

    Returns
    -------
    gpr : sklearn.gaussian_process.GaussianProcessRegressor
        Unfitted GPR model.
    """

    if kind == "space":
        k = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=10.0, length_scale_bounds=(0.5, 500.0), nu=nu)
            + WhiteKernel(noise_level=0.04, noise_level_bounds=(4e-2, 1e1))
        )

    elif kind == "space+elev":

        # Matern dominated by (x,y): make elev lengthscale huge
        k_space = (ConstantKernel(1.0, (1e-3, 1e3)) *
                   Matern(length_scale=10.0, length_scale_bounds=(0.5, 500.0), nu=1.5))

        # RBF dominated by elev: make (x,y) lengthscales huge
        k_elev = ConstantKernel(0.5, (1e-3, 1e3)) * RBF(
            length_scale=100.0,
            length_scale_bounds=(10.0, 10000.0),
        )

        k = k_space + k_elev + WhiteKernel(noise_level=0.01, noise_level_bounds=(5e-2, 1e1))

    else:
        raise ValueError("kind must be 'space' or 'space+elev'")

    gpr = GaussianProcessRegressor(
        kernel=k,
        alpha=0.0,               # keep 0; WhiteKernel represents nugget
        normalize_y=False,       # we standardize globally below
        n_restarts_optimizer=n_restarts,
        random_state=random_state
    )
    return gpr


def daily_train_test_gpr(
    rainfall_2005,
    test_frac=0.2,
    min_obs=3,
    kernel_kind="space",
    seed=42,
    y_global_standardize=True,
    y_mu=None,
    y_sigma=None,
    nu=1.5,
    n_restarts=2,
):
    """
    Per-day GPR cross-validation: randomly hold out stations within each day,
    fit a GPR on the remaining stations for that day, and evaluate predictions
    at the held-out stations.

    This evaluates "daily imputation" skill under missing-at-random station
    removal (within each date_id).

    Parameters
    ----------
    rainfall_2005 : pandas.DataFrame
        Input data. Must include the following columns:
          - "date_id" (day identifier; any hashable type)
          - "station_id"
          - "rainfall" (target; in mm or your chosen units)
          - "x_km", "y_km" (projected coordinates in km)

        If kernel_kind == "space+elev", also requires:
          - "Elevation" (or whatever feature name you use in your implementation)

        Assumes rainfall has no NaNs for evaluation (the hold-out is created inside).

    test_frac : float
        Fraction of stations to hold out per day (approximate, rounded down).
        Always leaves at least `min_obs` stations for training per day.

    min_obs : int
        Minimum number of observed stations required to fit a day model.
        Days with fewer than (min_obs + 1) stations are skipped.

    kernel_kind : {"space", "space+elev"}
        Which kernel family to use (passed to `make_gpr_model`).

    seed : int
        RNG seed used to select held-out stations per day.

    y_global_standardize : bool
        If True, train the GPR on globally standardized rainfall:
            y_std = (y - y_mu) / y_sigma
        and back-transform predictions/variances to rainfall units afterward.

        This reduces per-day scale pathologies and makes the WhiteKernel noise
        parameter more comparable across days.

    y_mu, y_sigma : float or None
        Global mean and standard deviation used when y_global_standardize=True.
        If None, they are computed from `rainfall_2005["rainfall"]`.

        Back-transform:
            y_pred = y_mu + y_sigma * y_pred_std
            Var(y) = (y_sigma^2) * Var(y_std)

    nu : float
        Matern smoothness parameter (passed through to `make_gpr_model`).

    n_restarts : int
        Number of optimizer restarts when fitting GPR hyperparameters.

    Returns
    -------
    pred_df : pandas.DataFrame
        Predictions for held-out stations only, with (at least) columns:
          - date_id, station_id
          - rainfall_true
          - rainfall_est
          - rainfall_var

        NOTE on rainfall_var:
        This should be the predictive *observation* variance in rainfall units
        (latent predictive variance + WhiteKernel noise variance, then back-transformed
        if standardization is used).

    metrics : dict
        Summary metrics computed over all held-out points:
          - N_test
          - MAE, RMSE, Bias
          - z_mean, z_std where z = (pred-true) / sqrt(rainfall_var)
          - y_mu, y_sigma if y_global_standardize=True
    """

    df_all = rainfall_2005.copy()

    # global standardization parameters
    if y_global_standardize:
        if y_mu is None:
            y_mu = float(np.nanmean(df_all["rainfall"].to_numpy(float)))
        if y_sigma is None:
            y_sigma = float(np.nanstd(df_all["rainfall"].to_numpy(float), ddof=1))
        if not np.isfinite(y_sigma) or y_sigma <= 0:
            y_sigma = 1.0  # guard
    else:
        y_mu, y_sigma = 0.0, 1.0

    rng = np.random.default_rng(seed)

    rows = []
    for day, g in tqdm(df_all.groupby("date_id", sort=True), desc=f"GPR per day ({kernel_kind})"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < (min_obs + 1):
            continue

        # choose test indices
        n_test = max(1, int(np.floor(test_frac * n)))
        n_train = n - n_test
        if n_train < min_obs:
            n_test = n - min_obs
            n_train = min_obs
        if n_test <= 0:
            continue

        test_idx = rng.choice(n, size=n_test, replace=False)
        train_idx = np.setdiff1d(np.arange(n), test_idx)

        # targets: globally standardized
        y = g["rainfall"].to_numpy(float)
        y_std = (y - y_mu) / y_sigma  # if y_global_standardize=False then y_mu=0,y_sigma=1

        y_train = y_std[train_idx]

        # features
        if kernel_kind == "space":
            X = g[["x_km", "y_km"]].to_numpy(float)
        else:
            elev = g["Elevation"].to_numpy(float)
            X = np.column_stack([
                g["x_km"].to_numpy(float),
                g["y_km"].to_numpy(float),
                elev
            ]).astype(float)

        X_train = X[train_idx]
        X_test  = X[test_idx]

        # fit
        gpr = make_gpr_model(kind=kernel_kind, nu=nu, n_restarts=n_restarts, random_state=0)
        gpr.fit(X_train, y_train)

        # predict latent std, then convert to observed std by adding WhiteKernel noise variance
        y_pred_std, y_std_f = gpr.predict(X_test, return_std=True)
        var_f = y_std_f**2

        # add fitted noise variance (WhiteKernel) for observation uncertainty
        noise_var = float(gpr.kernel_.k2.noise_level)

        var_y_std = var_f + noise_var  # in standardized-y units

        # back-transform to original rainfall units
        y_pred = y_mu + y_sigma * y_pred_std
        y_var  = (y_sigma**2) * var_y_std

        out = pd.DataFrame({
            "date_id": g.loc[test_idx, "date_id"].to_numpy(),
            "station_id": g.loc[test_idx, "station_id"].to_numpy(),
            "rainfall_true": y[test_idx],
            "rainfall_est": y_pred,
            "rainfall_var": y_var,
        })
        rows.append(out)

    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if len(pred_df) == 0:
        metrics = {"N_test": 0}
        return pred_df, metrics

    err = pred_df["rainfall_est"].to_numpy(float) - pred_df["rainfall_true"].to_numpy(float)
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    z = err / np.sqrt(np.clip(pred_df["rainfall_var"].to_numpy(float), 1e-12, None))
    metrics = {
        "N_test": int(len(pred_df)),
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "y_mu": float(y_mu),
        "y_sigma": float(y_sigma),
    }
    return pred_df, metrics

def gpr_impute_by_day(
    df,
    transformer,
    kernel_kind="space",
    min_obs=3,
    nu=1.5,
    n_restarts=5,
    seed=0,
    y_global_standardize=True,
    y_mu=None,
    y_sigma=None,
    clip_nonneg=True,
):
    """
    Forward imputation using a separate per-day Gaussian Process model.

    For each date_id:
      1) fit a GPR using the observed stations for that day, and
      2) predict rainfall (and predictive variance) at stations with missing rainfall.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing both observed and missing rainfall values.
        Required columns:
          - "date_id"
          - "station_id"
          - "rainfall"   (NaN for missing entries to be imputed)
          - "Lat", "Lon" (used to build projected coordinates)

        If kernel_kind == "space+elev", also requires:
          - "Elevation"

    transformer : pyproj.Transformer (or compatible)
        Coordinate transformer used to project Lon/Lat -> x/y (meters).
        The function typically stores coordinates as:
          x_km = x_m / 1000, y_km = y_m / 1000

    kernel_kind : {"space", "space+elev"}
        Which kernel family to use (see `make_gpr_model` docs).

    min_obs : int
        Minimum number of observed stations required to fit a day model.
        If a day has fewer than min_obs observations, missing entries are left as NaN.

    nu : float
        Matern smoothness parameter.

    n_restarts : int
        Number of optimizer restarts when fitting GPR hyperparameters.

    seed : int
        Seed used for any stochastic aspects (e.g., optimizer restarts randomness).

    y_global_standardize : bool
        If True, standardize rainfall globally using y_mu/y_sigma during training and
        back-transform predictions to rainfall units. Recommended for stability.

    y_mu, y_sigma : float or None
        Global mean/std used for standardization if enabled. If None, they are computed
        from the observed (non-NaN) rainfall values in `df`.

    clip_nonneg : bool
        If True, clip imputed rainfall_est to be >= 0.0 (rainfall is nonnegative).
        This does not change the reported variance.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of df with added columns:
          - "rainfall_est": observed rainfall where available; imputed values where missing
          - "rainfall_var": predictive observation variance at each row
                * 0 for observed rows (or you can store tau^2 if you want)
                * >0 for imputed rows when a model was fit
                * NaN for rows that could not be imputed (e.g., < min_obs)

        Variance is intended to be predictive *observation* variance (latent + nugget),
        in rainfall units.
    """

    df_out = df.copy()

    # project coords (km)
    x, y = transformer.transform(df_out["Lon"].to_numpy(float), df_out["Lat"].to_numpy(float))
    df_out["x_km"] = x / 1000.0
    df_out["y_km"] = y / 1000.0

    # global standardization constants (recommended)
    if y_global_standardize:
        y_all = df_out["rainfall"].to_numpy(float)
        y_all = y_all[np.isfinite(y_all)]
        if y_mu is None:
            y_mu = float(np.mean(y_all)) if y_all.size else 0.0
        if y_sigma is None:
            y_sigma = float(np.std(y_all, ddof=1)) if y_all.size >= 2 else 1.0
        if (not np.isfinite(y_sigma)) or (y_sigma <= 0):
            y_sigma = 1.0
    else:
        y_mu, y_sigma = 0.0, 1.0

    # init outputs
    df_out["rainfall_est"] = df_out["rainfall"].to_numpy(float)
    df_out["rainfall_var"] = np.nan

    rng = np.random.default_rng(seed)

    filled = []
    for date_id, g in tqdm(df_out.groupby("date_id", sort=True), desc=f"GPR impute by day ({kernel_kind})"):
        g = g.copy().reset_index(drop=True)

        mask_obs = np.isfinite(g["rainfall"].to_numpy(float))
        mask_mis = ~mask_obs

        # nothing to fill
        if mask_mis.sum() == 0:
            g["rainfall_var"] = np.where(mask_obs, 0.0, g["rainfall_var"])
            filled.append(g)
            continue

        # require at least min_obs
        if mask_obs.sum() < min_obs:
            # leave missing as NaN
            g.loc[mask_mis, "rainfall_est"] = np.nan
            g.loc[mask_mis, "rainfall_var"] = np.nan
            g.loc[mask_obs, "rainfall_var"] = 0.0
            filled.append(g)
            continue

        # features
        if kernel_kind == "space":
            X = g[["x_km", "y_km"]].to_numpy(float)
        else:
            X = g[["x_km", "y_km", "Elevation"]].to_numpy(float)

        y = g["rainfall"].to_numpy(float)
        y_std = (y - y_mu) / y_sigma

        X_train = X[mask_obs]
        y_train = y_std[mask_obs]
        X_pred  = X[mask_mis]

        # fit day model
        gpr = make_gpr_model(kind=kernel_kind, nu=nu, n_restarts=n_restarts, random_state=int(rng.integers(0, 2**31-1)))
        gpr.fit(X_train, y_train)

        # predict
        yhat_std, ystd_f = gpr.predict(X_pred, return_std=True)
        var_f = ystd_f**2

        # add noise variance (WhiteKernel)
        noise_var = float(gpr.kernel_.k2.noise_level)

        var_obs_std = var_f + noise_var

        # back-transform
        yhat = y_mu + y_sigma * yhat_std
        vhat = (y_sigma**2) * var_obs_std

        if clip_nonneg:
            yhat = np.maximum(yhat, 0.0)

        # write outputs
        g.loc[mask_mis, "rainfall_est"] = yhat
        g.loc[mask_mis, "rainfall_var"] = vhat

        g.loc[mask_obs, "rainfall_est"] = y[mask_obs]
        g.loc[mask_obs, "rainfall_var"] = 0.0

        filled.append(g)

    return pd.concat(filled, ignore_index=True)