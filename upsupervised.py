# Requires: pip install ordinalcorr
import numpy as np
import pandas as pd
from scipy import stats

# Try to import polychoric; if not available, we'll fall back to Spearman
try:
    from ordinalcorr import polychoric
    _HAS_POLY = True
except Exception:
    _HAS_POLY = False

def polychoric_matrix(df):
    """
    Compute full polychoric correlation matrix for ordinal columns in df.
    Returns: pandas.DataFrame (corr matrix)
    Raises ValueError if polychoric fails for all pairs.
    """
    cols = df.columns.tolist()
    p = len(cols)
    mat = np.eye(p)
    if not _HAS_POLY:
        raise ImportError("ordinalcorr.polychoric not available. Install 'ordinalcorr' or use fallback.")
    for i in range(p):
        xi = df.iloc[:, i].values
        for j in range(i+1, p):
            yj = df.iloc[:, j].values
            # polychoric returns a float estimate of correlation
            r = polychoric(xi, yj)
            mat[i, j] = mat[j, i] = float(r)
    return pd.DataFrame(mat, index=cols, columns=cols)

def spearman_matrix(df):
    """
    Compute Spearman correlation matrix as a fallback when polychoric is unavailable.
    Returns: pandas.DataFrame
    """
    return df.corr(method='spearman')

def kmo_from_corr(corr):
    """
    Compute overall KMO and KMO per item from a correlation matrix.
    Input: numpy array or pandas DataFrame (correlation matrix).
    Returns: (kmo_overall, pd.Series of kmo_per_item)
    """
    R = np.asarray(corr)
    invR = np.linalg.pinv(R)
    partial = -1 * (invR / np.sqrt(np.outer(np.diag(invR), np.diag(invR))))
    np.fill_diagonal(partial, 0.0)
    sum_sq_R = np.sum(np.square(R)) - np.sum(np.square(np.diag(R)))
    sum_sq_partial = np.sum(np.square(partial))
    kmo_overall = sum_sq_R / (sum_sq_R + sum_sq_partial)
    kmo_per_item = np.sum(np.square(R), axis=0) / (np.sum(np.square(R), axis=0) + np.sum(np.square(partial), axis=0))
    idx = corr.columns if hasattr(corr, 'columns') else None
    return float(kmo_overall), pd.Series(kmo_per_item, index=idx)

def bartlett_from_corr(corr, n_obs):
    """
    Bartlett's test of sphericity from a correlation matrix and sample size.
    Returns: (chi2, p_value, df)
    """
    R = np.asarray(corr)
    p = R.shape[0]
    detR = np.linalg.det(R)
    if detR <= 0:
        raise ValueError("Determinant non-positive; check multicollinearity or sparse contingency cells.")
    chi2 = -(n_obs - 1 - (2*p + 5)/6.0) * np.log(detR)
    dfree = p * (p - 1) // 2
    p_value = 1 - stats.chi2.cdf(chi2, dfree)
    return float(chi2), float(p_value), int(dfree)

def run_kmo_bartlett_for_group(df, cols, use_polychoric=True, dropna=True):
    """
    Wrapper to run polychoric (or fallback) then KMO and Bartlett for a list of columns.
    Inputs:
      - df: pandas.DataFrame with your data
      - cols: list of column names to include
      - use_polychoric: True to attempt polychoric, False to force Spearman
      - dropna: if True drop rows with NaN in these cols; otherwise expect pre-imputed df
    Returns: dict with keys: 'corr_matrix','kmo_overall','kmo_per_item','bartlett'
    """
    sub = df[cols].copy()
    if dropna:
        sub = sub.dropna()
    if sub.shape[0] == 0:
        raise ValueError("No observations after dropna; check your data or set dropna=False.")
    # compute correlation matrix
    if use_polychoric and _HAS_POLY:
        try:
            corr = polychoric_matrix(sub)
        except Exception as e:
            # fallback to Spearman if polychoric fails for some reason
            corr = spearman_matrix(sub)
    else:
        corr = spearman_matrix(sub)
    # compute KMO and Bartlett
    kmo_overall, kmo_items = kmo_from_corr(corr)
    chi2, pval, dfree = bartlett_from_corr(corr, n_obs=sub.shape[0])
    return {
        "corr_matrix": corr,
        "kmo_overall": kmo_overall,
        "kmo_per_item": kmo_items,
        "bartlett": {"chi2": chi2, "p_value": pval, "df": dfree},
        "n_obs": sub.shape[0]
    }

# Example usage (call this from your main script):
# df_clean = pd.read_csv("your_data.csv")  # or however you load data
# group1 = ["Sleep_Quality","Physical_Activity","Diet_Quality","Substance_Use"]
# group2 = ["Stress_Level","Depression_Score","Anxiety_Score","Financial_Stress"]
# res1 = run_kmo_bartlett_for_group(df_clean, group1, use_polychoric=True)
# res2 = run_kmo_bartlett_for_group(df_clean, group2, use_polychoric=True)
# print(res1["kmo_overall"], res1["bartlett"])
# print(res2["kmo_overall"], res2["bartlett"])
