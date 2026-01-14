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


def compute_baseline(df, group, use_polychoric=True):
    """Return baseline metrics for the full group."""
    return run_kmo_bartlett_for_group(df, group, use_polychoric)

def kmo_items(df, group, use_polychoric=True):
    """Return KMO per item (Series)."""
    res = run_kmo_bartlett_for_group(df, group, use_polychoric)
    return res["kmo_per_item"]

def evaluate_drops(df, group, use_polychoric=True):
    """
    Evaluate dropping each item: return DataFrame with kmo change and Bartlett p.
    Rows sorted by kmo_change descending.
    """
    base = compute_baseline(df, group, use_polychoric)
    rows = []
    for item in group:
        reduced = [g for g in group if g != item]
        try:
            res = run_kmo_bartlett_for_group(df, reduced, use_polychoric)
            rows.append({
                "dropped": item,
                "kmo_after": res["kmo_overall"],
                "kmo_change": res["kmo_overall"] - base["kmo_overall"],
                "bartlett_p_after": res["bartlett"]["p_value"],
                "n_obs": res["n_obs"]
            })
        except Exception:
            rows.append({"dropped": item, "kmo_after": None, "kmo_change": None, "bartlett_p_after": None, "n_obs": None})
    return pd.DataFrame(rows).sort_values("kmo_change", ascending=False).reset_index(drop=True)

def apply_best_drop(df, group, out_csv="clean_data_reduced.csv", use_polychoric=True):
    """
    Drop the best single item only if KMO improves and Bartlett remains significant.
    Returns (dropped_item, new_metrics) or (None, None).
    """
    eval_df = evaluate_drops(df, group, use_polychoric)
    if eval_df.empty:
        return None, None
    best = eval_df.iloc[0]
    if pd.notna(best["kmo_change"]) and best["kmo_change"] > 0 and best["bartlett_p_after"] is not None and best["bartlett_p_after"] < 0.05:
        new_group = [g for g in group if g != best["dropped"]]
        res = run_kmo_bartlett_for_group(df, new_group, use_polychoric)
        # save only the reduced columns for downstream analysis
        df[new_group].to_csv(out_csv, index=False)
        return best["dropped"], res
    return None, None

 #Example usage (call this from your main script)
if __name__ == "__main__":
    # load data
    df_clean = pd.read_csv("clean_data.csv")
    group1 = ["Sleep_Quality","Physical_Activity","Diet_Quality","Substance_Use"]
    group2 = ["Stress_Level","Depression_Score","Anxiety_Score","Financial_Stress"]

    # 1) baseline KMO/Bartlett using polychoric if available
    print("Running baseline KMO/Bartlett (polychoric if available)...")
    res1 = run_kmo_bartlett_for_group(df_clean, group1, use_polychoric=True)
    res2 = run_kmo_bartlett_for_group(df_clean, group2, use_polychoric=True)
    print("Group1 KMO:", res1["kmo_overall"], "Bartlett p:", res1["bartlett"]["p_value"])
    print("Group2 KMO:", res2["kmo_overall"], "Bartlett p:", res2["bartlett"]["p_value"])

    # 2) KMO per item and correlation matrix (polychoric matrix if used)
    print("\nKMO per item (group1):")
    print(res1["kmo_per_item"])
    print("\nCorrelation matrix used (group1):")
    print(res1["corr_matrix"])

    # 3) quick distribution checks to spot low-variance items
    print("\nValue counts (top categories) and variance:")
    for c in group1:
        print(f"\nColumn: {c}")
        print(df_clean[c].value_counts(normalize=True).head())
        print("Variance:", df_clean[c].var())

    # 4) evaluate single-item drops (polychoric)
    print("\nEvaluate single-item drops (sorted by KMO improvement):")
    eval_df = evaluate_drops(df_clean, group1, use_polychoric=True)
    print(eval_df)

    # 5) try recoding Substance_Use to binary and re-evaluate
    print("\nTry recoding Substance_Use to binary and re-evaluate:")
    df_clean["Substance_Use_bin"] = (df_clean["Substance_Use"] != 1).astype(int)
    group_bin = ["Sleep_Quality","Physical_Activity","Diet_Quality","Substance_Use_bin"]
    try:
        res_bin = run_kmo_bartlett_for_group(df_clean, group_bin, use_polychoric=True)
        print("Binary Substance KMO:", res_bin["kmo_overall"], "Bartlett p:", res_bin["bartlett"]["p_value"])
        print("KMO per item (binary):\n", res_bin["kmo_per_item"])
    except Exception as e:
        print("Binary recode evaluation failed:", e)

    # 6) try dropping 1 or 2 items (polychoric) and show top candidates
    print("\nTry dropping 1 or 2 items (polychoric) — top results by KMO:")
    from itertools import combinations
    results = []
    for r in (1,2):
        for combo in combinations(group1, r):
            reduced = [g for g in group1 if g not in combo]
            try:
                res = run_kmo_bartlett_for_group(df_clean, reduced, use_polychoric=True)
                results.append((combo, res["kmo_overall"], res["bartlett"]["p_value"]))
            except Exception:
                pass
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    for item in results_sorted[:10]:
        print(item)

import pandas as pd
import prince
import numpy as np

df = pd.read_csv("clean_data.csv")
group = ["Sleep_Quality","Physical_Activity","Diet_Quality","Substance_Use"]
df_cat = df[group].astype(str).dropna()   # drop rows with NA in these cols

# quick sanity checks
print("Shape:", df_cat.shape)
for c in df_cat.columns:
    print(c, "unique:", df_cat[c].nunique(), "top:", df_cat[c].value_counts(normalize=True).head().to_dict())

# try MCA with numpy engine
# MCA / fallback using one-hot + TruncatedSVD
import pandas as pd
import prince
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
import numpy as np

df = pd.read_csv("clean_data.csv")
group = ["Sleep_Quality","Physical_Activity","Diet_Quality","Substance_Use"]
df_cat = df[group].astype(str).dropna()

print("Shape:", df_cat.shape)
for c in df_cat.columns:
    print(c, "unique:", df_cat[c].nunique(), "top:", df_cat[c].value_counts(normalize=True).head().to_dict())

# Try MCA with supported engine (try 'scipy' or 'sklearn')
for engine in ('scipy', 'sklearn', 'fbpca'):
    try:
        print(f"\nTrying prince.MCA with engine='{engine}'")
        mca = prince.MCA(n_components=4, n_iter=5, copy=True, check_input=True, engine=engine)
        mca = mca.fit(df_cat)
        print("Explained inertia:", mca.explained_inertia_)
        scores = mca.transform(df_cat)
        print("Scores shape:", scores.shape)
        break
    except Exception as e:
        print(f"engine='{engine}' failed:", e)
else:
    # fallback: one-hot + TruncatedSVD
    print("\nAll prince engines failed — falling back to one-hot + TruncatedSVD")
    # use pandas.get_dummies (safe and simple)
    X = pd.get_dummies(df_cat, drop_first=False).astype(int)
    print("One-hot shape:", X.shape)
    svd = TruncatedSVD(n_components=2, random_state=0)
    comps = svd.fit_transform(X)
    print("TruncatedSVD explained variance (approx):", svd.explained_variance_ratio_)
    print("Components shape:", comps.shape)

