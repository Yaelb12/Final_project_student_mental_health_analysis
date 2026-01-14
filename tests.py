import logging
import numpy as np
import pandas as pd
from scipy import stats
from correlations import cohens_d

logger = logging.getLogger(__name__)

def welch_test(df, group_col='Is_STEM', outcome='Anxiety_Score'):
    logger.info("Running Welch t-test")
    if group_col not in df.columns or outcome not in df.columns:
        return {'t': np.nan, 'p': np.nan, 'cohens_d': np.nan}
    a = df[df[group_col] == 1][outcome].dropna()
    b = df[df[group_col] == 0][outcome].dropna()
    if len(a) < 2 or len(b) < 2:
        return {'t': np.nan, 'p': np.nan, 'cohens_d': np.nan}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)
    out = {'t': float(t), 'p': float(p), 'cohens_d': float(d)}
    pd.DataFrame([out]).to_csv("reports/welch_results.csv", index=False)
    return out

def mann_whitney(df, group_col='Is_STEM', outcome='Anxiety_Score'):
    logger.info("Running Mann-Whitney U test")
    if group_col not in df.columns or outcome not in df.columns:
        return {'U': np.nan, 'p': np.nan}
    a = df[df[group_col] == 1][outcome].dropna()
    b = df[df[group_col] == 0][outcome].dropna()
    if len(a) < 1 or len(b) < 1:
        return {'U': np.nan, 'p': np.nan}
    U, p = stats.mannwhitneyu(a, b)
    out = {'U': float(U), 'p': float(p)}
    pd.DataFrame([out]).to_csv("reports/mannwhitney.csv", index=False)
    return out
