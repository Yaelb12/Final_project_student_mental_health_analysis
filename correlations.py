import logging
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

logger = logging.getLogger(__name__)

def point_biserial(df, bin_col='Is_STEM', cont_col='Anxiety_Score'):
    logger.info("Running point-biserial correlation")
    if bin_col not in df.columns or cont_col not in df.columns:
        return {'r': np.nan, 'p': np.nan, 'n': 0}
    sub = df[[bin_col, cont_col]].dropna()
    if len(sub) < 3:
        return {'r': np.nan, 'p': np.nan, 'n': len(sub)}
    r, p = pointbiserialr(sub[bin_col], sub[cont_col])
    out = {'r': float(r), 'p': float(p), 'n': len(sub)}
    pd.DataFrame([out]).to_csv("reports/pointbiserial.csv", index=False)
    return out

def cohens_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) /
                     (len(x)+len(y)-2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else np.nan
