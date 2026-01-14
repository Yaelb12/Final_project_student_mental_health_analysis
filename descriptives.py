import logging
import pandas as pd

logger = logging.getLogger(__name__)

def describe_by_group(df, group='Is_STEM', outcome='Anxiety_Score'):
    logger.info("Computing descriptive statistics by group: %s", group)
    if group not in df.columns or outcome not in df.columns:
        logger.warning("Columns missing")
        return pd.DataFrame()
    desc = df.groupby(group)[outcome].agg(['mean', 'std', 'count'])
    desc.to_csv("reports/describe_by_group.csv")
    logger.info("Saved descriptive statistics")
    return desc
