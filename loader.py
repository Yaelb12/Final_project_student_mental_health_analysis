import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_clean(path="clean_data.csv"):
    logger.info("Loading cleaned dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    return df
