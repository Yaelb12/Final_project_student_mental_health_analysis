import pandas as pd
import data_cleaning
import stats_analysis

df = pd.read_csv("st_1.csv") 
df_clean=data_cleaning.pre_process(df)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s")

logger = logging.getLogger(__name__)
stats_analysis.run_all(df_clean)
