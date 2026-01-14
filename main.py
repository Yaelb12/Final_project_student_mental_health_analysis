import pandas as pd
import data_cleaning


df = pd.read_csv("st_1.csv") 
df_clean=data_cleaning.pre_process(df)
