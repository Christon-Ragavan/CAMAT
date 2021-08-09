import numpy as np
import pandas as pd


def getVoice(df_data:pd.DataFrame):
    v = df_data['Voice']
    return list(set(v))
