import pandas as pd
import numpy as np
import os

tail_dict = {'.xlsx': pd.read_excel, '.xls': pd.read_excel, '.csv': pd.read_csv}


def easy_read_data(path):
    # analysis tail
    tail = os.path.splitext(path)[-1]
    proc_fun = tail_dict[tail]
    return proc_fun(path)
