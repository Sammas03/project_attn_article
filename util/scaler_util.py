import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def easy_signal_transformer(target, sc_fun=None):
    # print(isinstance(target, pd.DataFrame))
    # print(isinstance(target, pd.DataFrame))
    # print(isinstance(target, pd.Series))
    # if(len(target.shape)==1):
    if sc_fun is None:
        return target.values
    # 判断shape
    process_data = sc_fun.fit_transform(target.values.reshape(-1, 1)).squeeze(1)
    return process_data


def easy_signal_inverse(target, sc_fun):
    assert sc_fun is not None
    process_data = sc_fun.inverse_transform(target.values.reshape(-1, 1)).squeeze(1)
    return process_data


def easy_mutil_transformer(table: pd.DataFrame, sc_list=None) -> (pd.DataFrame, list):
    '''

    :param table: 需要批量转换的DataFrame
    :param sc_list: 转换器列表 默认[]则使用MinMaxScaler
    :return:
    '''
    col_size = table.shape[1]
    sc_table = table.copy()
    columns = table.columns
    if sc_list is None:
        return table
    if len(sc_list) == 0:
        sc_list = [MinMaxScaler() for i in range(col_size)]

    for col, sc_fun in zip(columns, sc_list):
        sc_table[col] = pd.Series(easy_signal_transformer(table[col], sc_fun))

    return sc_table, sc_list


def easy_mutil_inverse(table: pd.DataFrame, sc_list):
    assert len(sc_list) != 0 and sc_list is not None
    col_size = table.shape[1]
    sc_table = table.copy()
    columns = table.columns

    for col, sc_fun in zip(columns, sc_list):
        sc_table[col] = pd.Series(easy_signal_inverse(table[col], sc_fun))

    return sc_table, sc_list
