from util import *


def get_predict_data_tables(path:str,col_list,train_rate=0.9, rows=2160):
    table = easy_read_data(path).iloc[:rows, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    return sc_table[int(rows * train_rate):rows], sc_list


def predict_result_summary(result):
    reals, predicts = [], []
    for ite in result:
        reals.extend(ite['real_y'][0][0]), predicts.extend(ite['predict_y'][0][0])
    return reals, predicts
