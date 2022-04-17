from typing import List
from util.dataframe_reader import easy_read_data
from util.scaler_util import easy_mutil_transformer
from common_dataloader.mutilple_loader import MutilSeqDataModule

col_list = ['power', 'temperature', 'humidity', 'dewPoint']


def easy_prepare_dataloader(path, config):
    batch_size = config['running.batch_size']
    history = config['common.history_seq_len']
    return prepare_daloader(path, col_list, batch_size=batch_size, history_seq_len=history)


def prepare_daloader(path: str,
                     col_list: List[str] = col_list,
                     mian_col='power',
                     batch_size=8,
                     history_seq_len=24,
                     rows=2160):
    table = easy_read_data(path).iloc[:rows, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    dataloader = MutilSeqDataModule(sc_table, mian_col,
                                    history_seq_len=history_seq_len,
                                    batch_size=batch_size)
    return dataloader


def prepare_diff_daloader(path: str,
                          col_list: List[str] = col_list,
                          mian_col='power',
                          batch_size=8,
                          history_seq_len=24,
                          rows=2160):
    table = easy_read_data(path).iloc[:rows, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table.diff(), [])  # 数据在放进之前进行一阶差分
    # data
    dataloader = MutilSeqDataModule(sc_table, mian_col,
                                    history_seq_len=history_seq_len,
                                    batch_size=batch_size)
    return dataloader


# 一阶差分和一阶差分还原

class MyDataloder():
    def __init__(self, path: str,
                 col_list: List[str] = col_list,
                 main_col='power',
                 batch_size=8,
                 history_seq_len=24,
                 rows=2160,
                 diff_end=300,
                 diff=1):
        self.path = path
        self.col_list = col_list
        self.main_col = main_col
        self.batch_size = batch_size
        self.history_seq_len = history_seq_len
        self.rows = rows
        self.diff = diff
        self.table = easy_read_data(path).iloc[:rows + diff_end, :][col_list]
        self.sc_table, self.sc_list = easy_mutil_transformer(self.table, [])  # 数据在放进之前进行一阶差分
        self.diff_sc_table = self.sc_table.diff().fillna(0) # 第一个值的差分为nan 替换为o
        self.norm_table = self.diff_sc_table.iloc[:rows, :]
        self.predict_table = self.diff_sc_table.iloc[rows - self.history_seq_len:, :]

    def _create_daloader(self, table, srate=[0.8, 0.1, 0.1]):
        # data
        dataloader = MutilSeqDataModule(table, self.main_col,
                                        history_seq_len=self.history_seq_len,
                                        batch_size=self.batch_size,
                                        srate=srate)
        return dataloader

    def prepare_daloader(self):
        return self._create_daloader(self.norm_table)

    def prepare_predict_daloader(self):  # 1阶差分数据
        return self._create_daloader(self.predict_table, srate=[0, 0, 1.0])

    def reverse_diff(self,predict_diff:List):
        import pandas as pd
        pd.Series({"predict_diff"})
        odata = self.predict_table[self.main_col].iloc[self.history_seq_len-1:]
        value_predict = predict_diff.add(odata)
        return value_predict

    def inverse_power(self,value):
        return self.sc_list[0].inverse_transform()(value)
