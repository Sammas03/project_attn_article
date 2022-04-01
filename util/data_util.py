from typing import List

from util import *
from common_dataloader.mutilple_loader import MutilSeqDataModule

umass_apt2 = {

    'path': '../data/Apt2_2015_hour_weather_bfill.xlsx',
    'col_list': ['power', 'temperature', 'humidity', 'dewPoint']
}


def prepare_daloader(path: str, col_list:List,mian_col='power', batch_size=8, history_seq_len=24, rows=2180):
    # path = r'../data/Apt2_2015_hour_weather_bfill.xlsx'
    # col_list = ['power', 'temperature', 'humidity', 'dewPoint']
    table = easy_read_data(path).iloc[:rows, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    dataloader = MutilSeqDataModule(sc_table, 'power',
                                    history_seq_len=history_seq_len,
                                    batch_size=batch_size)
    return dataloader
