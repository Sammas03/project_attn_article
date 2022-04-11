from typing import List
from util.dataframe_reader import easy_read_data
from util.scaler_util import easy_mutil_transformer
from common_dataloader.mutilple_loader import MutilSeqDataModule

col_list = ['power', 'temperature', 'humidity', 'dewPoint']


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
