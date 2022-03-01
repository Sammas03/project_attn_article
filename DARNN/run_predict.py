import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler


from DARNN.model import DARNN
from common_dataloader.signal_dataloader import OneSeqDataModule
from common_dataloader.mutilple_loader import MutilSeqDataModule
from DARNN.config import config
from util.dataframe_reader import easy_read_data
from util.scaler_util import easy_mutil_transformer

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # path = r'../data/Apt2_2015_hour.xls'
    # target = pd.read_excel(path).iloc[:2160, -1].values.reshape(-1, 1) # 适应sklearn做reshape
    # path = r'../data/sgsc_match_time_2013_hour_10006414.csv'
    path = r'../data/Apt2_2015_hour_weather_bfill.xlsx'
    col_list = ['power','temperature','humidity','windSpeed','dewPoint']
    table = easy_read_data(path).iloc[:1080,:][col_list]
   # table.to_excel('../data/Apt2_2015_hour_weather_bfill.xlsx')
    sc_table,sc_list = easy_mutil_transformer(table,[])
    # scaler = MinMaxScaler()
    # processed_data = easy_signal_transformer(target, scaler)
    # model
    model = DARNN(config)

    # data
    dataloader = MutilSeqDataModule(sc_table,'power',history_seq_len=config['seq_len'],batch_size=config['batch_size'])
    # dataloader = OneSeqDataModule(processed_data, history_seq_len=24, batch_size=config['running.batch_size'])

    # training
    trainer = pl.Trainer(
        fast_dev_run=False,  # 检查程序完整性时候执行
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        # limit_test_batches=0.5,
        val_check_interval=10,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),  # 记录验证loss
            # EarlyStopping(monitor="val_loss", mode="min")
        ]

    )

    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    trainer.predict(model, dataloader)
