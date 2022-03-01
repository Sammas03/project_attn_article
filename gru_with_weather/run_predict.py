import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler

from pytorch_lightning import seed_everything

from gru_with_weather.model import GruModel
from common_dataloader.mutilple_loader import MutilSeqDataModule
from gru_with_weather.config import parameter

from util import *

if __name__ == '__main__':

    path = r'../data/Apt2_2015_hour_weather_bfill.xlsx'
    col_list = ['power', 'temperature', 'humidity', 'dewPoint']
    table = easy_read_data(path).iloc[:1080, :][col_list]
    parameter['gru.input_size'] = table.shape[1]
    # table.to_excel('../data/Apt2_2015_hour_weather_bfill.xlsx')
    sc_table, sc_list = easy_mutil_transformer(table, [])

    # model
    model = GruModel(parameter)

    # data
    dataloader = MutilSeqDataModule(sc_table,'power',history_seq_len=6, batch_size=4)

    # training
    trainer = pl.Trainer(
        fast_dev_run=False,  # 检查程序完整性时候执行
        # show_progress_bar=False,
        limit_train_batches=0.3,
        limit_val_batches=0.5,
       # limit_test_batches=0.5,
        val_check_interval=10,
        #gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=400,
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),  # 记录验证loss
            #EarlyStopping(monitor="val_loss", mode="min")
        ]

    )

    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    trainer.predict(model,dataloader)
