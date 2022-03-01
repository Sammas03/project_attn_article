import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler

from pytorch_lightning import seed_everything

from dual_attn_block.model import MainModel
from common_dataloader.signal_dataloader import OneSeqDataModule
from dual_attn_block.config import parameter
from util.dataframe_reader import easy_read_data
from util.scaler_util import easy_signal_transformer

if __name__ == '__main__':
    path = r'../data/Apt2_2015_hour.xls'
    # target = pd.read_excel(path).iloc[:2160, -1].values.reshape(-1, 1) # 适应sklearn做reshape
    # path = r'../data/sgsc_match_time_2013_hour_10006414.csv'

    target = easy_read_data(path).iloc[:1080, -1]  # 适应sklearn做reshape
    scaler = MinMaxScaler()
    processed_data = easy_signal_transformer(target, scaler)
    # model
    model = MainModel(parameter)

    # data
    dataloader = OneSeqDataModule(processed_data, history_seq_len=24, batch_size=parameter['running.batch_size'])

    # training
    trainer = pl.Trainer(
        fast_dev_run=False,  # 检查程序完整性时候执行
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        # limit_test_batches=0.5,
        val_check_interval=10,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),  # 记录验证loss
           # EarlyStopping(monitor="val_loss", mode="min")
        ]

    )

    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    trainer.predict(model, dataloader)
