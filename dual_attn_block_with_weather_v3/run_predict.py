import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from util import *
from common_dataloader.mutilple_loader import MutilSeqDataModule
from dual_attn_block_with_weather_v3.model import MainModel
from dual_attn_block_with_weather_v3.config import parameter


if __name__ == '__main__':

    path = r'../data/Apt2_2015_hour_weather_bfill.xlsx'
    col_list = ['power', 'temperature', 'humidity', 'dewPoint']
    table = easy_read_data(path).iloc[:1080, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table, [])

    # model
    model = MainModel(parameter)

    # data
    dataloader = MutilSeqDataModule(sc_table,'power',history_seq_len=24, batch_size=4)

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
