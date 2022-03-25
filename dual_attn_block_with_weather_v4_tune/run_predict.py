import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import ray
from ray import tune

from util import *
from common_config.common_config import parents_config
from common_dataloader.mutilple_loader import MutilSeqDataModule
from dual_attn_block_with_weather_v4_tune.model import MainModel
from dual_attn_block_with_weather_v4_tune.config import parameter


def trial_name_string(trial):
    return str(trial)


def trial_dirname_creator(trail):
    return str(trail)


def lightning_run(config, dataloader, checkpoint_dir=None):
    model = MainModel(config)
    # training
    trainer = pl.Trainer(
        gpus=parents_config['gpu'],
        fast_dev_run=False,  # 检查程序完整性时候执行
        #limit_train_batches=0.3,
        #limit_val_batches=0.3,
        # limit_test_batches=0.5,
        #val_check_interval=10,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=parameter['running.max_epoch'],
        enable_progress_bar=False,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),  # 记录验证loss
            # EarlyStopping(monitor="val_loss", mode="min")
            TuneReportCallback(
                {
                    "v_loss": "test_loss",
                    # "p_loss": "predict_loss"
                },
                on="test_end")
        ]

    )

    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    trainer.predict(model, dataloader)


def tune_train():
    path = r'../data/Apt2_2015_hour_weather_bfill.xlsx'
    col_list = ['power', 'temperature', 'humidity', 'dewPoint']
    table = easy_read_data(path).iloc[:3240, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    dataloader = MutilSeqDataModule(sc_table, 'power',
                                    history_seq_len=parameter['common.history_seq_len'],
                                    batch_size=parameter['running.batch_size'])

    ray.init()
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        max_report_frequency=30,
        print_intermediate_tables=False,
        # parameter_columns=[],
        metric_columns=["v_loss"])

    train_fn_with_parameters = tune.with_parameters(lightning_run,
                                                    dataloader=dataloader
                                                    )

    resources_per_trial = {"cpu": 1, "gpu": 0.15}
    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="v_loss",
                        mode="min",
                        config=parameter,
                        num_samples=500,
                        local_dir='./ray_results',
                        # scheduler=scheduler,
                        progress_reporter=reporter,
                        trial_name_creator=trial_name_string,
                        trial_dirname_creator=trial_dirname_creator,
                        name="model_with_weather_v4")

    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best result", analysis.best_result_df)


if __name__ == '__main__':
    tune_train()
