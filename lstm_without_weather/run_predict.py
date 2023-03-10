import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import ray
from ray import tune

from util import *

from pytorch_lightning import seed_everything

seed_everything(2022, workers=True)

from model import LstmModel as BasicModel
from common_dataloader.signal_dataloader import OneSeqDataModule
from config import parameter
from common_tune.tune_config import trial_name_string, trial_dirname_creator
from util import *


def prepare_daloader():
    path = r'../data/HomeA-meter2_2014_clean1.xlsx'
    col_list = ['sum']
    table = easy_read_data(path).iloc[:6480, :][col_list]
    parameter['lstm.input_size'] = 1 # 只输入时序数据
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    dataloader = OneSeqDataModule(sc_table,
                                    history_seq_len=parameter['common.history_seq_len'],
                                    batch_size=parameter['running.batch_size'])
    return dataloader


def best_trails_predict(checkpoint, config, dataloader):
    model = BasicModel(config)
    # training
    trainer = pl.Trainer(
        gpus=parents_config['gpu'],
        fast_dev_run=False,  # 检查程序完整性时候执行
        # limit_train_batches=0.3,
        # limit_val_batches=0.3,
        # limit_test_batches=0.5,
        # val_check_interval=10,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=parameter['running.max_epoch'],
        enable_progress_bar=False,
        callbacks=[]
    )
    predict_result = trainer.predict(model, dataloader, ckpt_path=checkpoint)
    return predict_result


def lightning_run(config, dataloader, checkpoint_dir=None):
    model = BasicModel(config)
    # training
    devices = [0] if config['gpu'] else None
    trainer = pl.Trainer(
        accelerator='auto',
        devices=devices,
        # gpus=parents_config['gpu'],
        fast_dev_run=config['test'],  # 检查程序完整性时候执行
        # limit_train_batches=0.3,
        # limit_val_batches=0.3,
        # limit_test_batches=0.5,
        check_val_every_n_epoch=2,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=config['running.max_ep'
                          'och'],
        min_epochs=config['running.min_epoch'],
        enable_progress_bar=True,
        accumulate_grad_batches=2,  # 梯度累加获取跟大batch_size相同的效果
        log_every_n_steps=50,
        # gradient_clip_val=2.5,
        callbacks=[
            EarlyStopping(monitor="val_loss", min_delta=0.0, patience=10, verbose=False, mode="min"),
            # StochasticWeightAveraging(swa_lrs=1e-2)
        ]
    )
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    trainer.predict(model, dataloader)


def tune_train(dataloader):
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

    resources_per_trial = {"cpu": 1, "gpu": 1}
    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="v_loss",
                        mode="min",
                        config=parameter,
                        num_samples=1,
                        # scheduler=scheduler,
                        progress_reporter=reporter,
                        trial_name_creator=trial_name_string,
                        trial_dirname_creator=trial_dirname_creator,
                        local_dir='./ray_results',
                        name="lstm_with_weather_tune",
                        )

    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best result", analysis.best_result_df)
    return analysis


if __name__ == '__main__':
    dataloader = prepare_daloader()
    lightning_run(config=parameter,dataloader=dataloader)

