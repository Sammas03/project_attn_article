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


from model import GruModel
from common_dataloader.mutilple_loader import MutilSeqDataModule
from config import parameter
from common_tune.tune_config import trial_name_string,trial_dirname_creator
from util import *




def prepare_daloader():
    path = r'../data/Apt2_2015_hour_weather_bfill.xlsx'
    col_list = ['power', 'temperature', 'humidity', 'dewPoint']
    table = easy_read_data(path).iloc[:3240, :][col_list]
    parameter['gru.input_size'] = table.shape[1]
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    dataloader = MutilSeqDataModule(sc_table, 'power',
                                    history_seq_len=parameter['common.history_seq_len'],
                                    batch_size=parameter['running.batch_size'])
    return dataloader


def best_trails_predict(checkpoint, config, dataloader):

    model = GruModel(config)
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
    model = GruModel(config)
    # training
    trainer = pl.Trainer(
        gpus=parents_config['gpu'],
        fast_dev_run=False,  # 检查程序完整性时候执行
        # limit_train_batches=0.3,
        # limit_val_batches=0.3,
        # limit_test_batches=0.5,
        val_check_interval=10,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=parameter['running.max_epoch'],
        enable_progress_bar=False,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=[
            ModelCheckpoint(monitor='val_loss'),  # 记录验证loss
            # EarlyStopping(monitor="val_loss", mode="min")
            # TuneReportCallback(
            #     {
            #         "v_loss": "test_loss",
            #         # "p_loss": "predict_loss"
            #     },
            #     on="test_end")
            TuneReportCheckpointCallback(
                metrics={
                    "v_loss": "test_loss",
                },
                filename="checkpoint",
                on="test_end")
        ]
    )
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    # trainer.predict(model, dataloader)


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

    resources_per_trial = {"cpu": 1, "gpu": 0.15}
    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="v_loss",
                        mode="min",
                        config=parameter,
                        num_samples=500,
                        # scheduler=scheduler,
                        progress_reporter=reporter,
                        trial_name_creator=trial_name_string,
                        trial_dirname_creator=trial_dirname_creator,
                        local_dir='./ray_results',
                        name="gru_with_weather_tune",
                        )

    print("Best hyperparameters found were: ", analysis.best_config)
    print("Best result", analysis.best_result_df)
    return analysis







if __name__ == '__main__':
    dataloader = prepare_daloader()
    result = tune_train(dataloader)
    # 找到最佳配置和文件，恢复模型，进行预测
    ckp = result.get_best_checkpoint()
    best_config = result.get_best_config()
    # 重新运行模型 进行predict
    best_predict_result = best_trails_predict(ckp, best_config, dataloader)
