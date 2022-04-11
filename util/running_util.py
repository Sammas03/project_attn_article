import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from functools import partial
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
from common_tune.tune_config import trial_name_string, trial_dirname_creator

seed_everything(2022, workers=True)


def lightning_run(config,run_model, dataloader,model_callbacks:List, checkpoint_dir=None):
    model = run_model(config)
    # training
    trainer = pl.Trainer(
        gpus=parents_config['gpu'],
        fast_dev_run=config['test'],  # 检查程序完整性时候执行
        # limit_train_batches=0.3,
        # limit_val_batches=0.3,
        # limit_test_batches=0.5,
        val_check_interval=0.5,
        # gradient_clip_val=0.3,  # 梯度裁剪
        max_epochs=config['running.max_epoch'],
        enable_progress_bar=config['test'],
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        callbacks=model_callbacks
    )
    trainer.fit(model, dataloader)
    trainer.test(model, dataloader)
    # trainer.predict(model, dataloader)


def tune_train(run_model,
               dataloader,
               config,
               model_callbacks,
               saving_name,
               resources_per_trial,
               local_dir,
               num_samples=1):
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        max_report_frequency=30,
        print_intermediate_tables=False,
        # parameter_columns=[],
        metric_columns=["v_loss"])
    train_fn_with_parameters = partial(lightning_run,
                                       run_model=run_model,
                                       dataloader=dataloader,
                                       model_callbacks=model_callbacks
                                       )
    # train_fn_with_parameters = tune.with_parameters(lightning_run,
    #                                                 run_model=run_model,
    #                                                 dataloader=dataloader,
    #                                                 model_callbacks=callbacks
    #                                                 )

    analysis = tune.run(partial(lightning_run,
                               run_model=run_model,
                               dataloader=dataloader,
                               model_callbacks=model_callbacks),
                        resources_per_trial=resources_per_trial,
                        metric="v_loss",
                        mode="min",
                        config=config,
                        num_samples=num_samples,
                        # scheduler=scheduler,
                        progress_reporter=reporter,
                        trial_name_creator=trial_name_string,
                        trial_dirname_creator=trial_dirname_creator,
                        local_dir=local_dir,
                        name=saving_name,
                        )
    return analysis


def easy_run(data_path, run_model, config, saving_name, local_dir,num_samples=1):
    ray.init(local_mode=config['test'])
    resources_per_trial = {"cpu": 1, "gpu": 0.2} if parents_config['gpu'] else {"cpu": 1, "gpu": 0}
    dataloader = prepare_daloader(data_path,
                                  batch_size=config['running.batch_size'],
                                  history_seq_len=config['common.history_seq_len'])
    model_callbacks = [
        ModelCheckpoint(monitor='val_loss'),  # 记录验证loss
        TuneReportCheckpointCallback(
            metrics={
                "v_loss": "test_loss",
            },
            filename="checkpoint",
            on="test_end")
    ]
    result = tune_train(run_model=run_model,
                        dataloader=dataloader,
                        config=config,
                        model_callbacks=model_callbacks,
                        saving_name=saving_name,
                        resources_per_trial=resources_per_trial,
                        num_samples=num_samples)
    # 找到最佳配置和文件，恢复模型，进行预测
    ckp = "{}/{}".format(result.best_checkpoint, 'checkpoint')
    best_config = result.get_best_config()
    bmodel = run_model.load_from_checkpoint(checkpoint_path=ckp, config=best_config)
    trainer = pl.Trainer(gpus=parents_config['gpu'])
    trainer.predict(bmodel, dataloader)
    return result, dataloader
