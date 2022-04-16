from util import easy_read_data, easy_mutil_transformer
from ray.tune import ExperimentAnalysis
import torch
import pytorch_lightning as pl
from common_config.common_config import parents_config


# 从文件结果中加载训练好的最佳模型
def load_model_from_file(exp_path, model, device=torch.device('cpu')):
    exp = ExperimentAnalysis(
        default_metric='v_loss',
        default_mode='min',
        experiment_checkpoint_path=exp_path)
    ckp = "{}/{}".format(exp.best_checkpoint, 'checkpoint')
    best_config = exp.get_best_config()
    bmodel = model.load_from_checkpoint(checkpoint_path=ckp, config=best_config, map_location=device)
    return bmodel, exp


def get_predict_data_tables(path: str, col_list, train_rate=0.9, rows=2160):
    table = easy_read_data(path).iloc[:rows, :][col_list]
    sc_table, sc_list = easy_mutil_transformer(table, [])
    # data
    return sc_table[int(rows * train_rate):rows], sc_list


def predict_result_summary(result):
    reals, predicts = [], []
    for ite in result:
        reals.extend(ite['real_y'][0][0]), predicts.extend(ite['predict_y'][0][0])
    return reals, predicts


def easy_predict_from_result(exp_result, run_model, dataloader):
    ckp = "{}/{}".format(exp_result.best_checkpoint, 'checkpoint')
    best_config = exp_result.get_best_config()
    bmodel = run_model.load_from_checkpoint(checkpoint_path=ckp, config=best_config)
    trainer = pl.Trainer(gpus=parents_config['gpu'])
    trainer.predict(bmodel, dataloader)


def easy_predict_from_file(exp_path,data_path,run_model):
    from exp_util import easy_load_exp
    exp = easy_load_exp(exp_path)
    config = exp.get_best_config()
    from data_util import easy_prepare_dataloader
    dataloader = easy_prepare_dataloader(data_path,config)
    easy_predict_from_result(exp,run_model,dataloader)
    return exp,dataloader

