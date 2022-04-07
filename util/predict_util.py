from util import *
from ray.tune import ExperimentAnalysis


# 从文件结果中加载训练好的最佳模型
def load_model(exp_path, model):
    exp = ExperimentAnalysis(
        default_metric='v_loss',
        default_mode='min',
        experiment_checkpoint_path=exp_path)
    ckp = "{}/{}".format(exp.best_checkpoint, 'checkpoint')
    best_config = exp.get_best_config()
    bmodel = model.load_from_checkpoint(checkpoint_path=ckp, config=best_config)
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
