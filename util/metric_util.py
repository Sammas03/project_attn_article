from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    mean_squared_error, \
    r2_score  # 批量导入指标算法


def rmse(reals, predicts):
    import numpy as np
    return np.sqrt(mean_squared_error(reals, predicts))


def easy_metric(reals, predicts):
    model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    seq_metrics_score_list = []
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(reals, predicts)  # 计算每个回归指标结果
        seq_metrics_score_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    seq_metrics_score_list.append(rmse(reals, predicts))
    return seq_metrics_score_list
