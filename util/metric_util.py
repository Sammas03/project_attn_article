from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    mean_squared_error, \
    r2_score  # 批量导入指标算法


def metric_rmse(reals, predicts):
    import numpy as np
    return np.sqrt(mean_squared_error(reals, predicts))


def easy_metric(reals, predicts):
    r2 = r2_score(reals, predicts)
    evs = explained_variance_score(reals, predicts)
    mae = mean_absolute_error(reals, predicts)
    mse = mean_squared_error(reals, predicts)
    rmse = metric_rmse(reals, predicts)
    seq_metrics_score_dict = {"r2": r2, "evs": evs, "mae": mae, "mse": mse, "rmse": rmse}
    return seq_metrics_score_dict
