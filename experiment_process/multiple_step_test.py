import pandas as pd
import torch




def move_window_forecasting(model,
                            history_len: int,
                            step: int,
                            data_tables: pd.DataFrame) -> ([], []):
    '''
    每次只进行一组滑动窗口预测
    :param model:
    :param history_len:
    :param step:
    :param data_tables:
    :return:
    '''
    model.eval()
    model.freeze()
    predict_y = []
    reals = []
    datas = data_tables.copy().values
    # 根据步长对数据集进行重组
    for t in range(step):
        seqx = datas[t:history_len + t, :]
        seqx = torch.Tensor(seqx).unsqueeze(0)
        seqy = model(seqx)[0].item()
        predict_y.append(seqy)
        reals.append(datas[history_len + t, 0])
        datas[history_len + t, 0] = seqy
    return predict_y, reals


def cyc_move_window_forecasting(model,
                                history_len: int,
                                step: int,
                                data_tables: pd.DataFrame) -> ([], []):
    '''
        多次循环预测
    :param model:
    :param history_len:
    :param step:
    :param data_tables:
    :return:
    '''
    rows = data_tables.shape[0]
    reals = []
    predicts = []
    for i in range(0, rows - history_len - step, step):
        r, y = move_window_forecasting(model, history_len, step, data_tables.iloc[i:i + history_len + step, :])
        reals.append(r)
        predicts.append(y)
    return reals, predicts
