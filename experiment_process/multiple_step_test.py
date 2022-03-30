import pandas as pd
import torch

def move_window_forecasting(model,
                            history_len: int,
                            step: int,
                            data_tables: pd.DataFrame) -> ([], []):
    model.eval()
    model.freeze()
    predict_y = []
    reals = []
    datas = data_tables.copy()
    # 根据步长对数据集进行重组
    for t in range(step):
        seqx = datas.iloc[t:history_len + t, :]
        seqx = torch.Tensor(seqx)
        seqy = model(seqx).detech().item()
        predict_y.append(seqy)
        reals.append(datas.iloc[history_len + t, 0])
        datas.iloc[history_len + t, 0] = seqy
    return predict_y
