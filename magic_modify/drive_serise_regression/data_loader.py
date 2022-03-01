import os
from abc import ABC

import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from interface.abstract_dataloader import AbsDataModule
from util import *

class MutilSeqDataModule(AbsDataModule):

    def __init__(self,
                 table,
                 main_col,
                 history_seq_len,
                 prediction_horizon=1,
                 batch_size=1,
                 srate=[0.7, 0.2, 0.1],
                 data_succession=True):
        # 对所有列做minmax归一化。
        # 输入到网络就变成了 batch,col,history_seq
        # self.sup_scal = ColumnTransformer(
        #     transformers=[main_col],
        #     remainder=MinMaxScaler()
        # )
        self.history_seq_len = history_seq_len
        self.batch_size = batch_size
        self.col_size = table.shape[1] - 1
        self.prediction_horizon = prediction_horizon
        step = prediction_horizon if data_succession else history_seq_len
        data_size = table.shape[0]
        x_seqs, y_seqs = [], []
        cols = [i for i in table.columns if i != main_col]
        for i in range(0, data_size - history_seq_len - prediction_horizon, step):
            x = table[cols].iloc[i:i + history_seq_len, :]
            y = table[main_col].iloc[i + history_seq_len:i + history_seq_len + prediction_horizon]
            x_seqs.append(torch.FloatTensor(x.values.T))
            y_seqs.append(torch.FloatTensor(y.values).view(1, -1))
        self.train_x, self.val_x, self.test_x = self._split(x_seqs, *srate)
        self.train_y, self.val_y, self.test_y = self._split(y_seqs, *srate)

    def test_dataloader(self):
        dataset = self._to_dataset(self.train_x[:50], self.train_y[:50])
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)

    def predict_dataloader(self):
        dataset = self._to_dataset(self.test_x, self.test_y)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)


    def _to_dataset(self, x, y):
        x_var = torch.cat(x).view(-1, self.col_size,self.history_seq_len)
        y_var = torch.cat(y)
        return TensorDataset(x_var, y_var)

