import os
from abc import ABC

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from interface.abstract_dataloader import AbsDataModule


class OneSeqDataModule(AbsDataModule):

    def __init__(self, target, history_seq_len, prediction_horizon=1, batch_size=1, data_succession=True):
        super(OneSeqDataModule, self).__init__(target,
                                               history_seq_len,
                                               prediction_horizon=prediction_horizon,
                                               batch_size=batch_size,
                                               data_succession=data_succession)
    #
    # def test_dataloader(self):
    #     dataset = self._to_dataset(self.test_x, self.test_y)
    #     return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)
    #
    def predict_dataloader(self):
        dataset = self._to_dataset(self.test_x, self.test_y)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)

