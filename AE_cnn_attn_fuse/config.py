import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['running.lr'] = 0.001
p2['input_size'] = parents_config['common.history_seq_len']
p2['en.in_attn_hidden_size'] = 144
p2['de.lstm_hidden_size'] = 576 # 576
p2['de.temporal_attn_hidden_size'] = 144
p2['ar.layer1'] = 256
parameter = p2
