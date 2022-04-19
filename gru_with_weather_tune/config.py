import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['input_size']=4
p2['gru.hidden_num'] = 128
p2['gru.num_layers'] = 3
parameter = p2
