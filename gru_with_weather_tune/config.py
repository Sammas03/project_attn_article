import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['input_size']=4
p2['gru.hidden_num'] = tune.randint(32, 256)
p2['gru.num_layers'] = tune.choice([1, 2, 3, 4, 5])
parameter = p2
