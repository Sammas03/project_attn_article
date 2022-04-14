import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['input_size'] = 4
p2['en.hidden_num'] = tune.randint(16, 64)
p2['en.num_layers'] = tune.choice([1, 2, 3, 4, 5])
p2['de.hidden_num'] = tune.randint(64, 128)
parameter = p2
