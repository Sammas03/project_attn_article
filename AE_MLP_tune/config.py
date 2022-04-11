import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['unit.layer1.hidden_num'] = tune.choice([32, 64, 128, 256])
p2['unit.layer2.hidden_num'] = tune.choice([16, 32, 64])
p2['unit.layer3.hidden_num'] = tune.choice([8, 16, 32])
p2['aemlp.encode_size'] = tune.randint(12, 64)
p2['de.layer1'] = tune.choice([512, 1024])
p2['de.layer2'] = tune.choice([128, 256])
p2['de.layer3'] = tune.choice([32, 64])
parameter = p2
