import torch
from ray import tune

from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['mlp.layer1.hidden_num'] = tune.randint(512, 1024)
p2['mlp.layer2.hidden_num'] = tune.randint(256, 512)
p2['mlp.layer3.hidden_num'] = tune.randint(64, 128)
p2['mlp.layer4.hidden_num'] = tune.randint(32, 64)
parameter = p2
