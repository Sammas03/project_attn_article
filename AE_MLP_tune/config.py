import torch
from ray import tune

p1 = {
    # 编解码器中的公共参数
    "common.history_seq_len": 24,  # 可调整参数
    "common.prediction_horizon": 1,  # 预测的序列长度 可调整参数

    # mlp 参数
    "unit.layer1.hidden_num": 512,
    "unit.layer2.hidden_num": 256,
    "unit.layer3.hidden_num": 64,
    "unit.output_num": 1,

    # 运行时的一些参数
    "running.lr": 1e-4,
    "running.batch_size": 8,
    "running.num_epoch": 1,
    "running.lrs_step_size": 2000,  # 学习率下降步长
    "running.max_grad_norm": 0.1,  # 梯度剪切 最大梯度
    "running.gradient_accumulation_steps": 1,  # 梯度累计计算步长
    "running.reg1": False,  # L1 正则化
    "running.reg2": True,  # L2 正则化
    "running.reg_factor1": 1e-4,
    "running.reg_factor2": 1e-4,
    "running.data_succession": True  # 样本数据是否连续

}

# ray.tune功能测试，只跑一个epoch
p2 = p1.copy()
p2['unit.layer1.hidden_num'] = tune.choice([32, 64, 128, 256])
p2['unit.layer2.hidden_num'] = tune.choice([16, 32, 64])
p2['unit.layer3.hidden_num'] = tune.choice([8, 16, 32])
p2['aemlp.encode_size'] = tune.randint(12, 64)
p2['de.layer1'] = tune.choice([512, 1024])
p2['de.layer2'] = tune.choice([128, 256])
p2['de.layer3'] = tune.choice([32, 64])

# 服务器进行参数搜索使用
p3 = p2.copy()
p3["running.max_epoch"] = 600

parameter = p3
