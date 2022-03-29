import torch
from ray import tune

p1 = {
    # 编解码器中的公共参数
    "common.history_seq_len": 24,  # 可调整参数
    "common.prediction_horizon": 1,  # 预测的序列长度 可调整参数

    # mlp 参数
    "mlp.layer1.hidden_num": 512,
    "mlp.layer2.hidden_num": 256,
    "mlp.layer3.hidden_num": 64,
    "mlp.output_num": 1,

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
p2['mlp.layer1.hidden_num'] = tune.randint(512, 1024)
p2['mlp.layer2.hidden_num'] = tune.randint(256, 512)
p2['mlp.layer3.hidden_num'] = tune.randint(32, 128)

# 服务器进行参数搜索使用
p3 = p2.copy()
p3["running.max_epoch"] = 600

parameter = p3