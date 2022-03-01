import torch
from ray import tune

p1 = {

    "gru.hidden_num":64,
    "gru.output_num":1,
    "gru.num_layers":2,

    # 运行时的一些参数
    "running.lr": 1e-4,
    "running.batch_size": 2,
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

#ray.tune功能测试，只跑一个epoch
p2 = p1.copy()
p2['input_encoder.attn_hidden'] = tune.randint(64,256)
p2['input_encoder.hidden_size'] = tune.randint(64,256)
p2['group.hidden_size'] = tune.randint(32,128)
p2['group.output_size'] = tune.randint(64,256)
p2['temporal_decoder.hidden_size'] = tune.randint(64,256)




#服务器进行参数搜索使用
p3 = p2.copy()
p3["running.num_epoch"] = 600

parameter = p1

