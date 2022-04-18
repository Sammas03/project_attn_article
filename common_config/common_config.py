import torch

parents_config = {
    # gpu设备
    'gpu': 1 if torch.cuda.is_available() else 0,
    'test': False,
    'output_size': 1,

    # 公共参数
    "common.block_num": 3,  # 长周期使用的天数
    "common.block_len": 24,  # 长时序使用的序列长度
    "common.history_seq_len": 72,  # 可调整参数
    "common.prediction_horizon": 1,  # 预测的序列长度 可调整参数

    # 运行时的一些参数
    "running.lr": 0.0025,
    "running.batch_size": 4,
    "running.num_epoch": 1,
    "running.max_grad_norm": 0.1,  # 梯度剪切 最大梯度
    "running.gradient_accumulation_steps": 1,  # 梯度累计计算步长
    "running.reg1": False,  # L1 正则化
    "running.reg2": True,  # L2 正则化
    "running.reg_factor1": 1e-4,
    "running.reg_factor2": 1e-4,
    "running.data_succession": True,  # 样本数据是否连续
    "running.max_epoch": 500,
    "running.min_epoch": 75,

}
