import torch
from torch import nn
from torch.autograd import Variable
from common_config.common_config import parents_config

def init_rnn_hidden(batch, hidden_size: int, num_layers: int = 1,num_dir=1, xavier: bool = True):
    """
    :param batch:
    :param hidden_size: 隐层大小
    :param num_layers: 网络深度，<1则为cell型网络
    :param xavier: 是否进行normal初始化
    :return: tensor
         output
        输出维度(seq_len, batch, num_directions * hidden_size)，即（句子中字的数量，批量大小，LSTM方向数量∗ * ∗隐藏向量维度）
        h_n
        维度(num_layers * num_directions, batch, hidden_size)
        c_n
        维度(num_layers * num_directions, batch, hidden_size)
    """
    if(parents_config['gpu']):
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    tensor = torch.zeros(batch, hidden_size) if num_layers < 1 else torch.zeros(num_dir*num_layers, batch, hidden_size)
    return nn.init.xavier_normal_(tensor) if xavier else Variable(tensor)
