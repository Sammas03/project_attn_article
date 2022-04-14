import torch
## 指定数据存放在gpu上，默认类型是float64
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
t = torch.tensor(1.)
print(t.device, t.dtype, sep='\n')
# cuda:0
# torch.float64

from util.nn_util import init_rnn_hidden

h_n = init_rnn_hidden(batch=3,hidden_size=64,num_layers=1,num_dir=1)
print(h_n)