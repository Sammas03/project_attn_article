import torch
## 指定数据存放在gpu上，默认类型是float64
torch.set_default_tensor_type(torch.cuda.DoubleTensor)
t = torch.tensor(1.)
print(t.device, t.dtype, sep='\n')
# cuda:0
# torch.float64
