# --coding:utf-8--
import torch
from torch.backends import cudnn
a = torch.Tensor([1.])
print(a.cuda())
print(cudnn.is_acceptable(a.cuda()))