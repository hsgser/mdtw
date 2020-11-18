import numpy as np
from utils import *
from gmdtw import *
import torch

# Time series
x1 = torch.arange(6, dtype=torch.float).view(-1, 3)

# Initialization
p = x1.size(1)
dtype = x1.dtype
length = x1.size(0)
max_iter = 100
tol = 1e-3
gamma = 1.0
lr = 1e-1
y = torch.ones(length, p, dtype=dtype, requires_grad=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = y.to(device)
    x1 = x1.to(device)
    prev_loss = np.inf

    for i in range(max_iter):
        C = cost_tensor_3(y, x1, x1)
        C_ = C.detach().cpu().numpy()
        R = torch.Tensor(compute_forward_3(C_, gamma)).to(device).type(dtype)
        curr_loss = R[-2,-2,-2].item()
        if (curr_loss > prev_loss - tol):
            break
        prev_loss = curr_loss
        R_ = R.detach().cpu().numpy()
        E = torch.Tensor(compute_backward_3(C_, R_, gamma)).to(device).type(dtype)
        grad = jacobian_product_3(y, x1, x1, E)
        y = y - lr*grad
    
    print(prev_loss)
    print(y)


