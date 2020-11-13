##   
##  The code for soft-DTW is adapted from https://github.com/Sleepwalking/pytorch-softdtw
##  

import numpy as np
from numba import jit
import torch
from torch.autograd import Function
from utils import generate_step

@jit(nopython=True)
def compute_forward(C, gamma, steps):
    """
    Compute GMDTW for 3 time series.

    Parameters
    ----------
    C : array, shape [m1, m2, m3]
        Cost tensor
    gamma: float
        Regularization parameter
    steps : array, shape [2^M-1]
        All possible steps

    Returns
    -------
    R: array, shape [m1+2, m2+2, m3+2]
        Alignment tensor
    """
    B, m1, m2, m3 = C.shape
    R = np.ones((B, m1+2, m2+2, m3+2)) * np.inf
    R[:,0,0,0] = 0.0

    # Forward recursion to compute GMDTW
    for b in range(B):
        for i1 in range(1, m1+1):
            for i2 in range(1, m2+1):
                for i3 in range(1, m3+1):
                    r = np.zeros(len(steps))
                    for j, s in enumerate(steps):
                        r[j] = -R[b,i1-s[0],i2-s[1],i3-s[2]]/gamma
                    rmax = r.max()
                    rsum = np.exp(r - rmax).sum()
                    softmin = -gamma * (np.log(rsum) + rmax)
                    R[b,i1,i2,i3] = C[b,i1-1,i2-1,i3-1] + softmin
    
    return R

@jit(nopython=True)
def compute_backward(C_, R, gamma, steps):
    """
    Compute GMDTW grad 
    
    Parameters
    ----------
    C : array, shape [m1, m2, m3]
        Cost tensor
    R: array, shape [m1+2, m2+2, m3+2]
        Alignment tensor
    gamma: float
        Regularization parameter
    steps : array, shape [2^M-1]
        All possible steps

    Returns
    -------
    E: array, shape [m1+2, m2+2, m3+2]
        Gradient tensor
    """
    B, m1, m2, m3 = C_.shape
    C = np.zeros((B, m1+2, m2+2, m3+2))
    E = np.zeros((B, m1+2, m2+2, m3+2))
    C[:,1:m1+1,1:m2+1,1:m3+1] = C_
    E[:,-1,-1,-1] = 1.0
    R[:,:,:,-1] = -np.inf
    R[:,:,-1,:] = -np.inf
    R[:,-1,:,:] = -np.inf
    R[:,-1,-1,-1] = R[:,-2,-2,-2]

    # Back propagation to compute gradient
    for b in range(B):
        for i1 in range(m1,0,-1):
            for i2 in range(m2,0,-1):
                for i3 in range(m3,0,-1):
                    tmp = 0.0
                    for s in steps:
                        d = R[b,i1+s[0],i2+s[1],i3+s[2]] - R[b,i1,i2,i3] - C[b,i1+s[0],i2+s[1],i3+s[2]]
                        tmp += np.exp(d/gamma)
                    E[b,i1,i2,i3] = tmp
    
    return E

class _GMDTW(Function):
    M = 3
    steps = generate_step(M)

    @staticmethod
    def forward(ctx, C, gamma):
        dev = C.device
        dtype = C.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
        C_ = C.detach().cpu().numpy()
        g_ = gamma.item()
        R = torch.Tensor(compute_forward(C_, g_, steps)).to(dev).type(dtype)
        ctx.save_for_backward(C, R, gamma)
        return R[:, -2, -2, -2]
    
    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        C, R, gamma = ctx.saved_tensors
        C_ = C.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.Tensor(compute_backward(C_, R_, g_, steps)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1, 1).expand_as(E) * E, None