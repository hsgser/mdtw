"""
CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
"Developing a pattern discovery method in time series data and its GPU acceleration"
"""

import numpy as np
import torch
import torch.cuda
from numba import cuda
import math
from utils import *
import logging
logging.getLogger("numba").setLevel(logging.WARNING) # disable numba.cuda log info


@cuda.jit
def compute_forward_3_cuda(C, R, gamma, m):
    """
    Assume all time series has the same length.
    """
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma
    n_passes = 3*m-2

    # Loop through each spheere I + J + K = p
    for p in range(n_passes):
        remain = p - tid
        if 0 <= remain <= 2*m - 2:
            for J in range(min(m, remain+1)):
                K = remain - J
                if 0 <= K < m:
                    i = I + 1
                    j = J + 1
                    k = K + 1
                    r0 = -R[i, j, k-1] * inv_gamma
                    r1 = -R[i, j-1, k] * inv_gamma
                    r2 = -R[i, j-1, k-1] * inv_gamma
                    r3 = -R[i-1, j, k] * inv_gamma
                    r4 = -R[i-1, j, k-1] * inv_gamma
                    r5 = -R[i-1, j-1, k] * inv_gamma
                    r6 = -R[i-1, j-1, k-1] * inv_gamma
                    rmax = max(r0, r1, r2, r3, r4, r5, r6)
                    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax) + \
                        math.exp(r3 - rmax) + math.exp(r4 - rmax) + math.exp(r5 - rmax) + math.exp(r6 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[i,j,k] = C[i-1,j-1,k-1] + softmin
    
        cuda.syncthreads()

@cuda.jit
def compute_forward_2_cuda(C, R, gamma, m):
    """
    Assume all time series has the same length.
    """
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma
    n_passes = 2*m-1

    # Loop through each spheere I + J = p
    for p in range(n_passes):
        J = p - tid
        if 0 <= J < m:
            i = I + 1
            j = J + 1
            r0 = -R[i, j-1] * inv_gamma
            r1 = -R[i-1, j] * inv_gamma
            r2 = -R[i-1, j-1] * inv_gamma
            rmax = max(r0, r1, r2)
            rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax) 
            softmin = -gamma * (math.log(rsum) + rmax)
            R[i,j] = C[i-1,j-1] + softmin
    
        cuda.syncthreads()

@cuda.jit
def compute_backward_3_cuda(C, R, E, gamma, m):
    """
    Assume all time series has the same length.
    """
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma
    n_passes = 3*m-2

    # Loop through each spheere I + J + K = p
    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1
        remain = rev_p - tid
        if 0 <= remain <= 2*m - 2:
            for J in range(min(m-1, remain), -1, -1):
                K = remain - J
                if 0 <= K < m:
                    i = I + 1
                    j = J + 1
                    k = K + 1
                    d0 = math.exp((R[i, j, k+1] - R[i, j, k] - C[i, j, k+1]) * inv_gamma) * E[i, j, k+1]
                    d1 = math.exp((R[i, j+1, k] - R[i, j, k] - C[i, j+1, k]) * inv_gamma) * E[i, j+1, k]
                    d2 = math.exp((R[i, j+1, k+1] - R[i, j, k] - C[i, j+1, k+1]) * inv_gamma) * E[i, j+1, k+1]
                    d3 = math.exp((R[i+1, j, k] - R[i, j, k] - C[i+1, j, k]) * inv_gamma) * E[i+1, j, k]
                    d4 = math.exp((R[i+1, j, k+1] - R[i, j, k] - C[i+1, j, k+1]) * inv_gamma) * E[i+1, j, k+1]
                    d5 = math.exp((R[i+1, j+1, k] - R[i, j, k] - C[i+1, j+1, k]) * inv_gamma) * E[i+1, j+1, k]
                    d6 = math.exp((R[i+1, j+1, k+1] - R[i, j, k] - C[i+1, j+1, k+1]) * inv_gamma) * E[i+1, j+1, k+1]
                    E[i,j,k] = d0 + d1 + d2 + d3 + d4 + d5 + d6
    
        cuda.syncthreads()

@cuda.jit
def compute_backward_2_cuda(C, R, E, gamma, m):
    """
    Assume all time series has the same length.
    """
    tid = cuda.threadIdx.x
    I = tid
    inv_gamma = 1.0 / gamma
    n_passes = 2*m-1

    # Loop through each spheere I + J = p
    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = rev_p - tid
        if 0 <= J < m:
            i = I + 1
            j = J + 1
            d0 = math.exp((R[i, j+1] - R[i, j] - C[i, j+1]) * inv_gamma) * E[i, j+1]
            d1 = math.exp((R[i+1, j] - R[i, j] - C[i+1, j]) * inv_gamma) * E[i+1, j]
            d2 = math.exp((R[i+1, j+1] - R[i, j] - C[i+1, j+1]) * inv_gamma) * E[i+1, j+1]
            E[i, j] = d0 + d1 + d2
    
        cuda.syncthreads()

class GMDTW3_CUDA_Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3, gamma):
        device = x1.device
        dtype = x1.dtype
        gamma = torch.Tensor([gamma]).to(device).type(dtype)
        C = cost_tensor_3(x1, x2, x3)
        m = C.size(0)
        R = torch.ones((m+2, m+2, m+2), device=device, dtype=dtype) * math.inf
        R[0, 0, 0] = 0
        compute_forward_3_cuda[1, m](cuda.as_cuda_array(C.detach()), 
                                    cuda.as_cuda_array(R), 
                                    gamma.item(),
                                    m)
        ctx.save_for_backward(C, R, x1, x2, x3, gamma)

        return R[-2, -2, -2]
    
    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        dtype = grad_output.dtype
        C_, R, x1, x2, x3, gamma = ctx.saved_tensors
        m = C_.size(0)
        C = torch.zeros((m+2, m+2, m+2), device=device, dtype=dtype)
        E = torch.zeros((m+2, m+2, m+2), device=device, dtype=dtype)
        E[-1,-1,-1] = 1.0
        C[1:m+1, 1:m+1, 1:m+1] = C_
        R[:,:,-1] = -math.inf
        R[:,-1,:] = -math.inf
        R[-1,:,:] = -math.inf
        R[-1,-1,-1] = R[-2,-2,-2]
        
        compute_backward_3_cuda[1, m](cuda.as_cuda_array(C.detach()), 
                                    cuda.as_cuda_array(R), 
                                    cuda.as_cuda_array(E),
                                    gamma.item(),
                                    m)
        G = jacobian_product_3(x1, x2, x3, E[1:m+1, 1:m+1, 1:m+1])

        return G, None, None, None

class GMDTW2_CUDA_Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, gamma):
        device = x1.device
        dtype = x1.dtype
        gamma = torch.Tensor([gamma]).to(device).type(dtype)
        C = cost_tensor_2(x1, x2)
        m = C.size(0)
        R = torch.ones((m+2, m+2), device=device, dtype=dtype) * math.inf
        R[0, 0] = 0
        compute_forward_2_cuda[1, m](cuda.as_cuda_array(C.detach()), 
                                    cuda.as_cuda_array(R), 
                                    gamma.item(),
                                    m)
        ctx.save_for_backward(C, R, x1, x2, gamma)
        
        return R[-2, -2]
    
    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        dtype = grad_output.dtype
        C_, R, x1, x2, gamma = ctx.saved_tensors
        m = C_.size(0)
        C = torch.zeros((m+2, m+2), device=device, dtype=dtype)
        E = torch.zeros((m+2, m+2), device=device, dtype=dtype)
        E[-1,-1] = 1.0
        C[1:m+1, 1:m+1] = C_
        R[:,-1] = -math.inf
        R[-1,:] = -math.inf
        R[-1,-1] = R[-2,-2]
        compute_backward_2_cuda[1, m](cuda.as_cuda_array(C.detach()), 
                                    cuda.as_cuda_array(R), 
                                    cuda.as_cuda_array(E),
                                    gamma.item(),
                                    m)
        G = jacobian_product_2(x1, x2, E[1:m+1, 1:m+1])

        return G, None, None

class GMDTW_CUDA(torch.nn.Module):
    def __init__(self, gamma=1.0, version=2):
        super(GMDTW_CUDA, self).__init__()
        self.gamma = gamma
        self.version = version
    
    def forward(self, y, X):
        if self.version == 2:
            return GMDTW2_CUDA_Loss.apply(y, *X, self.gamma)
        elif self.version == 3:
            return GMDTW3_CUDA_Loss.apply(y, *X, self.gamma)
        else:
            raise ValueError("Support version either 2 or 3!")