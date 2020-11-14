##   
##  This file is adapted from https://github.com/Sleepwalking/pytorch-softdtw
##  

import numpy as np
import torch
from torch.autograd import Function
from utils import generate_step

def compute_forward(C, gamma):
    """
    Compute GMDTW for multiple time series.

    Parameters
    ----------
    C: array, shape [m_1, m_2, ..., m_M]
        Cost tensor
    gamma: float
        Regularization parameter

    Returns
    -------
    R: array, shape [m_1+2, m_2+2, ..., m_M+2]
        Alignment tensor
    """
    m = np.array(C.shape)
    M = len(m)
    R = np.ones(m+2) * np.inf
    R[tuple([0]*M)] = 0.0
    steps = generate_step(M)
    indices = np.array([0]*M)

    # Forward recursion to compute GMDTW
    def _forward(count):
        if count == M:
            r = np.zeros(len(steps))
            for j, s in enumerate(steps):
                r[j] = -R[tuple(indices-s)]/gamma
            rmax = r.max()
            rsum = np.exp(r-rmax).sum()
            softmin = -gamma * (np.log(rsum) + rmax)
            R[tuple(indices)] = C[tuple(indices-1)] + softmin
        else:
            for i in range(1, m[count]+1):
                indices[count] = i
                count += 1
                _forward(count)
                count -= 1
    
    _forward(0)
    
    return R

def compute_backward(C_, R, gamma):
    """
    Compute GMDTW grad 
    
    Parameters
    ----------
    C_: array, shape [m1, m2, m3]
        Cost tensor
    R: array, shape [m1+2, m2+2, m3+2]
        Alignment tensor
    gamma: float
        Regularization parameter

    Returns
    -------
    E: array, shape [m1+2, m2+2, m3+2]
        Gradient tensor
    """
    m = np.array(C_.shape)
    M = len(m)
    C = np.zeros(m+2)
    E = np.zeros(m+2)
    C[tuple([slice(1,m[i]+1) for i in range(M)])] = C_
    E[tuple([-1]*M)] = 1.0
    slices = [slice(0,m[i]+2) for i in range(M)]
    for i in range(M):
        slices[i] = -1
        R[tuple(slices)] = -np.inf
        slices[i] = slice(0, m[i]+2)
    R[tuple([-1]*M)] = R[tuple([-2]*M)]
    steps = generate_step(M)
    indices = np.array([0]*M)

    # Back propagation to compute gradient
    def _backward(count):
        if count == M:
            tmp = 0.0
            for s in steps:
                d = R[tuple(indices+s)] - R[tuple(indices)] - C[tuple(indices+s)]
                tmp += np.exp(d/gamma)
            E[tuple(indices)] = tmp
        else:
            for i in range(1, m[count]+1):
                indices[count] = i
                count += 1
                _backward(count)
                count -= 1

    _backward(0)
    
    return E