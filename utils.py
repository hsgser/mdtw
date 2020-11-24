import numpy as np
from numba import jit
from itertools import product
import torch

def generate_step(M):
    """
    Generate all possible steps.

    Parameters
    ----------
    M: int
        Number of time series
    
    Returns
    -------
    steps: array, shape [2^M-1]
        All possible steps
    """
    return list(product(range(2), repeat=M))[1:]

def cost_tensor_3(x1, x2, x3):
    """
    Compute cost tensor for 3 time series.

    Parameters
    ----------
    x1, x2, x3: array, shape [m_1, p], [m_2, p], [m_3, p]
        Time series

    Returns
    -------
    C: array, shape [m_1, m_2, m_3]
        Cost tensor between time series
    """
    m1 = x1.size(0)
    m2 = x2.size(0)
    m3 = x3.size(0)
    p = x1.size(1)
    x1 = x1.unsqueeze(1).unsqueeze(2).expand(m1, m2, m3, p)
    x2 = x2.unsqueeze(0).unsqueeze(2).expand(m1, m2, m3, p)
    x3 = x3.unsqueeze(0).unsqueeze(1).expand(m1, m2, m3, p)
    # Pair-wise squared euclidean distance
    c12 = torch.pow(x1-x2, 2).sum(3)
    c23 = torch.pow(x2-x3, 2).sum(3)
    c13 = torch.pow(x1-x3, 2).sum(3)
    C = c12 + c23 + c13
    return C

def cost_tensor(X):
    """
    Compute cost tensor for multiple time series.

    Parameters
    ----------
    X: list, shape M
        List of time series. ith time series is array with shape [m_i, p].

    Returns
    -------
    C: array, shape [m_1, m_2, ..., m_M]
        Cost tensor between time series
    """
    M = len(X)
    m = [len(X[i]) for i in range(M)]
    p = X[1].size(1)
    for i in range(M):
        for j in range(M):
            if j != i:
                X[i] = X[i].unsqueeze(j)
        X[i] = X[i].expand(m + [p])
    # Pair-wise squared euclidean distance
    C = torch.zeros(m)
    for i in range(M-1):
        for j in range(i, M):
            tmp = torch.pow(X[j]-X[i], 2).sum(-1)
            C.add_(tmp)
    return C

def jacobian_product_3(x1, x2, x3, B):
    """
    Compute the transpose of the Jacobian applied to a tensor.
    Jacobian is the derivative of the cost matrix w.r.t the first time series.
    The cost used here is the sum of pair-wise squared Euclidean distance.

    Parameters
    ----------
    x1, x2, x3: array, shape [m_1, p], [m_2, p], [m_3, p]
        Time series
    
    B: array, shape [m_1, m_2, m_3]
        Tensor

    Returns
    -------
    S: array, shape [m_1, p]
        Jacobian product.
    """
    B1 = torch.sum(B, dim=[1, 2]).unsqueeze(1).expand_as(x1)
    B12 = torch.sum(B, dim=2)
    B13 = torch.sum(B, dim=1)
    # S = 2 * x1.T * B1 - torch.matmul(x2.T, B12.T) - torch.matmul(x3.T, B13.T)
    S = 2 * x1 * B1 - torch.matmul(B12, x2) - torch.matmul(B13, x3)
    return 2*S