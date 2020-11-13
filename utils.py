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

def cost_tensor(x1, x2, x3):
    """
    Compute cost tensor between 3 time series.

    Parameters
    ----------
    x1, x2, x3: array, shape [m1, p], [m2, p], [m3, p]
        Time series

    Returns
    -------
    C: array, shape [m1, m2, m3]
        Cost tensor between time series
    """
    m1 = x1.size(0)
    m2 = x2.size(0)
    m3 = x3.size(0)
    p = x1.size(1)
    x1 = x1.unsqueeze(1).unsqueeze(2).expand(m1, m2, m3, p)
    x2 = x2.unsqueeze(0).unsqueeze(2).expand(m1, m2, m3, p)
    x3 = x3.unsqueeze(0).unsqueeze(1).expand(m1, m2, m3, p)
    # Pair-wise euclidean distance
    c12 = torch.pow(x1-x2, 2).sum(3).sqrt()
    c23 = torch.pow(x2-x3, 2).sum(3).sqrt()
    c13 = torch.pow(x1-x3, 2).sum(3).sqrt()
    C = c12 + c23 + c13
    return C