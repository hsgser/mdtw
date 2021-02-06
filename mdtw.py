##   
##  This file is adapted from https://github.com/samcohen16/Aligning-Time-Series
##    

import numpy as np
from utils import generate_step
from numba import jit
from sklearn.metrics.pairwise import euclidean_distances

def mdtw(C):
    """
    Compute MDTW for multiple time series.

    Parameters
    ----------
    C: array, shape [m_1, m_2, ..., m_M]
        Cost tensor

    Returns
    -------
    R: array, shape [m_1+1, m_2+1, ..., m_M+1]
        Alignment tensor
    """
    m = np.array(C.shape)
    M = len(m)
    R = np.ones(m+1) * np.inf
    R[tuple([0]*M)] = 0.0
    steps = generate_step(M)
    indices = np.array([0]*M)

    # Forward recursion to compute MDTW
    def _mdtw(count):
        if count == M:
            softmin = np.inf
            for s in steps:
                softmin = min(softmin, R[tuple(indices-s)])
            R[tuple(indices)] = C[tuple(indices-1)] + softmin
        else:
            for i in range(1, m[count]+1):
                indices[count] = i
                count += 1
                _mdtw(count)
                count -= 1

    _mdtw(0)
    
    return R

@jit(nopython=True)
def mdtw_3(C):
    """
    Compute MDTW for 3 time series. Fast implementation using Numba.

    Parameters
    ----------
    C: array, shape [m_1, m_2, m_3]
        Cost tensor

    Returns
    -------
    R: array, shape [m_1+1, m_2+1, m_3+1]
        Alignment tensor
    """
    m1, m2, m3 = C.shape
    R = np.ones((m1+1, m2+1, m3+1)) * np.inf
    R[0,0,0] = 0
    steps = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    # Forward recursion to compute MDTW
    for i1 in range(1, m1+1):
        for i2 in range(1, m2+1):
            for i3 in range(1, m3+1):
                softmin = np.inf
                for s in steps:
                    softmin = min(softmin, R[i1-s[0],i2-s[1],i3-s[2]])
                R[i1,i2,i3] = C[i1-1,i2-1,i3-1] + softmin
    
    return R

@jit(nopython=True)
def mdtw_2(C):
    """
    Compute MDTW for 2 time series. Fast implementation using Numba.

    Parameters
    ----------
    C: array, shape [m_1, m_2]
        Cost tensor

    Returns
    -------
    R: array, shape [m_1+1, m_2+1]
        Alignment tensor
    """
    m1, m2 = C.shape
    R = np.ones((m1+1, m2+1)) * np.inf
    R[0,0] = 0
    steps = [(0, 1), (1, 0), (1, 1)]

    # Forward recursion to compute MDTW
    for i1 in range(1, m1+1):
        for i2 in range(1, m2+1):
            softmin = np.inf
            for s in steps:
                softmin = min(softmin, R[i1-s[0],i2-s[1]])
            R[i1,i2] = C[i1-1,i2-1] + softmin
    
    return R

def dtw_loss(X, y):
    length = len(X)
    weights = np.ones(length)/length
    X = np.array(X)
    y = np.array(y)
    loss = 0

    for i in range(length):
        C = euclidean_distances(y, X[i], squared=True)
        loss += weights[i] * mdtw_2(C)[-1, -1]

    return loss