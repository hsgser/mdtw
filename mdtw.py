##   
##  This file is adapted from https://github.com/samcohen16/Aligning-Time-Series
##    

import numpy as np
from utils import generate_step

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
