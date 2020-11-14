##   
##  This file is adapted from https://github.com/samcohen16/Aligning-Time-Series
##    

import numpy as np
from numba import jit

@jit(nopython=True)
def mdtw(C, steps):
    """
    Compute MDTW for 3 time series.

    Parameters
    ----------
    C : array, shape [m1, m2, m3]
        Cost tensor
    steps : array, shape [2^M-1]
        All possible steps

    Returns
    -------
    R: array, shape [m1+1, m2+1, m3+1]
        Alignment tensor
    """
    m1, m2, m3 = C.shape
    R = np.ones((m1+1, m2+1, m3+1)) * np.inf
    R[0,0,0] = 0

    # Forward recursion to compute MDTW
    for i1 in range(1, m1+1):
        for i2 in range(1, m2+1):
            for i3 in range(1, m3+1):
                softmin = np.inf
                for s in steps:
                    softmin = min(softmin, R[i1-s[0],i2-s[1],i3-s[2]])
                R[i1,i2,i3] = C[i1-1,i2-1,i3-1] + softmin
    
    return R
