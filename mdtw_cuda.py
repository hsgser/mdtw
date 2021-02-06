import math
from utils import *
from utils import cost_tensor_2
from numba import cuda
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

@cuda.jit
def mdtw_3_cuda(C, R, m):
    tid = cuda.threadIdx.x
    I = tid
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
                    softmin = min(R[i, j, k-1], 
                                R[i, j-1, k],
                                R[i, j-1, k-1],
                                R[i-1, j, k],
                                R[i-1, j, k-1],
                                R[i-1, j-1, k],
                                R[i-1, j-1, k-1])
                    R[i,j,k] = C[i-1,j-1,k-1] + softmin
    
        cuda.syncthreads()

@cuda.jit
def mdtw_2_cuda(C, R, m):
    tid = cuda.threadIdx.x
    I = tid
    n_passes = 2*m-1

    # Loop through each spheere I + J = p
    for p in range(n_passes):
        J = p - tid
        if 0 <= J < m:
            i = I + 1
            j = J + 1
            softmin = min(R[i, j-1], 
                        R[i-1, j],
                        R[i-1, j-1])
            R[i,j] = C[i-1,j-1] + softmin
    
        cuda.syncthreads()

def dtw_loss_cuda(X, y):
    device = X.device
    dtype = X.dtype
    length = X.size(0)
    weights = np.ones(length)/length
    loss = 0

    for i in range(length):
        C = cost_tensor_2(y, X[i])
        m = C.size(0)
        R = torch.ones((m+2, m+2), device=device, dtype=dtype) * math.inf
        R[0, 0] = 0
        mdtw_2_cuda[1, m](cuda.as_cuda_array(C.detach()), 
                        cuda.as_cuda_array(R), 
                        m)
        loss += weights[i] * R[-2, -2]
    
    return loss