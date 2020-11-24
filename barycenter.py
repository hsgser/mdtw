import numpy as np
from utils import *
from gmdtw import *
import torch

def gmdtw_3_barycenter(X, gamma=1.0, weights=None, lr=1e-3, tol=1e-3, max_iter=200):
    """
    Compute barycenter for multiple time series.

    Parameters
    ----------
    X: array, shape [M, n, p]
        List of time series. 
        Each time series has shape [n, p]
    gamma: float, default = 1.0
        Regularization parameter.
    weights: None or array, default = None
        Weights of each X[i]. Must be the same size as len(X).
    lr: float, default = 1e-3
        Learning rate.
    tol: float, default 1e-3
        Tolerance of the method used.
    max_iter: int, default = 200
        Maximum number of iterations.

    Returns
    -------
    y: array, shape [n, p]
        Barycenter of multiple time series.
    """
    M, n, p = X.size()
    dtype = X[0].dtype
    y = torch.ones(n, p, dtype=dtype, requires_grad=True)
    if weights is None:
        weights = np.ones((M, M))
    weights = np.array(weights)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = y.to(device)
        lr = torch.Tensor([lr]).to(device).type(dtype)
        prev_loss = np.inf
        X = X.to(device)
       
        # Gradient Descent
        for iter in range(max_iter):
            curr_loss = 0
            grad = torch.zeros_like(y, dtype=dtype).to(device)

            for i in range(M-1):
                for j in range(i+1, M):
                    C = cost_tensor_3(y, X[i], X[j])
                    C_ = C.detach().cpu().numpy()
                    R = torch.Tensor(compute_forward_3(C_, gamma)).to(device).type(dtype)
                    curr_loss += weights[i, j] * R[-2, -2, -2].item()
                    R_ = R.detach().cpu().numpy()
                    E = torch.Tensor(compute_backward_3(C_, R_, gamma)).to(device).type(dtype)
                    grad += weights[i, j] * jacobian_product_3(y, X[i], X[j], E)
            
            if iter % 10 == 0: # Print loss every 10 iterations
                print(f"Iter {iter}: Loss = {curr_loss}")

            if (curr_loss > prev_loss - tol):
                break
            prev_loss = curr_loss
            y = y - lr*grad
    
    return y

