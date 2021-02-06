from itertools import combinations
import numpy as np
from utils import *
from gmdtw import *
from gmdtw_cuda import *
import torch
from torch import optim
from torch.autograd import Variable
from scipy.optimize import minimize
from mdtw import *
from mdtw_cuda import *
torch.manual_seed(0)

def gmdtw_barycenter(X, version, log, gamma=1.0, lr=1e-3, tol=1e-3, 
                    random_init=False, use_cuda=True):
    """
        Compute barycenter for multiple time series.

        Parameters
        ----------
        X: array, shape [M, n, p]
            List of time series. 
            Each time series has shape [n, p].
        version: {2, 3}
            If 2 use SDTW, else use GMDTW.
        log: Logger object
            Store loss info
        gamma: float, default = 1.0
            Regularization parameter.
        lr: float, default = 1e-3
            Learning rate.
        tol: float, default 1e-3
            Tolerance of the method used.
        random_init: bool, default = False
            If true, initialize barycenter randomly. 
            Otherwise, use Euclidean mean initialization.
        use_cuda: bool, default = True
            If true, use CUDA version.
            Otherwise, use CPU version.

        Returns
        -------
        y: array, shape [n, p]
            Barycenter of multiple time series.
    """
    # Initialization
    M = X.size(0)
    dtype = X[0].dtype
    tuple_idx = list(combinations(range(M), version-1))
    
    if version == 2:
        max_iter = 500
    elif version == 3:
        max_iter = 100 
    else:
        raise ValueError("Support version either 2 or 3!")

    if use_cuda:
        gmdtw = GMDTW_CUDA(gamma=gamma, version=version)
    else:
        gmdtw = GMDTW(gamma=gamma, version=version)
    
    prev_loss = np.inf
    log_iter = max_iter // 10
    
    # Initialize barycenter
    if random_init:
        y = torch.rand(X[0].size(), dtype=dtype)
    else:
        y = X.mean(axis=0)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = y.to(device)
        X = X.to(device)
        y = Variable(y, requires_grad=True)
        optimizer = optim.SGD([y], lr=lr)
       
        for iter in range(max_iter):
            optimizer.zero_grad()
            curr_loss = torch.Tensor([0]).to(device).type(dtype)

            for idx in tuple_idx:
                curr_loss += gmdtw(y, X[list(idx)])                     

            if iter % log_iter == 0: # Log loss
                log.info(f"Iter {iter}: Loss = {curr_loss.item()}")
            
            if (curr_loss > prev_loss - tol):
                break
            prev_loss = curr_loss.item()
            curr_loss.backward()
            optimizer.step()

            if iter % log_iter == 0: # Log dtw loss
                log.info(f"DTW loss: {dtw_loss_cuda(X, y)}")
            
        log.info(f"Final loss: {prev_loss}")

    _dtw_loss = dtw_loss_cuda(X, y)
    log.info(f"DTW loss: {_dtw_loss}") 

    return y, prev_loss, _dtw_loss.item()

def gmdtw_barycenter_scipy(X, version, log, gamma=1.0, method="L-BFGS-B", 
                            tol=1e-3, max_iter=100, random_init=False, use_cuda=True):
    """
        Compute barycenter for multiple time series.
        Optimize using scipy instead of pytorch.

        Parameters
        ----------
        X: array, shape [M, n, p]
            List of time series. 
            Each time series has shape [n, p].
        version: {2, 3}
            If 2 use SDTW, else use GMDTW.
        log: Logger object
            Store loss info
        gamma: float, default = 1.0
            Regularization parameter.
        method: string, default = L-BFGS-B
            Optimization method. 
        tol: float, default 1e-3
            Tolerance of the method used.
        max_iter: int, default = 100
            Maximum number of iterations.
        random_init: bool, default = False
            If true, initialize barycenter randomly. 
            Otherwise, use Euclidean mean initialization.
        use_cuda: bool, default = True
            If true, use CUDA version.
            Otherwise, use CPU version.

        Returns
        -------
        y: array, shape [n, p]
            Barycenter of multiple time series.
    """
    # Initialization
    M = X.size(0)
    device = torch.device("cuda")
    X = X.to(device)
    dtype = X[0].dtype
    log_iter = max_iter // 10
    tuple_idx = list(combinations(range(M), version-1))
    
    # Initialize barycenter
    if random_init:
        barycenter_init = torch.rand(X[0].size(), dtype=dtype).numpy()
    else:
        barycenter_init = X.mean(axis=0).cpu().numpy()
    
    if use_cuda:
        gmdtw = GMDTW_CUDA(gamma=gamma, version=version)
    else:
        gmdtw = GMDTW(gamma=gamma, version=version)

    def _func(y):
        # Compute objective value and grad at y.
        y = torch.from_numpy(y.reshape(*barycenter_init.shape)).to(device)
        y = Variable(y, requires_grad=True)
        curr_loss = torch.Tensor([0]).to(device).type(dtype)
            
        for idx in tuple_idx:
            curr_loss += gmdtw(y, X[list(idx)]) 
        
        curr_loss.backward()

        return curr_loss.item(), y.grad.detach().cpu().numpy().ravel()
    
    def generate_print_callback():
        saved_params = { "iteration_number" : 0 }

        def print_callback(y):
            iter = saved_params["iteration_number"]
            if iter % log_iter == 0:
                log.info(f"Iter {iter}: DTW Loss = {dtw_loss(X.detach().cpu(), y.reshape(*barycenter_init.shape))}")
            saved_params["iteration_number"] += 1

        return print_callback

    # The function works with vectors so we need to vectorize barycenter_init.
    res = minimize(_func, barycenter_init.ravel(), method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False), callback=generate_print_callback())

    log.info(f"Final loss: {res.fun}")
    y = res.x.reshape(*barycenter_init.shape)
    _dtw_loss = dtw_loss(X.detach().cpu(), y)
    log.info(f"DTW loss: {_dtw_loss}")

    return y, res.fun, _dtw_loss