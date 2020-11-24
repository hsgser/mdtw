# Import libraries
import datetime
import numpy as np
import torch
from mdtw import *
from gmdtw import *
from dataset import load_ucr
from utils import *

# Parameters
N = 10 
name = "ECG200"
M = 3
rng = np.random.RandomState(0)

# Load data
X_tr, y_tr, X_te, y_te = load_ucr(name)
print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
classes = np.unique(y_tr)

# Experiments
if torch.cuda.is_available():
    device = torch.device("cuda")
    # MDTW
    start = datetime.datetime.now()
    for _ in range(N):
        k = rng.randint(len(classes))
        X = X_tr[y_tr == classes[k]]
        x1, x2, x3 = X[rng.permutation(len(X))[:M]]
        x1 = torch.from_numpy(x1).to(device)
        x2 = torch.from_numpy(x2).to(device)
        x3 = torch.from_numpy(x3).to(device)
        C = cost_tensor_3(x1, x2, x3)
        C = C.detach().cpu().numpy()
        R = mdtw_3(C)
    end = datetime.datetime.now()
    print(f"MDTW: {(end-start).total_seconds()/N} s")

    # GMDTW
    start = datetime.datetime.now()
    for _ in range(N):
        k = rng.randint(len(classes))
        X = X_tr[y_tr == classes[k]]
        x1, x2, x3 = X[rng.permutation(len(X))[:M]]
        x1 = torch.from_numpy(x1).to(device)
        x2 = torch.from_numpy(x2).to(device)
        x3 = torch.from_numpy(x3).to(device)
        C = cost_tensor_3(x1, x2, x3)
        C = C.detach().cpu().numpy()
        R = compute_forward_3(C, 1.0)
    end = datetime.datetime.now()
    print(f"GMDTW: {(end-start).total_seconds()/N} s")