# Import libraries
from barycenter import *
from utils import *
import datetime
import numpy as np
import torch
from dataset import load_ucr
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import euclidean_distances
plt.style.use('ggplot')
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

# Parameters
name = "ECG200"
M = 10
rng = np.random.RandomState(0)

# Load data
X_tr, y_tr, X_te, y_te = load_ucr(name)
print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
classes = np.unique(y_tr)

# Experiments
if torch.cuda.is_available():
    device = torch.device("cuda")
    k = rng.randint(len(classes))
    X = X_tr[y_tr == classes[k]]
    X = torch.from_numpy(X[rng.permutation(len(X))[:M]])

    fig = plt.figure(figsize=(15,4))
    fig_pos = 131

    for gamma in (0.1, 1, 10):
        ax = fig.add_subplot(fig_pos)

        for x in X:
            ax.plot(x.view(-1), c="k", linewidth=3, alpha=0.15)
        
        start = datetime.datetime.now()
        y = gmdtw_barycenter_3(X, gamma=gamma)
        y = y.detach().cpu().numpy()
        end = datetime.datetime.now()
        print(f"gamma = {gamma}: {(end-start).total_seconds()} s")
        ax.plot(y.ravel(), c="r", linewidth=3, alpha=0.7)
        ax.set_title(f"GMDTW $\gamma$={gamma}")

        fig_pos += 1
    
    # plt.savefig(f"{name}_barycenter_2_lbfgsb.png")
    # plt.savefig(f"{name}_barycenter_2_sgd.png")
    plt.savefig(f"{name}_barycenter_3_sgd.png")
    plt.close("all")