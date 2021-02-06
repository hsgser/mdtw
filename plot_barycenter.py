# Import libraries
import os
import logging
from barycenter import *
from utils import *
import datetime
import numpy as np
import torch
from dataset import load_ucr
import matplotlib.pylab as plt
plt.style.use('ggplot')
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

# Reproduction
np_gen = np.random.RandomState(0)
torch.manual_seed(0)

# Parameters
repeats = 10
version = 3
N = 10
gammas = np.array([0.1, 1, 10])
dataset = "ECG200"
optimizer = "sgd" # "sgd" or "lbfgsb"
init = "random" # "random" or "euclidean"
use_cuda = True
if (optimizer == "sgd") and (init == "random") and (version == 3):
    lr = {0.1: 2e-4, 1: 2e-4, 10: 1e-3}
else:
    lr = {0.1: 1e-3, 1: 1e-3, 10: 1e-3}
if init == "euclidean":
    random_init = False
else:
    random_init = True
"""
Please create the following structure
results
     ____ dtw_loss
    |____ figures
    |____ log
    |____ sdtw_loss
"""
filename = f"{dataset}_barycenter_{version}_{optimizer}_{init}_init"
fig_file = f"results/figures/{filename}.png"
sdtw_loss_file = f"results/sdtw_loss/{filename}.csv"
dtw_loss_file = f"results/dtw_loss/{filename}.csv"
sdtw_loss = {gamma:np.nan for gamma in gammas}
dtw_loss = {gamma:np.nan for gamma in gammas}

# Logger
logging.basicConfig(filename=f"results/log/{filename}.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("barycenter")

# Load data
X_tr, y_tr, X_te, y_te = load_ucr(dataset)
logger.info(f"Data shape: X = {X_tr.shape}, y = {y_tr.shape}")

# Run experiments
logger.info("Start!")
for idx in range(repeats):
    logger.info(f"Run {idx+1}")
    classes = np.unique(y_tr)
    k = np_gen.randint(len(classes))
    logger.info(f"Class {k}")
    X = X_tr[y_tr == classes[k]]
    X = torch.from_numpy(X[np_gen.permutation(len(X))[:N]])
    logger.info(f"X shape: {X.size()}")

    fig = plt.figure(figsize=(5,4))
    fig_pos = 111

    for gamma in gammas:
    # for gamma in (0.1,): # For random initialization, it is necessary to run each gamma separately 10 times
        ax = fig.add_subplot(fig_pos)

        for x in X:
            ax.plot(x.view(-1), c="k", linewidth=3, alpha=0.15)
            
        start = datetime.datetime.now()
        if optimizer == "sgd":
            y, _sdtw_loss, _dtw_loss = gmdtw_barycenter(X,
                                                        version, 
                                                        lr=lr[gamma], 
                                                        log=logger, 
                                                        gamma=gamma, 
                                                        random_init=random_init, 
                                                        use_cuda=use_cuda)
            y = y.detach().cpu()
        else:
            y, _sdtw_loss, _dtw_loss = gmdtw_barycenter_scipy(X, 
                                                        version, 
                                                        log=logger, 
                                                        gamma=gamma, 
                                                        random_init=random_init,
                                                        use_cuda=use_cuda)
        sdtw_loss[gamma] = _sdtw_loss
        dtw_loss[gamma] = _dtw_loss
        y = np.array(y)
        end = datetime.datetime.now()
        logger.info(f"Run time gamma = {gamma}: {(end-start).total_seconds()} s")
        ax.plot(y.ravel(), c="r", linewidth=3, alpha=0.7)
        ax.set_title(f"GMDTW $\gamma$={gamma}")

        fig_pos += 1
        fig_file = f"results/figures/{filename}_gamma={gamma}.png"
    plt.savefig(fig_file)
    plt.close("all")

    # Save loss
    write_loss_to_file(sdtw_loss_file, sdtw_loss, gammas)
    write_loss_to_file(dtw_loss_file, dtw_loss, gammas)
logger.info("Finish!")