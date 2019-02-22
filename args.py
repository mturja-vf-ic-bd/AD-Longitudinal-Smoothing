import numpy as np


class Args:
    n_eig = 0.005  # Number of eigen values to be zero
    lmd = 0
    dfw = 1
    sw = 0
    lmw = 0
    k = 90
    debug = False
    n_iter = 10
    c_dim = (148, 148)  # matrix dimension
    n_module = 8  # expected number of modules in the network

    rbf_sigma = 0.01  # spread of rbf kernel
    lambda_m = 0.1  # regularization factor
    beta = np.ones(c_dim) / 3
    mu = 0.5
    r = 2
    intra_r = 0.8
    root_directory = "/home/turja/AD-Longitudinal-Smoothing"


