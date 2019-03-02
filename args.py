import numpy as np


class Args:
    n_eig = 0.005  # percent of eigen values to be zero
    lmd = 0.5
    dfw = 2
    sw = 1
    lmw = 1
    k = 100
    debug = False
    n_iter = 10
    c_dim = (148, 148)  # matrix dimension

    rbf_sigma = 0.01  # spread of rbf kernel
    lambda_m = 0.1  # regularization factor
    beta = np.ones(c_dim) / 3
    mu = 0.5
    r = 2
    intra_r = 0.7
    root_directory = "/home/turja/AD-Longitudinal-Smoothing"
    data_directory = "/home/turja/AD-Data_Organized"
    raw_data_directory = "/home/turja/AD-Data"
    data_file = "/home/turja/DXData.csv"


