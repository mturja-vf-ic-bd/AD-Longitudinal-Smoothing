import numpy as np


class Args:
    n_eig = 0.005  # percent of eigen values to be zero
    eps = 1e-12
    th = 0.01  # Threshold of connectome
    dfw = 1
    sw = 1
    lmw = 1
    lmd = 1
    k = 50
    threshold = 0.0002 # threshold for raw connectomes
    debug = False
    n_iter = 10
    c_dim = (148, 148)  # matrix dimension

    rbf_sigma = 0.01  # spread of rbf kernel
    lambda_m = 1
    beta = np.ones(c_dim) / 3
    mu = 0.5
    r = 2
    intra_r = 0.7
    pro = 1
    root_directory = "/home/turja/AD-Longitudinal-Smoothing"
    data_directory = "/home/turja/AD-Data_Organized"
    raw_data_directory = "/home/turja/AD-Data"
    data_file = "/home/turja/DXData.csv"


