import numpy as np


class Args:
    def __init__(self, c_dim=(148, 148)):
        self.debug = False
        self.n_iter = 20
        self.c_dim = c_dim  # matrix dimension
        self.n_module = 7  # expected number of modules in the network

        self.rbf_sigma = 1  # spread of rbf kernel
        self.lambda_m = 0.1  # regularization factor
        self.beta = np.ones(self.c_dim)/3
        self.mu = 0.5
        self.r = 2
        self.k = 60  # number of non-zero elements in each row of the connectome matrix
        self.root_directory = "/playpen/mturja/AD-Longitudinal-Smoothing"

        # loss function weights for different terms like: data fitting
        self.dfw = 300
        self.sw = 50
        self.lmw = 50

        self.lmd_cut = 500
