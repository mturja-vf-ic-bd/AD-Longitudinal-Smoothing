from rbf import *
import pickle
import numpy as np
from utils.helper import find_mean, get_eigen, get_gamma
from utils.L2_distance import *
from utils import EProjSimplex
import time
from pprint import pprint


if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_4926'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    T = len(connectome_list)
    sigma = 1
    lambda_m = 0.005
    debug = False
    rbf = RBF(sigma, lambda_m, debug)
    cdim = connectome_list[0].shape
    W = [np.ones(cdim) for i in range(0, len(connectome_list))]
    n_iter = 100
    c = 12  # number of components
    Beta = np.ones(cdim)
    mu = 1
    k = 50
    start_time = time.time()
    smoothed_connectomes = rbf.fit_rbf_to_longitudinal_connectomes(connectome_list)
    print("Execution time: ", time.time() - start_time)

    for i in range(0, n_iter):
        print("Iteration: ", i)
        M = find_mean(smoothed_connectomes, W)  # link-wise mean of the connectomes
        W = [np.exp(-np.exp(np.subtract(smoothed_connectome, M))/(sigma ** 2))
             for smoothed_connectome in smoothed_connectomes]  # get weight W for each connectome

        D = np.diag(M.sum(axis=1))
        L = np.subtract(D, M)
        eig_val, F = get_eigen(L, c)

        dF = L2_distance(np.transpose(F), np.transpose(F))

        for t in range(0, len(smoothed_connectomes)):
            dX = (1 - connectome_list[t]) ** 2
            dI = (1 - connectome_list[t]) ** 2 + (1 - smoothed_connectomes[t]) ** 2 +\
                 np.multiply(Beta, np.add((1 - M) ** 2, mu * dF))

            gamma = get_gamma(dI, k)
            S_new = np.zeros(cdim)
            for j in range(0, cdim[0]):
                vv, _ = EProjSimplex.EProjSimplex(-dI[j]/gamma[j])
                S_new[j] = vv

            smoothed_connectomes[t] = np.asarray(S_new)

        smoothed_connectomes = rbf.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)
        for s in smoothed_connectomes:
            print((s > 0).sum())



