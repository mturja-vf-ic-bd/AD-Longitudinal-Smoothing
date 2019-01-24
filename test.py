from rbf import *
import pickle
import numpy as np
from utils.helper import find_mean, get_eigen, get_gamma
from utils.L2_distance import *
from utils import EProjSimplex
import time
from args import Args


def optimize_longitudinal_connectomes(connectome_list):
    args = Args()
    rbf_fit = RBF(args.rbf_sigma, args.lambda_m, args.debug)
    wt_local = [np.ones(args.c_dim) / len(connectome_list)for i in range(0, len(connectome_list))]
    beta = np.ones(args.c_dim) / 2
    smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(connectome_list)

    for i in range(0, args.n_iter):
        print("Iteration: ", i)
        M = find_mean(smoothed_connectomes, wt_local)  # link-wise mean of the connectomes
        wt_local = [np.exp(-np.exp(np.subtract(smoothed_connectome, M)) / (args.rbf_sigma ** 2))
                    for smoothed_connectome in smoothed_connectomes]  # get weight W for each connectome

        D = np.diag(M.sum(axis=1))
        L = np.subtract(D, M)
        eig_val, F = get_eigen(L, args.n_module)

        dF = L2_distance(np.transpose(F), np.transpose(F))

        for t in range(0, len(smoothed_connectomes)):
            dX = (1 - connectome_list[t]) ** 2
            dI = dX + (1 - smoothed_connectomes[t]) ** 2 + \
                 np.multiply(beta, np.add((1 - M) ** 2, args.mu * dF))

            gamma = get_gamma(dI, args.k)
            S_new = np.zeros(args.c_dim)
            for j in range(0, args.c_dim[0]):
                vv, _ = EProjSimplex.EProjSimplex(-dI[j] / gamma[j])
                S_new[j] = vv

            smoothed_connectomes[t] = np.asarray(S_new)

        smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)
        n_comp_, _ = get_number_of_components(smoothed_connectomes)
        print("\nNumber of component: ", n_comp_)

    return smoothed_connectomes


if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_4926'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    optimize_longitudinal_connectomes(connectome_list)
