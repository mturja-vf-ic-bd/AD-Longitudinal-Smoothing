from rbf import *
import pickle
import numpy as np
from utils.helper import find_mean, get_eigen



if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_5110'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    T = len(connectome_list)
    sigma = 1
    lambda_m = 0.1
    debug = False
    rbf = RBF(sigma, lambda_m, debug)
    W = [np.ones(connectome_list[0].shape) for i in range(0, len(connectome_list))]
    print(W)
    n_iter = 3
    c = 6  # number of components

    smoothed_connectomes = pickle.load(open(sub + '_cont_mat_list.pkl', 'rb'))
    #smoothed_connectomes = rbf.fit_rbf_to_longitudinal_connectomes(connectome_list)
    with open(sub + '_cont_mat_list.pkl', 'wb') as f:
        pickle.dump(smoothed_connectomes, f)
    for i in range(0, n_iter):
        M = find_mean(smoothed_connectomes, W)  # link-wise mean of the connectomes
        W = [np.exp(-np.exp(smoothed_connectome - M)/(sigma ** 2))
             for smoothed_connectome in smoothed_connectomes]  # get weight W for each connectome

        D = np.diag(M.sum(axis=1))
        L = D - M
        eig_val, eig_vec = get_eigen(L, c)
        print(eig_val)




