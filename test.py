from rbf import *
import pickle
from utils.helper import find_mean

if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_4926'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    T = len(connectome_list)
    sigma = 1
    lambda_m = 0.1
    debug = False
    rbf = RBF(sigma, lambda_m, debug)
    W = np.ones(T) / T
    print(W)
    n_iter = 2

    for i in range(0, n_iter):
        # l_smoothed_connectome_list = rbf.fit_rbf_to_longitudinal_connectomes(connectome_list)
        smoothed_connectomes = pickle.load(open(sub + '_cont_mat_list.pkl', 'rb'))
        M = find_mean(smoothed_connectomes, W)  # link-wise mean of the connectomes
        W = [np.exp(-np.exp(smoothed_connectome - M)/sigma ** 2)
             for smoothed_connectome in smoothed_connectomes]  # get weight W for each connectome

