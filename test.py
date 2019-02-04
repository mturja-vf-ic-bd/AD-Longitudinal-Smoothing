from typing import List

from numpy.core.multiarray import ndarray

from rbf import *
from utils.create_brain_net_files import *
from utils.helper import *
from utils.L2_distance import *
from utils import EProjSimplex, get_hemisphere
from bct.utils import visualization
from args import Args


def optimize_longitudinal_connectomes(connectome_list):
    c_dim = connectome_list[0].shape
    args = Args(c_dim)
    rbf_fit = RBF(args.rbf_sigma, args.lambda_m, args.debug)
    wt_local = [np.ones(args.c_dim) for i in range(0, len(connectome_list))]
    k = args.k
    DX = []
    diag_dist_factor = 1
    smoothed_connectomes = []
    for t in range(0, len(connectome_list)):
        A = connectome_list[t]
        distX = 1 - A
        #np.fill_diagonal(distX, diag_dist_factor)
        distX = distX + diag_dist_factor * np.diag(np.diag(A.sum(axis=1)))
        DX.append(distX)

    # Initialization

    for distX in DX:
        num = distX.shape[0]
        distX1 = np.sort(distX, axis=1)
        idx = np.argsort(distX, axis=1)
        A = np.zeros((num, num))
        rr = get_gamma(distX, k)
        r = np.mean(rr)
        lmd = r
        eps = 10e-10
        for i in range(0, num):
            A[i, idx[i, 1: k + 1]] = -distX1[i, 1: k + 1]/(2 * rr[i] + eps) + 1/k + 1/(2*k*rr[i]) * distX1[i, 1:k+1].sum()
            #A[i, idx[i, 1: k + 2]] = 2 * (distX1[i, k + 1] - distX1[i, 1: k + 2]) / (rr[i] + eps)

        np.fill_diagonal(A, 0)
        A = (A.T + A)/2
        A = row_normalize(A)
        smoothed_connectomes.append(A)
    #smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(connectome_list)
    F = None
    # Iteration
    for i in range(0, args.n_iter):
        print("Iteration: ", i)
        M = find_mean(smoothed_connectomes, wt_local)  # link-wise mean of the connectomes
        M = 0.5 * np.add(M, M.T)
        M = row_normalize(M)
        D = np.diag(M.sum(axis=1))
        L = np.subtract(D, M)
        dM = (1 - M) ** 2
        F_old = F
        eig_val, F = get_eigen(L, args.n_module)

        dF = L2_distance(np.transpose(F), np.transpose(F))

        for t in range(0, len(smoothed_connectomes)):
            dX = (1 - connectome_list[t]) ** 2
            dS = (1 - smoothed_connectomes[t]) ** 2

            dX = dX + diag_dist_factor * np.diag(np.diag(A.sum(axis=1)))
            dS = dS + diag_dist_factor * np.diag(np.diag(A.sum(axis=1)))
            dM = dM + diag_dist_factor * np.diag(np.diag(A.sum(axis=1)))

            dI = 1000 * dX + 10 * dS \
                 + 10 * np.multiply(wt_local[t], dM) + np.multiply(lmd, dF)

            gamma = get_gamma(dI, args.k)
            r = np.mean(gamma)
            S_new = np.zeros(args.c_dim)
            for j in range(0, args.c_dim[0]):
                vv, _ = EProjSimplex.EProjSimplex(-dI[j] / r)
                S_new[j] = vv

            np.fill_diagonal(S_new, 0)
            S_new = (S_new.T + S_new)/2
            S_new = row_normalize(S_new)
            smoothed_connectomes[t] = np.array(S_new)

        smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)

        for t in range(0, len(smoothed_connectomes)):
            smoothed_connectomes[t] = row_normalize(smoothed_connectomes[t])
            wt_local[t] = np.exp(-(smoothed_connectomes[t] - M) ** 2)

        if sum(eig_val[:args.n_module]) > eps and lmd < 300:
            lmd = lmd * 2
        elif sum(eig_val[:args.n_module + 1]) < eps and lmd > 0.01:
            lmd = lmd / 2
            F = F_old
        else:
            break

        print("lamda: ", lmd)

    return smoothed_connectomes


if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_2336'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))

    smoothed_connectomes = optimize_longitudinal_connectomes(connectome_list)
    output_dir = os.path.join(data_dir, sub + '_smoothed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(0, len(smoothed_connectomes)):
        with open(os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)), 'w') as out:
            np.savetxt(out, smoothed_connectomes[t])

    n_comp_, label_list = get_number_of_components(connectome_list)
    print("\nNumber of component: ", n_comp_)
    n_comp_, label_list = get_number_of_components(smoothed_connectomes)
    print("\nNumber of component: ", n_comp_)
    #create_brain_net_node_files(sub, label_list)
    #create_brain_net_node_files(sub + "_smoothed", label_list)

