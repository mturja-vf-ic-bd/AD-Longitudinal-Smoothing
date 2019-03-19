from rbf import *
from utils.create_brain_net_files import *
from utils.L2_distance import *
from utils import EProjSimplex, get_hemisphere
from args import Args
import os

from utils.helper import row_normalize, get_gamma_splitted, get_eigen, sort_idx2, get_subject_names, get_scan_count, \
    find_mean, add_noise_all, connectome_median, get_avg_zeros_per_row
from utils.readFile import readMatricesFromDirectory


def get_split_stat(M, k):
    num = len(M)
    sum_split = M[:, 0:num // 2].sum(axis=1) * k / (M.sum(axis=1) + 1e-10)
    K1 = np.clip(np.around(sum_split).astype(int), 2, num//2 - 2) + 1
    K2 = np.clip(k - K1, 2, num//2 - 3) + 2
    return K1, K2


def process_connectome(cn):
    """
    Finds a sparse representation of cn and get some important parameters for optimization
    :param cn: The input matrix for the connectome
    :return:
    sparse_cn: sparse matrix
    rt = regularization parameters required for optimizaiton
    nm = number of modes to optimize
    idx = sorted index of distX=(1-cn)
    K1 = number of non-zero elements in the left half of cn
    K2 = number of non-zero elements in the right half of cn
    """

    c_dim = cn.shape
    k = Args.k

    eps = 1e-10
    num = c_dim[0]

    distX = (1 - cn)
    np.fill_diagonal(distX, 1 + eps)  # To make the diagonal elements largest in their row
    idx = sort_idx2(distX)

    # Find the number of modes that we will collapse
    cn_s = (cn + cn.T) / 2
    D_cn = np.diag(cn_s.sum(axis=1))
    L_cn = np.subtract(D_cn, cn_s)
    _, _, nm = get_eigen(L_cn, Args.n_eig)

    # Find how many inter and intra hemispheric connection to keep
    K1, K2 = get_split_stat(cn, k)
    r1, r2, r3, r4 = get_gamma_splitted(distX, K1, K2, mean=True)
    rt = (r1, r2, r3, r4)

    # Get a sparse representation of cn
    A = np.zeros((num, num))
    for i in range(0, num):
        k1 = K1[i]
        k2 = K2[i]
        A[i, idx[i, 0:k1]] = 2 * (distX[i, idx[i, k1]] - distX[i, idx[i, 0:k1]])
        A[i, idx[i, num // 2:num // 2 + k2]] = 2 * (distX[i, idx[i, num // 2 + k2]] -
                                                    distX[i, idx[i, num // 2: num // 2 + k2]])

    sparse_cn = row_normalize(A)

    return sparse_cn, rt, nm, idx, K1, K2


def initialize_connectomes(connectome_list):
    c_dim = connectome_list[0].shape
    k = Args.k
    smoothed_connectomes = []

    eps = 1e-10
    # Initialization
    rt = []

    num = c_dim[0]
    idx_list = []
    n_mode_list = []

    for cn in connectome_list:
        distX = (1 - cn)
        np.fill_diagonal(distX, 1 + eps)  # To make the diagonal elements largest in their row
        idx = sort_idx2(distX)
        idx_list.append(idx)

        # Find the number of modes that we will collapse
        cn_s = (cn + cn.T) / 2
        D_cn = np.diag(cn_s.sum(axis=1))
        L_cn = np.subtract(D_cn, cn_s)
        _, _, nm = get_eigen(L_cn, Args.n_eig)
        n_mode_list.append(nm)
        A = np.zeros((num, num))

        # Find how many inter and intra hemispheric connection to keep
        K1, K2 = get_split_stat(cn, k)

        rr11, rr12, rr21, rr22 = get_gamma_splitted(distX, K1, K2)
        r1 = np.mean(rr11)
        r2 = np.mean(rr12)
        r3 = np.mean(rr21)
        r4 = np.mean(rr22)
        rt.append((r1, r2, r3, r4))
        # r = np.mean(get_gamma(distX, k))
        # rt.append((r, r, r, r))
        for i in range(0, num):
            k1 = K1[i]
            k2 = K2[i]
            A[i, idx[i, 0:k1]] = 2 * (distX[i, idx[i, k1]] - distX[i, idx[i, 0:k1]])
            A[i, idx[i, num//2:num//2 + k2]] = 2 * (distX[i, idx[i, num//2 + k2]] -
                                                        distX[i, idx[i, num//2: num//2 + k2]])

        smoothed_connectomes.append(row_normalize(A))

    return smoothed_connectomes, rt, n_mode_list, idx_list, K1, K2


def optimize_longitudinal_connectomes(connectome_list, dfw, sw, lmw,
                                      lmd, pro=Args.pro, rbf_sigma=Args.rbf_sigma, lambda_m=Args.lambda_m):
    c_dim = connectome_list[0].shape
    rbf_fit = RBF(rbf_sigma, lambda_m, Args.debug)
    wt_local = [np.ones(Args.c_dim) for i in range(0, len(connectome_list))]
    eps = 1e-10
    num = c_dim[0]

    smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(connectome_list)
    prev_smoothed_connectomes = [None] * len(smoothed_connectomes)

    M = connectome_median(connectome_list)
    loss = 0
    F = None
    # Iteration
    for i in range(0, Args.n_iter):
        if Args.debug:
            print("Iteration: ", i)

        M_sparse, rt, n_modes, idx, K1, K2 = process_connectome(M)
        M_s = 0.5 * np.add(M_sparse, M_sparse.T)
        D = np.diag(M_s.sum(axis=1))
        L = np.subtract(D, M_s)
        dM = abs(1 - M_sparse)
        F_old = F
        eig_val, F, _ = get_eigen(L, n_modes)
        #F /= F.sum(axis=0)
        dF = L2_distance(np.transpose(F), np.transpose(F))
        dF = (dF > 0) * dF

        sum_split = M_sparse[:, 0:num // 2].sum(axis=1) / (M_sparse.sum(axis=1) + eps)
        prev_loss = loss
        loss = 0

        r1, r2, r3, r4 = rt
        for t in range(0, len(smoothed_connectomes)):
            dX = abs(1 - connectome_list[t])
            dS = abs(1 - smoothed_connectomes[t])
            dI = (dfw * dX + sw * dS
                  + lmw * (dM + lmd * dF) / (1 + lmd)) / (dfw + sw + lmw)

            S_new = np.zeros(Args.c_dim)
            for j in range(0, num):
                k1 = K1[j]
                k2 = K2[j]

                if j < num / 2:
                    dI11 = -dI[j, idx[j, 0:k1]] / (2 * r1)
                    dI12 = -dI[j, idx[j, num//2:num//2 + k2]] / (2 * r2)
                    S_new[j, idx[j, 0:k1]], _, l1 = EProjSimplex.EProjSimplex(dI11,
                                                                          sum_split[j])
                    S_new[j, idx[j, num//2:num//2 + k2]], _, l2 = EProjSimplex.EProjSimplex(dI12,
                                                                              1 - sum_split[j])
                    loss = loss + l1 + l2
                else:
                    dI21 = -dI[j, idx[j, 0:k1]] / (2 * r3)
                    dI22 = -dI[j, idx[j, num//2: num//2 + k2]] / (2 * r4)
                    S_new[j, idx[j, 0:k1]], _, l3 = EProjSimplex.EProjSimplex(dI21, sum_split[j])
                    S_new[j, idx[j, num//2: num//2 + k2]], _, l4 = EProjSimplex.EProjSimplex(dI22,
                                                                                  1 - sum_split[j])
                    loss = loss + l3 + l4

            prev_smoothed_connectomes[t] = smoothed_connectomes[t]
            smoothed_connectomes[t] = S_new
            wt_local[t] = np.exp(-((smoothed_connectomes[t] - row_normalize(M_sparse)) ** 2) / (pro ** 2))

        M = find_mean(smoothed_connectomes, wt_local)  # link-wise mean of the connectomes
        smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)

        if Args.debug:
            print("lamda: ", lmd)
        print("Loss: ", loss)


        if prev_loss < loss and prev_loss != 0:
            loss = prev_loss
            smoothed_connectomes = prev_smoothed_connectomes
            '''if sum(eig_val[:n_modes]) > 1e-2:
                lmd = lmd * 2
                print("Increasing lmd")
                continue
            elif eig_val[n_modes] < 1e-5:
                print("Decreasing lmd")
                lmd = lmd / 2
                F = F_old
                continue
            else:
                break'''
            break

    #smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)
    for t in range(0, len(smoothed_connectomes)):
        smoothed_connectomes[t] = row_normalize(smoothed_connectomes[t])

    return smoothed_connectomes, M_sparse, loss


if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    #sub_names = get_subject_names(3)
    sub_names = ["027_S_5110"]
    nr = []
    ns = []
    for sub in sub_names:
        scan_count = get_scan_count(sub)
        if scan_count > 1:
            print("---------------\n\nRunning ", sub, " with scan count : ", scan_count)
            connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
            smoothed_connectomes, M, E = optimize_longitudinal_connectomes(connectome_list, Args.dfw, Args.sw, Args.lmw,
                                                                           Args.lmd)

            '''
            connectome_list_noisy = add_noise_all(connectome_list)
            smoothed_connectomes_noisy, M, E = optimize_longitudinal_connectomes(connectome_list_noisy, Args.dfw, Args.sw, Args.lmw,
                                                                   Args.lmd)
            # Compute noise in raw
            noise_rw = 0
            noise_sm = 0
            for t in range(0, len(connectome_list)):
                noise_rw = noise_rw + abs(connectome_list[t] - connectome_list_noisy[t]).sum()
                noise_sm = noise_sm + abs(smoothed_connectomes[t] - smoothed_connectomes_noisy[t]).sum()

            print("Raw: ", noise_rw,
                  "\nSM: ", noise_sm)

            l = len(smoothed_connectomes)
            nr.append(noise_rw/l)
            ns.append(noise_sm/l)
            '''

            output_dir = os.path.join(data_dir, sub + '_smoothed')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for t in range(0, len(smoothed_connectomes)):
                with open(os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)), 'w') as out:
                    np.savetxt(out, smoothed_connectomes[t])

            #n_comp_, label_list = get_number_of_components(connectome_list)
            #print("\nNumber of component: ", n_comp_)
            #n_comp_, label_list = get_number_of_components(smoothed_connectomes)
            #print("\nNumber of component: ", n_comp_)

    #print("Raw: ", np.mean(nr), "+-", np.std(nr),
    #      "Smooth: ", np.mean(ns), "+-", np.std(ns))
