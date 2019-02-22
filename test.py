from rbf import *
from utils.create_brain_net_files import *
from utils.helper import *
from utils.L2_distance import *
from utils import EProjSimplex, get_hemisphere
from args import Args
from statistics import median

def get_split_stat(M, k):
    num = len(M)
    sum_split = M[:, 0:num // 2].sum(axis=1) / (M.sum(axis=1) + 1e-10)
    K1 = int(round(sum_split * k))
    return K1


def initialize_connectomes(connectome_list):
    c_dim = connectome_list[0].shape
    k = Args.k
    smoothed_connectomes = []

    eps = 10e-10
    # Initialization
    rt = []

    num = c_dim[0]
    idx_list = []
    k1 = int(round(Args.intra_r * k))
    n_mode_list = []

    for cn in connectome_list:
        distX = (1 - cn)
        np.fill_diagonal(distX, 1 + eps)  # To make the diagonal elements largest in their row
        idx, idx11, idx12, idx21, idx22 = sort_idx(distX, k, Args.intra_r)
        idx_list.append(idx)

        # Find the number of modes that we will collapse
        cn_s = (cn + cn.T) / 2
        D_cn = np.diag(cn_s.sum(axis=1))
        L_cn = np.subtract(D_cn, cn_s)
        _, _, nm = get_eigen(L_cn, Args.n_eig)
        n_mode_list.append(nm)
        A = np.zeros((num, num))

        rr11, rr12, rr21, rr22 = get_gamma_splitted(distX, k, Args.intra_r)
        r1 = np.mean(rr11)
        r2 = np.mean(rr12)
        r3 = np.mean(rr21)
        r4 = np.mean(rr22)
        rt.append((r1, r2, r3, r4))
        #r = np.mean(get_gamma(distX, k))
        #rt.append((r, r, r, r))
        for i in range(0, num):
            if i < num / 2:
                A[i, idx11[i, 0:k1]] = 2 * (distX[i, idx11[i, k1]] - distX[i, idx11[i, 0:k1]])
                A[i, idx12[i, 0:k - k1]] = 2 * (distX[i, idx12[i, k - k1]] - distX[i, idx12[i, 0:k - k1]])
            else:
                A[i, idx21[i - num // 2, 0:k - k1]] = 2 * (
                        distX[i, idx21[i - num // 2, k - k1]] - distX[i, idx21[i - num // 2, 0:k - k1]])
                A[i, idx22[i - num // 2, 0:k1]] = 2 * (
                        distX[i, idx22[i - num // 2, k1]] - distX[i, idx22[i - num // 2, 0:k1]])

        smoothed_connectomes.append(row_normalize(A))

    return smoothed_connectomes, rt, n_mode_list, idx_list


def optimize_longitudinal_connectomes(connectome_list, dfw, sw, lmw, lmd, r=-1):
    c_dim = connectome_list[0].shape
    rbf_fit = RBF(Args.rbf_sigma, Args.lambda_m, Args.debug)
    wt_local = [np.ones(Args.c_dim) for i in range(0, len(connectome_list))]
    k = Args.k
    eps = 1e-10
    num = c_dim[0]

    smoothed_connectomes, rt, n_mode_list, idx_list = initialize_connectomes(connectome_list)

    n_modes = int(median(n_mode_list))
    print(n_modes)
    smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)

    F = None
    # Iteration
    loss = 0
    for i in range(0, Args.n_iter):
        if Args.debug:
            print("Iteration: ", i)

        M = find_mean(smoothed_connectomes, wt_local)  # link-wise mean of the connectomes
        M_s = 0.5 * np.add(M, M.T)
        D = np.diag(M_s.sum(axis=1))
        L = np.subtract(D, M_s)
        dM = (1 - M)
        eig_val, F, _ = get_eigen(L, n_modes)
        dF = L2_distance(np.transpose(F), np.transpose(F))

        sum_split = M[:, 0:num // 2].sum(axis=1) / (M.sum(axis=1) + eps)

        for t in range(0, len(smoothed_connectomes)):
            idx = idx_list[t]
            r1, r2, r3, r4 = rt[t]
            print(rt[t])
            dX = (1 - connectome_list[t])
            dS = (1 - smoothed_connectomes[t])
            dI = (dfw * dX + sw * dS
                  + lmw * dM + lmd * dF) / (dfw + sw + lmw + lmd)

            S_new = np.zeros(Args.c_dim)
            for j in range(0, num):
                print("j, sum: ", j, sum_split[j])
                # S_new[j, idx[j, 0:k]], _ = EProjSimplex.EProjSimplex(-dI[j, idx[j, 0:k]] / (2*rt[t]))

                if j < num / 2:
                    k1 = int(round(sum_split[j] * k))
                    dI11 = -dI[j, idx[j, 0:k1]] / (2 * r1)
                    dI12 = -dI[j, idx[j, k1 + 1:k]] / (2 * r2)
                    S_new[j, idx[j, 0:k1]], _ = EProjSimplex.EProjSimplex(dI11,
                                                                          sum_split[j])
                    S_new[j, idx[j, k1 + 1:k]], _ = EProjSimplex.EProjSimplex(dI12,
                                                                              1 - sum_split[j])
                else:
                    k1 = int(round((1-sum_split[j]) * k))
                    dI21 = -dI[j, idx[j, 0:k - k1]] / (2 * r3)
                    dI22 = -dI[j, idx[j, k - k1 + 1:k]] / (2 * r4)
                    S_new[j, idx[j, 0:k - k1]], _ = EProjSimplex.EProjSimplex(dI21,
                                                                              sum_split[j])
                    S_new[j, idx[j, k - k1 + 1:k]], _ = EProjSimplex.EProjSimplex(dI22,
                                                                                  1 - sum_split[j])

            smoothed_connectomes[t] = S_new
            wt_local[t] = np.exp(-((smoothed_connectomes[t] - row_normalize(M)) ** 2) / (Args.rbf_sigma ** 2))
            loss = (S_new + dI).sum()

        smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)

        if Args.debug:
            print("lamda: ", lmd)

        print("Loss: ", loss)

    for t in range(0, len(smoothed_connectomes)):
        smoothed_connectomes[t] = row_normalize(smoothed_connectomes[t])

    # loss=1
    # M = None
    return smoothed_connectomes, M, loss


if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_2336'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    Args = Args()
    smoothed_connectomes, M, E = optimize_longitudinal_connectomes(connectome_list, Args.dfw, Args.sw, Args.lmw,
                                                                   Args.lmd)
    output_dir = os.path.join(data_dir, sub + '_smoothed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(0, len(smoothed_connectomes)):
        with open(os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)), 'w') as out:
            np.savetxt(out, smoothed_connectomes[t])

    # with open(os.path.join(output_dir, sub + "_smoothed_u"), 'w') as out:
    #   np.savetxt(out, M)

    n_comp_, label_list = get_number_of_components(connectome_list)
    print("\nNumber of component: ", n_comp_)
    n_comp_, label_list = get_number_of_components(smoothed_connectomes)
    print("\nNumber of component: ", n_comp_)
    # create_brain_net_node_files(sub, label_list)
    # create_brain_net_node_files(sub + "_smoothed", label_list)

    connectome_list.append(find_mean(connectome_list))
    smoothed_connectomes.append(M)

    # test_result(sub, connectome_list, smoothed_connectomes)
