from rbf import *
from utils.create_brain_net_files import *
from utils.helper import *
from utils.L2_distance import *
from utils import EProjSimplex, get_hemisphere
from args import Args
from test_result import test_result


def optimize_longitudinal_connectomes(connectome_list, dfw, sw, lmw, diag_dist_factor=0, r=-1, islocal=True):
    c_dim = connectome_list[0].shape
    args = Args(c_dim)
    rbf_fit = RBF(args.rbf_sigma, args.lambda_m, args.debug)
    wt_local = [np.ones(args.c_dim) for i in range(0, len(connectome_list))]
    k = args.k
    smoothed_connectomes = []

    eps = 10e-10
    # Initialization
    rt = []

    num = c_dim[0]
    idx_list = []
    for cn in connectome_list:
        distX = (1 - cn)
        distX = distX + np.diag(np.ones(num) * eps)
        distX1 = np.sort(distX, axis=1)
        idx = np.argsort(distX, axis=1)
        idx_list.append(idx)
        A = np.zeros((num, num))
        if r < 0:
            rr = get_gamma(distX, k)
            r = np.mean(rr)
        rt.append(r)
        lmd = r
        for i in range(0, num):
            A[i, idx[i, 0: k]] = 2 * (distX1[i, k] - distX1[i, 0: k]) / (r + eps)

        smoothed_connectomes.append(row_normalize(A))

    smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)

    F = None
    # Iteration
    for i in range(0, args.n_iter):
        if args.debug:
            print("Iteration: ", i)

        M = find_mean(smoothed_connectomes, wt_local)  # link-wise mean of the connectomes
        M = 0.5 * np.add(M, M.T)
        M = np.clip(M, 0, 1)
        D = np.diag(M.sum(axis=1))
        L = np.subtract(D, M)
        dM = (1 - M)
        F_old = F
        #n_modes = get_n_modes(M, threshold=0.99)
        #print("Number of modes: ", n_modes)
        eig_val, F = get_eigen(L, args.n_module)

        dF = L2_distance(np.transpose(F), np.transpose(F))
        #dF = np.sqrt((dF >= 0) * dF)
        np.fill_diagonal(dF, 0)

        loss = 0
        for t in range(0, len(smoothed_connectomes)):
            idx = idx_list[t]
            dX = (1 - connectome_list[t])
            dS = (1 - np.clip(smoothed_connectomes[t], 0, 1))
            dI = (dfw * dX + sw * dS
                  + lmw * dM) / (dfw + sw + lmw) + lmd * dF

            S_new = np.zeros(args.c_dim)
            for j in range(0, args.c_dim[0]):
                if islocal:
                    idxa0 = idx[j, 0:k]
                else:
                    idxa0 = np.arange(num)

                vv, _ = EProjSimplex.EProjSimplex(-dI[j, idxa0] / (2*rt[t]))
                S_new[j, idxa0] = np.real(vv)
            np.fill_diagonal(S_new, 0)
            smoothed_connectomes[t] = row_normalize(S_new)
            wt_local[t] = np.exp(-((smoothed_connectomes[t] - row_normalize(M)) ** 2) / (args.rbf_sigma ** 2))
            loss = loss + ((S_new + dI) ** 2).sum()

        smoothed_connectomes = rbf_fit.fit_rbf_to_longitudinal_connectomes(smoothed_connectomes)

        if sum(eig_val[:args.n_module]) > eps:
            lmd = lmd * 2
        elif sum(eig_val[:args.n_module + 1]) < eps:
            lmd = lmd / 2
            F = F_old
        else:
            break

        if args.debug:
            print("lamda: ", lmd)

        print("Loss: ", loss)

    for t in range(0, len(smoothed_connectomes)):
        smoothed_connectomes[t] = row_normalize(smoothed_connectomes[t])

    return smoothed_connectomes, M, loss


if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '052_S_4944'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    args = Args()
    smoothed_connectomes, M, E = optimize_longitudinal_connectomes(connectome_list, args.dfw, args.sw, args.lmw)
    output_dir = os.path.join(data_dir, sub + '_smoothed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(0, len(smoothed_connectomes)):
        with open(os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)), 'w') as out:
            np.savetxt(out, smoothed_connectomes[t])

    #with open(os.path.join(output_dir, sub + "_smoothed_u"), 'w') as out:
     #   np.savetxt(out, M)

    n_comp_, label_list = get_number_of_components(connectome_list)
    print("\nNumber of component: ", n_comp_)
    n_comp_, label_list = get_number_of_components(smoothed_connectomes)
    print("\nNumber of component: ", n_comp_)
    #create_brain_net_node_files(sub, label_list)
    #create_brain_net_node_files(sub + "_smoothed", label_list)

    connectome_list.append(find_mean(connectome_list))
    smoothed_connectomes.append(M)

    #test_result(sub, connectome_list, smoothed_connectomes)
