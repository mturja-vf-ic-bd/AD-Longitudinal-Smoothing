from utils.helper import get_gamma, get_eigen
from utils.L2_distance import *
from utils.readFile import *
from utils.create_brain_net_files import *
from args import Args
from compare_network import get_number_of_components


def EProjSimplex(v, k = 1):
    """
        Problem:

        min 1/2 * || x - v || ^ 2
        s.t. x >= 0 1'x = 1
    """

    ft = 1
    n = len(v)
    v = np.asarray(v)
    v0 = v - np.mean(v) + k/n
    v1 = v0
    vmin = min(v0)

    if vmin < 0:
        sum_pos = 1
        lambda_m = 0
        while abs(sum_pos) > 0.00000000001:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = sum(posidx)
            sum_pos = sum(v1[posidx]) - k
            lambda_m = lambda_m + sum_pos/npos
            ft = ft + 1
            if ft > 100:
                break

        x = [v1[i] if v1[i] >= 0 else 0 for i in range(0, n)]
    else:
        x = v0

    return x, ft


if __name__ == '__main__':
    # Read data
    data_dir = os.path.join(os.path.join(os.path.dirname(os.getcwd()), os.pardir), 'AD-Data_Organized')
    sub = '094_S_4234'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    args = Args()

    t = 0
    A = connectome_list[t]
    dX = 1 - A
    gamma = get_gamma(dX, args.k)
    row, col = A.shape
    S = np.zeros(A.shape)
    for i in range(0, row):
        S[i, :], _ = EProjSimplex(-1 * dX[i, :] / (2 * gamma[i]))

    print("\nAverage number of non zero elements per row before optimizing: ", (A > 0).sum() / 148,
          "\nAverage number of non zero elements per row after optimizing : ", (S > 0).sum() / 148)
    n_comp, label_list = get_number_of_components([S])
    print("\nNumber of component before optimization: ", n_comp)
    # Modularity constraint optimization

    for itr in range(0, args.n_iter):
        if args.debug:
            print("Iteration: ", itr)
        S = (np.transpose(S) + S) / 2
        D_S = np.diag(S.sum(axis=1))
        L = D_S - S
        eig_val, F = get_eigen(L, args.n_module)
        dF = L2_distance(np.transpose(F), np.transpose(F))
        lmd = 0.5
        d = dX + lmd * dF
        gamma = get_gamma(d, args.k)
        S_old = S
        S = np.zeros(A.shape)
        for j in range(0, row):
            S[j, :], _ = EProjSimplex(-1 * d[j, :] / (2 * gamma[j]))

            if args.debug:
                print("Change: ", abs(S - S_old).sum()/S_old.sum())

    print("\nAverage number of non zero elements per row after optimizing : ", (S > 0).sum() / 148)
    print("\nNumber of component after optimization: ", get_number_of_components([S]))

    output_dir = os.path.join(data_dir, sub + '_smoothed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)), 'w') as out:
        np.savetxt(out, S)
        print("Saved file ", os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)))

    create_brain_net_node_files(sub, label_list)
    create_brain_net_node_files(sub + "_smoothed", label_list)
