import numpy as np
from args import Args
from utils.EProjSimplex import *
from utils.helper import *
from utils.readFile import readMatricesFromDirectory
from numpy.core import isfinite


def CAN(distX, c, k=15, r=-1, islocal=True):
    """

    :param distX: n x n distance matrix for n data points
    :param c: number of clusters
    :param k: number of neighbors to determine the initial graph, and the parameter r if r<=0
    :param r: paremeter, which could be set to a large enough value. If r<0, then it is determined by algorithm with k
    :param islocal:
        1: only update the similarities of the k neighbor pairs, faster
        0: update all the similarities
    :return:
        A: num*num learned symmetric similarity matrix
        evs: eigenvalues of learned graph Laplacian in the iterations
    """

    num = distX.shape[0]
    arg = Args()
    NITER = arg.n_iter

    if arg.debug:
        print('\nInitial Parameters:',
              '\nc = ', c,
              '\nk = ', k,
              '\nr = ', r,
              '\nislocal = ', islocal,
              '\ndimension = ', num,
              '\nNITER = ', NITER)

    distX1 = np.sort(distX, axis=1)
    idx = np.argsort(distX, axis=1)
    A = np.zeros((num, num))
    rr = get_gamma(distX, k)

    if r < 0:
        r = np.mean(rr)

    lmd = np.mean(rr)

    eps = 10e-10
    for i in range(0, num):
        A[i, idx[i, 1: k + 2]] = 2 * (distX1[i, k + 1] - distX1[i, 1: k + 2]) / (r + eps)

    print("\nAverage number of non zero elements per row before optimizing: ", (A > 0).sum() / 148,
          "\nNumber of negative elements: ", (A < 0).sum())

    A0 = (A + A.T) / 2
    A0 = row_normalize(A0)
    D0 = np.diag(A0.sum(axis=1))
    L0 = D0 - A0
    evs, F = get_eigen(L0, c + 1)  # Taking c + 1 eig values
    F = F[:, 0:c]  # removing last one
    ev = []
    if sum(evs) < 10e-10:
        raise Exception('The number of connected component in the graph is greater than {}', c)

    for iter in range(0, NITER):
        distF = L2_distance(F.T, F.T)
        distF = np.sqrt((distF >= 0) * distF)
        A = np.zeros((num, num))

        for i in range(0, num):
            if islocal:
                idxa0 = idx[i, 1:k+1]
            else:
                idxa0 = np.arange(num)

            dfi = distF[i, idxa0]
            dxi = distX[i, idxa0]
            d = dxi + lmd * dfi
            ad = -d / (2 * r)
            res, _ = EProjSimplex(ad)
            A[i, idxa0] = res

        np.fill_diagonal(A, 0)
        A = (A + A.T) / 2
        A = row_normalize(A)
        D = np.diag(A.sum(axis=1))
        L = D - A
        F_old = F

        evs, F = get_eigen(L, c + 1)
        ev.append(evs)
        F = F[:, 0:c]  # removing last one

        if sum(evs[0:c]) > 10e-10:
            lmd = 2 * lmd
        elif sum(evs) < 10e-10:
            lmd = lmd / 2
            F = F_old
        else:
            break
        print("Iter:", iter,
              "lambda = ", lmd,
              "Regularization Parameter r = ", r)

    return A, ev


if __name__ == '__main__':
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '027_S_2336'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    args = Args()

    output_dir = os.path.join(data_dir, sub + '_smoothed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    smoothed_connectomes = []
    for t in range(0, len(connectome_list)):
        A = connectome_list[t]
        dX = np.sqrt(1 - A)
        np.fill_diagonal(dX, 2)
        S, _ = CAN(dX, args.n_module, k=args.k, islocal=True)
        #S = (S.T + S)/2
        np.fill_diagonal(S, 0)
        S = row_normalize(S)
        smoothed_connectomes.append(S)

        print("\nAverage number of non zero elements before optimizing: ", (A > 0.3).sum(),
              "\nAverage number of non zero elements after optimizing : ", (S > 0.3).sum())

        with open(os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)), 'w') as out:
            np.savetxt(out, S)
            print("Saved file ", os.path.join(output_dir, sub + "_smoothed_t" + str(t + 1)))

    n_comp_, label_list = get_number_of_components(connectome_list)
    print("\nNumber of component: ", n_comp_)
    n_comp_, label_list = get_number_of_components(smoothed_connectomes)
    print("\nNumber of component: ", n_comp_)

    #import test_result
    #test_result.test_result(sub, connectome_list, smoothed_connectomes)




