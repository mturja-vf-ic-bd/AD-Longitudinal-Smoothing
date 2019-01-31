import numpy as np
from args import Args
from utils.EProjSimplex import *
from utils.helper import *
from utils.readFile import readMatricesFromDirectory
from numpy.core import isfinite


def CAN(DX, c, k=15, r=-1, islocal=True):
    """

    :param DX: a temporal sequence of n x n distance matrix for n data points
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

    arg = Args()
    N_ITER = arg.n_iter
    SDX = []

    if arg.debug:
        print('\nInitial Parameters:',
              '\nc = ', c,
              '\nk = ', k,
              '\nr = ', r,
              '\nislocal = ', islocal,
              '\nNITER = ', N_ITER)

    # Initialization
    for distX in DX:
        num = distX.shape[0]
        distX1 = np.sort(distX, axis=1)
        idx = np.argsort(distX, axis=1)
        A = np.zeros((num, num))
        rr = get_gamma(distX, k)

        eps = 10e-10
        for i in range(0, num):
            A[i, idx[i, 1: k + 2]] = 2 * (distX1[i, k + 1] - distX1[i, 1: k + 2]) / (rr[i] + eps)

        SDX.append(A)
        if r < 0:
            r = np.mean(rr)

        lmd = np.mean(rr)

    wt_local = [np.ones(args.c_dim) / len(connectome_list) for i in range(0, len(connectome_list))]

    for itr in range(0, N_ITER):
        for t in range(0, len(DX)):
            distX = DX[t]
            A = SDX[t]


    return A

