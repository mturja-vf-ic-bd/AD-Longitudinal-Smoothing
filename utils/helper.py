import numpy as np
from numpy import linalg as LA
from utils.L2_distance import *
import copy


def group_elements_of_matrices(input_mat_list):
    mat_list = []
    i = 1
    for mat in input_mat_list:
        mat_list.append(np.asarray(mat).flatten().tolist())
        i = i + 1

    element_list = []
    for j in range(0, len(mat_list[0])):
        el = []
        for i in range(0, len(mat_list)):
            el.append(mat_list[i][j])
        element_list.append(el)

    return element_list


def find_mean(mat_list, weights):
    total_weight = np.zeros(mat_list[0].shape)
    for w in weights:
        total_weight = np.add(w, total_weight)

    weighted_mat_list = [np.divide(np.multiply(np.asarray(mat_list[i]), np.asarray(weights[i])), total_weight) for i in range(0, len(mat_list))]
    avg = np.zeros(mat_list[0].shape)

    for wm in weighted_mat_list:
        avg = np.add(avg, wm)

    return avg


def get_eigen(L, c):
    eigenValues, eigenVectors = LA.eig(L)
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return eigenValues[0:c], eigenVectors[:, 0:c]


def get_gamma(d, k):
    d_new = copy.deepcopy(d)
    d_new.sort(axis=1)
    return np.array(0.5 * (k * d_new[:, k] - d_new[:, 0:k].sum(axis=1)))


if __name__ == '__main__':
    mat = np.array([[1, 2, 3], [5, 1, 6], [11, 13, 9]])
    gamma = get_gamma(mat, 1)
    print(gamma)
    # Add more test
