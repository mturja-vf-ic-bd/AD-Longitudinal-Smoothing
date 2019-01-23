import numpy as np
from numpy import linalg as LA


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


if __name__ == '__main__':
    mat_list = [np.array([[1, 2, 3], [4, 5, 7]]), np.array([[3, 2, 1], [6, 5, 4]])]
    w = [np.array([[0.2, 0.4, 0.9], [0.5, 0.5, 0.5]]), np.array([[0.1, 0.6, 0.4], [0.5, 0.5, 0.5]])]
    mn = np.random.random((10, 10))
    M = np.matmul(mn, np.transpose(mn))
    L = np.diag(M.sum(axis=1)) - M
    ev, evec = get_eigen(L, 3)
    print('\neigen value: ', ev,
          '\neigen vector: ', evec)
