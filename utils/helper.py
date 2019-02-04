import os
from numpy import linalg as LA
from utils.L2_distance import *
import copy
from args import Args
import json

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

    weighted_mat_list = [np.divide(np.multiply(np.array(mat_list[i]), np.array(weights[i])), total_weight) for i in range(0, len(mat_list))]
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
    return np.array(0.5 * (k * d_new[:, k+1] - d_new[:, 1:k+1].sum(axis=1)))


def get_data_folder(subject):
    args = Args()
    return os.path.join(os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized'), subject)


def get_coordinates(subject):
    table_file = os.path.join(get_data_folder(subject), 'helper_files/parcellationTable.json')
    with open(table_file, 'r') as f:
        table = json.load(f)
        coord = [row["coord"] for row in table]

    return coord

def row_normalize(A):
    A = np.array(A)
    row_sum = A.sum(axis=1) + 10e-10
    A = A / row_sum[:, np.newaxis]
    return A

if __name__ == '__main__':
    mat = np.array([[1, 2, 3], [5, 1, 6], [11, 13, 9]])
    gamma = get_gamma(mat, 1)
    print(gamma)
    # Add more test

    row_normalize(mat)
    print(mat)
