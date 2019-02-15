import os
from numpy import linalg as LA
from utils.L2_distance import *
import copy
from args import Args
import json
import matplotlib.pyplot as plt
import warnings
from utils.readFile import *
from utils.helper import *
from matplotlib import pyplot as plt
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


def find_mean(mat_list, weights=None):
    if weights is None and len(mat_list) > 0:
        weights = [np.ones(mat_list[0].shape) for i in range(0, len(mat_list))]
    total_weight = np.zeros(mat_list[0].shape)
    for w in weights:
        total_weight = np.add(w, total_weight)

    total_weight = (total_weight == 0) + total_weight
    weighted_mat_list = [np.divide(np.multiply(np.array(mat_list[i]), np.array(weights[i])), total_weight) for i in range(0, len(mat_list))]
    avg = np.zeros(mat_list[0].shape)

    for wm in weighted_mat_list:
        avg = np.add(avg, wm)

    return avg


def get_eigen(L, c):
    eigenValues, eigenVectors = LA.eig(L)
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenValues = (eigenValues > 10e-10) * eigenValues
    eigenVectors = eigenVectors[:, idx]
    return eigenValues[0:c], eigenVectors[:, 0:c]


def get_gamma(d, k):
    d_new = copy.deepcopy(d)
    d_new.sort(axis=1)
    return np.array(0.5 * (k * d_new[:, k + 1] - d_new[:, 1:k + 1].sum(axis=1)))


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
    row_sum = A.sum(axis=1) + 10e-15
    A = A / row_sum[:, np.newaxis]
    return A


def get_n_modes(mat, threshold=0.9):
    """

    :param mat: Input matrix
    :param threshold: threshold indicates how far the eigen vectors should explain the variation
    :return: number of eigen modes
    """
    eigenValues, eigenVectors = LA.eig(mat)
    if (eigenValues < 0).sum() > 0:
        warnings.warn('Matrix is not positive semidefinite.Replacing with zeros')
        eigenValues = (eigenValues > 0) * eigenValues

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    ts = sum(eigenValues) * threshold
    a = 0
    count = 0
    for e in eigenValues:
        a = a + e
        count = count + 1

        if a > ts:
            break

    return count


def rescale_matrix(a, factor):
    print(factor.shape, a.shape)
    return a * factor


def plot_histogram(matrix):
    matrix = np.log(np.array(matrix) + 1)
    hist, bins = np.histogram(matrix, density=False, range=(10e-10, np.max(matrix)))
    #hist, bins = np.histogram(matrix)
    return hist, bins

def plot_histogram_list(mat_list):
    hist = []
    for matrix in mat_list:
        hist.append(plot_histogram(matrix))

    return hist


if __name__ == '__main__':
    A = np.array([[1,2,0], [3,2,0]])
    '''
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    sub = '027_S_4926'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub), False)
    f = connectome_list[0].sum(axis=1)[:, np.newaxis]
    total = connectome_list[0].sum()
    print(total)
    threshold = 0.0005
    connectome_list.append(find_mean(connectome_list))
    smoothed_connectome = readMatricesFromDirectory(os.path.join(data_dir, sub + '_smoothed'), False)
    for t in range(0, len(smoothed_connectome)):
        smoothed_connectome[t] = rescale_matrix(smoothed_connectome[t], f)/total
        connectome_list[t] = connectome_list[t]/total
        connectome_list[t] = (connectome_list[t] > threshold) * connectome_list[t]

    hist = plot_histogram_list(connectome_list)
    #hist = plot_histogram_list(smoothed_connectome)
    n = len(hist)
    row = (n + 1) // 2
    col = 2

    for t in range(0, n - 1):
        h, b = hist[t]
        print(h)
        print(b)
        b = b[1:]
        plt.subplot(row, col, t + 1)
        plt.bar(b[1:], h[1:], width=b[2] - b[1])
        #plt.yscale("log", nonposy="clip")
        #plt.ylim(0, 0.5)

    plt.show()
    '''
