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


def get_eigen(L, p):
    eigenValues, eigenVectors = LA.eigh(L)
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenValues = (eigenValues > 0) * eigenValues
    count = p
    if p < 1:
        ts = sum(eigenValues) * p
        a = 0
        count = 0
        for e in eigenValues:
            a = a + e
            count = count + 1

            if a > ts:
                break

    eigenVectors = eigenVectors[:, idx]
    return eigenValues[1:count + 2], eigenVectors[:, 1:count + 2], count


def get_gamma(d, k):
    d_new = copy.deepcopy(d)
    d_new.sort(axis=1)
    return np.array(0.5 * (k * d_new[:, k + 1] - d_new[:, 1:k + 1].sum(axis=1)))

def get_gamma_splitted(d, k, r):
    row, col = d.shape
    k1 = int(round(r * k))
    k2 = k - k1
    rr11 = get_gamma(d[0:row//2, 0:col//2], k1)
    rr12 = get_gamma(d[0:row//2, col//2:], k2)
    rr21 = get_gamma(d[row//2:, 0:col//2], k2)
    rr22 = get_gamma(d[row//2:, col//2:], k1)

    return rr11, rr12, rr21, rr22



def get_data_folder(subject):
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

def sort_idx2(distX, k, r):
    dim = distX.shape[0]
    idx_11 = np.argsort(distX[0:dim//2, 0:dim//2], axis=1)
    idx_21 = np.argsort(distX[dim//2:, 0:dim//2], axis=1)
    idx_12 = np.argsort(distX[0:dim//2, dim//2:], axis=1) + dim//2
    idx_22 = np.argsort(distX[dim//2:, dim//2:], axis=1) + dim//2
    c = int(round(k * r)) + 1
    idx = np.vstack((np.hstack((idx_11[:, 0:c], idx_12[:, 0:k+2 - c])), np.hstack((idx_21[:, 0:k+2-c], idx_22[:, 0:c]))))
    return idx, idx_11[:, 0:c], idx_12[:, 0:k+2-c], idx_21[:, 0:k+2-c], idx_22[:, 0:c]

def sort_idx(distX, k, r):
    dim = distX.shape[0]
    idx_11 = np.argsort(distX[0:dim//2, 0:dim//2], axis=1)
    idx_21 = np.argsort(distX[dim//2:, 0:dim//2], axis=1)
    idx_12 = np.argsort(distX[0:dim//2, dim//2:], axis=1) + dim//2
    idx_22 = np.argsort(distX[dim//2:, dim//2:], axis=1) + dim//2
    c = int(round(k * r)) + 1
    idx = np.vstack((np.hstack((idx_11[:, 0:c], idx_12[:, 0:k+2 - c])), np.hstack((idx_21[:, 0:k+2-c], idx_22[:, 0:c]))))
    return idx, idx_11[:, 0:c], idx_12[:, 0:k+2-c], idx_21[:, 0:k+2-c], idx_22[:, 0:c]


def rescale_matrix(a, factor):
    print(factor.shape, a.shape)
    return a * factor

def get_entropy(matrix):
    hist, bins = np.histogram(matrix, density=True)
    hist /= hist.sum()
    return -1 * np.dot(hist, np.ma.log(hist))


def get_histogram(matrix, bins=[]):
    matrix = np.log(np.array(matrix) + 1)

    if len(bins) == 0:
        bins = 30
    hist, bins = np.histogram(matrix, bins=bins, density=False)
    #hist, bins = np.histogram(matrix)
    return hist, bins


def get_histogram_list(mat_list, bins=None):
    hist = []
    for matrix in mat_list:
        if bins == None:
            hist.append(get_histogram(matrix))
        else:
            hist.append(get_histogram(matrix, bins[t]))

    return hist


def get_top_links(connectome, count=1, offset=0):
    row, col = connectome.shape
    idx = np.argsort(connectome, axis=None)[::-1]
    idx_row = idx // col
    idx_col = idx % col
    idx_coord = list(zip(idx_row, idx_col))
    return idx_coord[offset:count + offset]


if __name__ == '__main__':
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    sub = '027_S_2336'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub), False)
    T = len(connectome_list)
    for t in range(0, T):
        idx = get_top_links(connectome_list[t], count=20)
        print("\nt = ", t)
        for a, b in idx:
            print("A(", a, b, ") = ", connectome_list[t][a, b])
    '''
    f = connectome_list[0].sum(axis=1)[:, np.newaxis]
    total = connectome_list[0].sum()
    print(total)
    threshold = 1e-5
    se = []
    re = []
    smoothed_connectome = readMatricesFromDirectory(os.path.join(data_dir, sub + '_smoothed'), False)
    for t in range(0, len(smoothed_connectome)):
        smoothed_connectome[t] = rescale_matrix(smoothed_connectome[t], f)/total
        connectome_list[t] = connectome_list[t]/total
        connectome_list[t] = (connectome_list[t] > threshold) * connectome_list[t]
        smoothed_connectome[t] = (smoothed_connectome[t] > threshold) * connectome_list[t]
        se.append(get_entropy(smoothed_connectome[t]))
        re.append(get_entropy(connectome_list[t]))

    print(se,
          '\n',
          re)
    plt.plot(se, color='b')
    plt.plot(re, color='r')
    #plt.ylim(0, 0.3)

    
    hist1 = get_histogram_list(connectome_list)
    bins1 = [hist1[t][1] for t in range(0, len(connectome_list))]
    hist2 = get_histogram_list(smoothed_connectome, bins=bins1)
    n = len(hist1)
    row = (n + 1) // 2
    col = 2

    for t in range(0, n - 1):
        h1, b1 = hist1[t]
        h2, b2 = hist2[t]
        b1 = b1[1:]
        plt.subplot(row, col, t + 1)
        plt.plot(b1[1:], h1[1:], color='r')
        plt.plot(b2[1:-1], h2[1:], color='b')
        print(b1[0:2],
              b2[0:2])
        #plt.bar(b[1:], h[1:], width=b[2] - b[1])
        #plt.yscale("log", nonposy="clip")
        plt.ylim(0, 150)
    '''



    plt.show()

