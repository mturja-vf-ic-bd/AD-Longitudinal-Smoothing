from numpy import linalg as LA
from args import Args
import json
import numpy as np
import copy
import os
from bct import *
import sys


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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


def get_gamma2(d, k):
    d_new = copy.deepcopy(d)
    d_new.sort(axis=1)
    gamma = []
    for i in range(0, len(d_new)):
        gamma.append(0.5 * (k[i] * d_new[i, k[i]] - d_new[i, 0:k[i]].sum()))

    return np.array(gamma)

def get_gamma(d, k):
    d_new = copy.deepcopy(d)
    d_new.sort(axis=1)
    return np.array(0.5 * (k * d_new[:, k + 1] - d_new[:, 1:k + 1].sum(axis=1)))

def get_gamma_splitted(d, K1, K2, mean=False):
    row, col = d.shape
    rr11 = get_gamma2(d[0:row//2, 0:col//2], K1[0:row//2])
    rr12 = get_gamma2(d[0:row//2, col//2:], K2[0:row//2])
    rr21 = get_gamma2(d[row//2:, 0:col//2], K1[row//2:])
    rr22 = get_gamma2(d[row//2:, col//2:], K2[row//2:])

    if mean:
        return np.mean(rr11), np.mean(rr12), np.mean(rr21), np.mean(rr22)

    return rr11, rr12, rr21, rr22


def get_scan_count(subject):
    path = get_data_folder(subject)
    return len(os.listdir(path)) - 1


def get_subject_names(count=0):
    dir = os.path.join(os.path.join(Args.root_directory, os.pardir), 'AD-Data_Organized')
    sub_names = os.listdir(dir)
    return [s for s in sub_names if 'smoothed' not in s and get_scan_count(s) > count]


def get_data_folder(subject):
    return os.path.join(os.path.join(os.path.join(Args.root_directory, os.pardir), 'AD-Data_Organized'), subject)


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

def sort_idx2(distX):
    dim = distX.shape[0]
    idx_11 = np.argsort(distX[0:dim//2, 0:dim//2], axis=1)
    idx_12 = np.argsort(distX[0:dim//2, dim//2:], axis=1) + dim//2
    idx_21 = np.argsort(distX[dim//2:, 0:dim//2], axis=1)
    idx_22 = np.argsort(distX[dim//2:, dim//2:], axis=1) + dim//2
    idx = np.vstack((np.hstack((idx_11, idx_12)), np.hstack((idx_21, idx_22))))
    return idx

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
    '''
    Multiplies each row of matrix 'a' with values in the vector 'factor'
    :param a: n * n matrix
    :param factor: (n, ) vector
    :return: a multiplied by each values in vector 'factor'
    '''

    return a * factor[:, np.newaxis]


def get_entropy_matrix(matrix):
    hist, bins = np.histogram(matrix, density=True)
    hist /= hist.sum()
    return -1 * np.dot(hist, np.ma.log(hist))


def get_entropy_list(in_list):
    in_list = np.array(in_list)
    in_list /= in_list.sum()
    return -1 * np.dot(in_list.T, np.ma.log(in_list))


def get_histogram(matrix, bins=[]):
    matrix = np.log(np.array(matrix) + 1)
    if len(bins) == 0:
        bins = 30
    hist, bins = np.histogram(matrix, bins=bins, density=False)
    return hist, bins


def get_histogram_list(mat_list, bins=None):
    hist = []
    for matrix in mat_list:
        if bins == None:
            hist.append(get_histogram(matrix))
        else:
            hist.append(get_histogram(matrix, bins[t]))

    return hist


def get_top_links(connectome, count=1, offset=0, weight=False):
    connectome = np.array(connectome)
    row, col = connectome.shape
    idx = np.argsort(connectome, axis=None)[::-1]
    idx_row = idx // col
    idx_col = idx % col
    if not weight:
        idx_coord = list(zip(idx_row, idx_col))
    else:
        idx_coord = list(zip(idx_row, idx_col, connectome[idx_row, idx_col]))

    return idx_coord[offset:count + offset]


def get_centrality_measure(M):
    bc = betweenness_wei(M)
    bc /= bc.sum()
    return np.dot(bc, M.sum(axis=1))


def groupElementsOfMatrices(input_mat_list):
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


def findMeanAndStd(mat_list):
    dim = len(mat_list[0])
    element_list = groupElementsOfMatrices(mat_list)
    stddev = [np.std(element_list[i]) for i in range(0, dim*dim)]
    meanind = [np.average(element_list[i]) for i in range(0, dim*dim)]

    return np.array(meanind).reshape(dim, dim), np.array(stddev).reshape(dim, dim)

def find_distance_between_matrices(mat_list):
    d = []
    for i in range(0, len(mat_list)):
        for j in range(i, len(mat_list)):
            d.append(((mat_list[i] - mat_list[j])**2).sum())

    return d


def add_noise(connectome, t=0.1, type='gaussian'):
    """
    adds gaussian noise to connectome
    :param connectome:
    :return:
    """
    shape = connectome.shape
    noise = np.zeros(shape)
    for row in range(shape[0]):
        for col in range(shape[1]):
            if type == 'gamma':
                noise[row][col] = np.random.gamma(connectome[row][col], t * connectome[row][col])
            else:
                a = connectome[row][col]
                if a > 0:
                    noise[row][col] = np.random.normal(a, t * a)
                else:
                    noise[row][col] = np.random.uniform(low=0, high=0.02)

    noise = np.clip(noise, 0, 1)
    return noise


def add_noise_all(connectome_list, noise):
    connectome_list_noisy = []
    for t in range(len(connectome_list)):
        connectome_list_noisy.append(add_noise(connectome_list[t], noise))

    return connectome_list_noisy


def threshold(connectome, vmin=0, vmax=1):
    connectome_th = copy.deepcopy(connectome)
    connectome_th[connectome_th < vmin] = 0
    connectome_th[connectome_th > vmax] = 0
    return connectome_th

def threshold_all(connectome_list, vmin=0, vmax=1):
    connectome_list_th = [threshold(connectome_list[t], vmin, vmax) for t in range(len(connectome_list))]
    return connectome_list_th


def connectome_median(connectome_list):
    assert len(connectome_list) > 0, "Empty connectome list"
    shape = connectome_list[0].shape
    element_list = groupElementsOfMatrices(connectome_list)
    return np.median(element_list, axis=1).reshape(shape)


def get_avg_zeros_per_row(connectome_list):
    res = 0
    for connectome in connectome_list:
        res = res + np.mean((connectome == 0).sum(axis=1))

    res = res / len(connectome_list)
    return res

def plot_matrix(connectome, fname="connectome", vmin=0, vmax=0.25):
    from matplotlib import pyplot as plt
    im = plt.matshow(connectome, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.show()

def rescale_sm_mat_to_raw(raw, sm):
    max_rw = np.max(raw, axis=None)
    max_sm = np.max(sm, axis=None)
    return min(max_rw/max_sm, 1.4) * sm

def get_parcellation_table():
    pt_name = Args.root_directory + "/utils/parcellationTable_Ordered.json"  # parcellation table to edit VisuOrder
    # Read parcellation table to edit VisuOrder
    with open(pt_name) as f:
        pt = json.load(f)
    f.close()
    return pt
    
def get_lobe_idx():
    pt = get_parcellation_table()
    lobe_idx = {}
    for entry in pt:
        lobe = entry["VisuHierarchy"].split('.')[-1]
        row = entry["MatrixRow"]
        if row < len(pt) / 2:
            lobe = lobe + '.l'
        else:
            lobe = lobe + '.r'

        if lobe in lobe_idx.keys():
            lobe_idx[lobe].append(row)
        else:
            lobe_idx[lobe] = [row]

    return lobe_idx

def get_lobe_order(ignore_hem=False):
    if ignore_hem:
        return ["Frontal", "Limbic", "Insula", "Temporal", "Occipital", "Parietal",
                "Parietal", "Occipital", "Temporal", "Insula", "Limbic", "Frontal"]

    return ["Frontal.l", "Limbic.l", "Insula.l", "Temporal.l", "Occipital.l", "Parietal.l",
                  "Parietal.r", "Occipital.r", "Temporal.r", "Insula.r", "Limbic.r", "Frontal.r"]

def get_sorted_node_count():
    lobe_order = get_lobe_order()
    lobe_idx = get_lobe_idx()
    node_count = []
    for lobe in lobe_order:
        node_count.append(len(lobe_idx[lobe]))

    return node_count

def forward_diff(feature):
    T, n = feature.shape
    diff = np.empty((T - 1, n))
    for i in range(0, T - 1):
        diff[i, :] = (feature[i + 1] - feature[i])

    return diff

def forward_diff_2(feature):
    T, n = feature.shape
    diff = np.empty((T - 2, n))
    for i in range(1, T - 1):
        diff[i - 1, :] = (feature[i + 1] + feature[i - 1] - 2 * feature[i])
    return diff

def central_difference_of_links(connectome_list):
    n_feat = connectome_list[0].shape[0] * connectome_list[0].shape[1]
    feature = np.empty((len(connectome_list), n_feat))
    for i, cn in enumerate(connectome_list):
        feature[i] = cn.flatten()

    diff_2 = forward_diff_2(feature)
    #feature[feature == 0] = np.inf
    #percent_diff_2 = diff_2 / feature[1:-1]
    return np.mean(np.abs(diff_2), axis=None)



if __name__ == '__main__':
    get_lobe_idx()
