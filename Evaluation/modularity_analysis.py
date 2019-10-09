# Modularity Analysis of the raw and smooth connectomes
# The goal is to show that our method produce connectomes that has consistent modularity along time
#from bct import community_louvain
import numpy as np
from utils.readFile import readSubjectFiles
from plot_functions import plot_matrix_all, plot_community_structure_variation
from math import log
from utils.helper import get_subject_names
from collections import Counter
from bct import community_louvain

def match_list(l1, l2):
    return len(set(l1).intersection(set(l2)))

def compare_community_structure(c1, c2):
    matching = {}
    if len(c1) < len(c2):
        c1, c2 = c2, c1

    for i in range(len(c1)):
        max = -1
        max_idx = -1
        for j in range(len(c2)):
            temp = match_list(c1[i], c2[j])
            if temp > max:
                max = temp
                max_idx = j

        matching[i] = max_idx



def build_longitudinal_community_structure(c_i_list):
    for i, c_s in enumerate(c_i_list):
        for j in range(i + 1, len(c_i_list)):
            c_d = c_i_list[j]



def build_community_structure(c_i):
    """
    Returns a list of list with each list representing the indices of a community
    :param c_i: communitiy labels of the nodes of graph
    :return: idx_ordered: nested list with each element is the indices of each community
    """

    community_dict = {}
    label_set = set(c_i)
    for label in label_set:
        idx_c = np.nonzero(c_i == label)[0]
        key = min(idx_c)

        community_dict[key] = idx_c

    idx_ordered = []
    for k in sorted(community_dict.keys()):
        idx_ordered.append(list(community_dict[k]))
    return idx_ordered

def sort_connectomes_by_modularity(mat, c_i=None):
    """
    Sort a matrix by community structure
    :param mat: adjacancy matrix (N*N)
    :return: sorted adjacancy matrix (N*N)
    """
    c_i, q = community_louvain(np.asarray(mat), gamma=1, ci=c_i)
    com_struct = build_community_structure(c_i)
    idx_ordered = []
    for idx in com_struct:
        idx_ordered = idx_ordered + idx
    mat = mat[idx_ordered, :]
    return mat[:, idx_ordered], c_i

def variation_of_information(X, Y):
    n = float(sum([len(x) for x in X]))
    sigma = 0.0
    for x in X:
        p = len(x) / n
        for y in Y:
            q = len(y) / n
            r = len(set(x) & set(y)) / n
            if r > 0.0:
                sigma += r * (log(r / p, 2) + log(r / q, 2))
    return abs(sigma)

def voi_between_community_structure(mat1, mat2, gamma=1):
    """
    Measures normalized variation of information between two community structure
    :param mat1: N*N adjcancy matrix of graph one.
    :param mat2: N*N adjcancy matrix of graph two.
    :return: nvoi: normalized variation of information between two community structure
    """

    c_1, q = community_louvain(np.asarray(mat1), gamma=gamma)
    c_2, q = community_louvain(np.asarray(mat2), gamma=gamma)
    N = len(c_1)

    X = build_community_structure(c_1)
    Y = build_community_structure(c_2)

    return variation_of_information(X, Y) / log(N, 2)

def mean_std_voi(sub_names):
    voi_rw = []
    voi_sm = []
    for sub in sub_names:
        connectome_list, smoothed_connectomes = readSubjectFiles(sub, method="row")

        voi_rw = voi_rw + [voi_between_community_structure(v1, v2) for
                  v1 in connectome_list for v2 in connectome_list if v1 is not v2]
        voi_sm = voi_sm + [voi_between_community_structure(v1, v2) for
                  v1 in smoothed_connectomes for v2 in smoothed_connectomes if v1 is not v2]

    voi_rw_mean = np.mean(voi_rw)
    voi_rw_std = np.std(voi_rw)

    voi_sm_mean = np.mean(voi_sm)
    voi_sm_std = np.std(voi_sm)

    return voi_rw_mean, voi_rw_std, voi_sm_mean, voi_sm_std


if __name__ == '__main__':
    sub_names = get_subject_names(3)
    #sub_names = ["027_S_5110"]
    print(mean_std_voi(sub_names))
    '''
    for sub in sub_names:
        connectome_list, smoothed_connectomes = readSubjectFiles(sub, method="row")
        connectome_list = [sort_connectomes_by_modularity(connectome) for connectome in connectome_list]
        smoothed_connectomes = [sort_connectomes_by_modularity(connectome) for connectome in smoothed_connectomes]
        plot_matrix_all(connectome_list, fname="raw_mod", savefig=True)
        plot_matrix_all(smoothed_connectomes, fname="sm_mod", savefig=True)
        '''










