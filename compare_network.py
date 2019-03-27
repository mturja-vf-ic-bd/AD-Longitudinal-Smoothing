from collections import Counter
from utils.get_hemisphere import *
from bct import community_louvain
from matplotlib import pyplot as plt
import operator
import numpy as np
import os

def n_comp(M, gamma=1):
    c_i, q = community_louvain(np.asarray(M), gamma=gamma)
    return len(set(c_i))


def get_number_of_components(connectomes):
    n_component = []
    label_component = []

    for connectome in connectomes:
        c_i, q = community_louvain(np.asarray(connectome), gamma=1)
        n_component.append(len(set(c_i)))
        label_component.append(c_i)

    return n_component, label_component


def compare_network(a, b):
    a_i, a_q = community_louvain(np.asarray(a))
    b_i, b_q = community_louvain(np.asarray(b))

    a_i_set = set(a_i)
    b_i_set = set(b_i)

    if len(a_i_set) > len(b_i_set):
        a_i, b_i = b_i, a_i
        a_i_set, b_i_set = b_i_set, a_i_set

    bidirectional_mapping = {}
    for a_label in a_i_set:
        a_ind = [i for i, value in enumerate(a_i) if value == a_label]
        b_label = max([b_i[i] for i in a_ind])
        bidirectional_mapping[a_label] = b_label

    return np.array([bidirectional_mapping[a_i[i]] == b_i[i] for i in range(len(a_i))]).sum() * 100 / len(a_i)


def get_sorted_index(c1):
    hist = Counter(c1)
    return sorted(hist.items(), key=operator.itemgetter(1))


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    subjects = [f for f in os.listdir(data_dir)]
    matches = []
    for sub in subjects:
        mat_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
        lh_list = [get_left_hemisphere(c) for c in mat_list]
        rh_list = [get_right_hemisphere(c) for c in mat_list]

        n_comp, label_comp = get_number_of_components(mat_list)
        lh_n_comp, lh_label_comp = get_number_of_components(lh_list)
        rh_n_comp, rh_label_comp = get_number_of_components(rh_list)

        comp_pair = [(rh_list[i], rh_list[i + 1]) for i in range(0, len(rh_list) - 1)]
        match = [compare_network(a, b) for a, b in comp_pair]
        print(match)
        matches.append(np.average(match))

    plt.hist(matches)
    plt.show()



