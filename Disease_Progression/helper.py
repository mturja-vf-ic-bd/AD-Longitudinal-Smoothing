import numpy as np
import collections

def forward_diff(con_list):
    T, n, m = con_list.shape
    diff = np.empty((T - 1, n, m))
    for i in range(0, T - 1):
        diff[i, :] = (con_list[i + 1] - con_list[i])

    return diff


def degree_distribution(g):
    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    return deg, cnt