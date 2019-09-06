import numpy as np
import collections
import networkx as nx

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

def degree_distribution_weighted(A):
    deg = A.sum(axis=1)
    hist, bin_edges = np.histogram(deg, density=False, bins=50)
    return hist, bin_edges[1:]

def get_btw_centrality(g):
    bt_dict = nx.betweenness_centrality(g, weight='weight')
    ordered_dict = sorted(bt_dict.items(), key=lambda kv: kv[0])
    values_list = [a[1] for a in ordered_dict]
    return values_list

def btw_cen_dist(g):
    btw_cen = get_btw_centrality(g)
    hist, bin_edges = np.histogram(btw_cen, bins=50)
    return hist, bin_edges[1:]

def diff_nets(conn_list):
    diff = [np.abs(conn_list[i + 1] - conn_list[i]) for i in range(len(conn_list) - 1)]
    return diff