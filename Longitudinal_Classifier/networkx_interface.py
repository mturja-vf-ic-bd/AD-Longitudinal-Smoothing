import networkx as nx

def convert_to_netx(A):
    return nx.convert_matrix.from_numpy_array(A, create_using=nx.MultiGraph, parallel_edges=False)

def convert_to_netx_list(A_list):
    return [convert_to_netx(A) for A in A_list]