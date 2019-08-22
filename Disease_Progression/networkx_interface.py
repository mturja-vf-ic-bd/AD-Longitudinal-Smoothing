import networkx as nx

def convert_to_netx(A):
    return nx.from_numpy_array(A)

def convert_to_netx_list(A_list):
    return [convert_to_netx(A) for A in A_list]