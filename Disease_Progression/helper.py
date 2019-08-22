import numpy as np

def forward_diff(con_list):
    T, n, m = con_list.shape
    diff = np.empty((T - 1, n, m))
    for i in range(0, T - 1):
        diff[i, :] = (con_list[i + 1] - con_list[i])

    return diff