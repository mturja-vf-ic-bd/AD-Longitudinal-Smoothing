from read_file import *
import numpy as np

def process_data(data):
    F = []
    M = data['adjacency_matrix']
    for i in range(len(M)):
        F.append(data['node_feature'][i][:, 0])
        M[i] = (M[i] + M[i].T)/2
        M[i] /= M[i].sum()

    return F, M

def total_variation(M, x):
    D = np.diag(M.sum(axis=1))
    L = D - M
    return np.dot(np.dot(x.T, L), x)


if __name__ == '__main__':
    subject_id = '027_S_5109'
    a = read_subject_data(subject_id)
    F, M = process_data(a)

    for i in range(len(F)):
        print(total_variation(M[i], F[i]))
