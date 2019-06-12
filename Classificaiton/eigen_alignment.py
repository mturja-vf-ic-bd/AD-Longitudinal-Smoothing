"""
Project all the temporal networks to the same eigen-system and measure change of the
total variation in that eigen-system
"""

from utils.readFile import readSubjectFiles
from numpy import linalg as LA
import numpy as np

def diff_eig_val(temp_net):
    print("*** Project all the temporal networks to the "
          "same eigen-system and measure change of the "
          "total variation in that eigen-system ***")

    t = len(temp_net)
    baseline = temp_net[0]
    eigval = np.zeros((len(baseline), t))
    eigval[:, 0], eigvec = LA.eigh(baseline)

    for i in range(1, t):
        D = np.diag(temp_net[i].sum(axis=1))
        L = D - temp_net[i]
        projection = np.matmul(np.matmul(eigvec.T, L), eigvec)
        #eigval[:, i] = np.diag(projection)
        eigval[:, i] = np.diag(np.sqrt(np.dot(projection, projection.T)))

    diff = np.diag(np.ones(t - 1) * -1)
    diff = np.vstack((np.ones((1, t - 1)), diff))
    diff_eigval = np.dot(eigval, diff)
    normalizer = eigval[:, 0][:, np.newaxis]
    #diff_eigval /= normalizer
    print(diff_eigval)

def diff_eig_vec(temp_net, eig_num=0):
    print("*** Angular difference between "
          "eigen vectors of the temporal networks ***")

    t = len(temp_net)
    n = len(temp_net[0])
    eigvec = np.zeros((n, t))

    for i in range(0, t):
        D = np.diag(temp_net[i].sum(axis=1))
        L = D - temp_net[i]
        _, eig = LA.eigh(L)
        eigvec[:, i] = eig[:, eig_num]

    ang_dist = np.abs(np.dot(eigvec.T, eigvec))
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print(ang_dist)


if __name__ == '__main__':
    # Input
    s_name = '027_S_5109'
    _, temp_net = readSubjectFiles(s_name)
    temp_net, _ = readSubjectFiles(s_name)
    diff_eig_vec(temp_net, 4)
    # diff_eig_val(temp_net)
