import math
import os
from os.path import join
import numpy as np


def readMatrixFromTextFile(fname, debug=False):
    if debug == True:
        print("Reading File: " + fname)
    a = []
    fin = open(fname, 'r')
    for line in fin.readlines():
        a.append([float(x) for x in line.split()])

    a = np.asarray(a)
    return a


def readMatricesFromDirectory(directory, normalize=True):
    files = [f for f in os.listdir(directory)]
    files.sort()
    mat_list = []
    for file in files:
        file = os.path.join(directory, file)
        if os.path.isdir(file):
            print(file, " is a directory")
        elif os.path.isfile(file):
            print("Reading ", file)
            a = readMatrixFromTextFile(join(directory, file))
            if normalize:
                a = (a + np.transpose(a)) / 2
                row_sums = a.sum(axis=1)
                a /= row_sums[:, np.newaxis]
            mat_list.append(a)
        else:
            print(file, " is weird")

    return mat_list
