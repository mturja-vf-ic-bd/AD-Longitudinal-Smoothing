import math
import os
from os.path import join
import numpy as np


def readMatrixFromTextFile(fname, debug = False):
    if debug == True:
        print("Reading File: " + fname)
    a = []
    fin = open(fname, 'r')
    for line in fin.readlines():
        a.append([float(x) for x in line.split()])

    a = np.asarray(a)
    a = (a + np.transpose(a))/2
    return a

def readMatricesFromDirectory(directory):
    files = [f for f in os.listdir(directory)]
    mat_list = []
    for file in files:
        a = readMatrixFromTextFile(join(directory, file))
        a /= a.sum()
        mat_list.append(a)

    return mat_list