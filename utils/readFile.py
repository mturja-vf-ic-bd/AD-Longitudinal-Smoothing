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
                a = normalize_matrix(a)
            mat_list.append(a)
        else:
            print(file, " is weird")

    return mat_list

def normalize_matrix(mat):
    mat = (mat.T + mat)/2
    row_sums = mat.sum(axis=1)
    mat /= row_sums[:, np.newaxis]
    return mat


if __name__ == '__main__':
    from args import Args
    import collections
    from matplotlib import pyplot as plt
    import plotly.plotly as py
    import plotly.tools as tls

    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir) , 'AD-Data_Organized')
    rect_files = []
    for f in os.listdir(data_dir):
        if f.find('smoothed') == -1:
            rect_files.append(f)

    count = [len(os.listdir(os.path.join(data_dir, f))) for f in rect_files]
    hist = collections.Counter(count)
    print(hist)

    plt.hist(count, align='left', bins=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('#scans')
    plt.ylabel('#subjects')
    plt.show()




