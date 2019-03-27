from __future__ import print_function

from args import Args

print(__doc__)

import time
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from utils.readFile import get_subject_info, readSubjectFiles
from grakel import datasets
from grakel import GraphKernel
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from plot_functions import plot_confusion_matrix
from utils.helper import threshold_all
import numpy as np
import collections

def sec_to_time(sec):
    """Print time in a correct format."""
    dt = list()
    days = int(sec // 86400)
    if days > 0:
        sec -= 86400*days
        dt.append(str(days) + " d")

    hrs = int(sec // 3600)
    if hrs > 0:
        sec -= 3600*hrs
        dt.append(str(hrs) + " h")

    mins = int(sec // 60)
    if mins > 0:
        sec -= 60*mins
        dt.append(str(mins) + " m")

    if sec > 0:
        dt.append(str(round(sec, 2)) + " s")
    return " ".join(dt)

def read_data(size=3, sel_label=["1", "3"]):
    # Reading one scan( both raw and smooth) per subject and their DX label
    sub_info = get_subject_info(size)
    X1 = []  # raw
    X2 = []  # smooth
    y = []
    N = 148

    for sub in sub_info.keys():
        rw_all, sm_all = readSubjectFiles(sub, "row")
        t = 0
        assert len(rw_all) == len(sm_all), "Size mismatch in sub " + sub + " " + str(len(rw_all)) + " " + str(len(sm_all))

        if sub_info[sub][0]["DX"] not in sel_label:
            continue

        X1.append(rw_all[0])
        y.append(sub_info[sub][0]["DX"])
        X2.append(sm_all[0])

    print(collections.Counter(y))
    return X1, X2, y


def smote(X, y):
    sm = SMOTE(random_state=1)
    shape = X[0].shape
    X = [G.flatten() for G in X]
    X, y = sm.fit_sample(X, y)
    X = [G.reshape(shape) for G in X]
    return X, y

# Loads the MUTAG, ENZYMES dataset from:
# https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
# the biggest collection of benchmark datasets for graph_kernels.

def prepare_data(G, y, random_state=10, threshold=None):
    if threshold is not None:
        G = threshold_all(G, vmin=threshold)

    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.8, random_state=random_state)
    G_train, y_train = smote(G_train, y_train)

    node_label = {i: i for i in range(N)}
    G_train = [[G, node_label] for G in G_train]
    G_test = [[G, node_label] for G in G_test]

    return G_train, G_test, y_train, y_test


if __name__ == '__main__':
    kernels = {
        "Random-Walk": [{"name":"random_walk"}]
        #"Weisfeiler-Lehman/Subtree": [{"name": "weisfeiler_lehman", "niter": 5},
        #                              {"name": "subtree_wl"}]
    }
    rows = sorted(list(kernels.keys()))
    data_dataset = list()
    data_kernel_rw = list()
    data_kernel_sm = list()
    G_rw, G_sm, y = read_data(3)
    N = len(G_rw[0])

    labels = {'1': 'NC', '2': 'MCI', '3': 'AD'}
    rw_ac = []
    sm_ac = []
    for iter in range(3):
        print("Iter: ", iter)
        # Train-test split of graph data
        G_train_rw, G_test_rw, y_train_rw, y_test_rw = prepare_data(G_rw, y, random_state=iter)
        G_train_sm, G_test_sm, y_train_sm, y_test_sm = prepare_data(G_sm, y, random_state=iter)

        print("Data Set prepared")
        for (i, k) in enumerate(rows):
            print(k, end=" ")
            gk = GraphKernel(kernel=kernels[k], normalize=True)
            print("", end=".")

            # Calculate the kernel matrix for raw data
            start = time.time()
            K_train_rw = gk.fit_transform(G_train_rw)
            K_test_rw = gk.transform(G_test_rw)
            end = time.time()
            print("", end=".")

            # Initialise an SVM and fit.
            clf = svm.SVC(kernel='precomputed')
            clf.fit(K_train_rw, y_train_rw)
            print("", end=". ")

            # Predict and test.
            y_pred_rw = clf.predict(K_test_rw)
            print("Confusion Matrix: \n", confusion_matrix(y_test_rw, y_pred_rw))
            plot_confusion_matrix(y_test_rw, y_pred_rw, labels, title="Confusion Matrix Before Smoothing")

            # Calculate accuracy of classification.
            data_kernel_rw.append(
                sec_to_time(round(end - start, 2)) +
                " ~ " + str(round(accuracy_score(y_test_rw, y_pred_rw)*100, 2)) + "%")
            rw_ac.append((accuracy_score(y_test_rw, y_pred_rw)))
            print("Raw: ", data_kernel_rw[-1])

            # Calculate the kernel matrix for Smooth data
            start = time.time()
            K_train_sm = gk.fit_transform(G_train_sm)
            K_test_sm = gk.transform(G_test_sm)
            end = time.time()
            print("", end=".")

            # Initialise an SVM and fit.
            clf = svm.SVC(kernel='precomputed')
            clf.fit(K_train_sm, y_train_sm)
            print("", end=". ")

            # Predict and test.
            y_pred_sm = clf.predict(K_test_sm)
            print("Confusion Matrix: \n", confusion_matrix(y_test_sm, y_pred_sm))
            plot_confusion_matrix(y_test_sm, y_pred_sm, labels, title="Confusion Matrix After Smoothing")

            # Calculate accuracy of classification.
            sm_ac.append(accuracy_score(y_test_sm, y_pred_sm))
            data_kernel_sm.append(
                sec_to_time(round(end - start, 2)) +
                " ~ " + str(round(accuracy_score(y_test_sm, y_pred_sm) * 100, 2)) + "%")
            print("Smooth: ", data_kernel_sm[-1])
        data_dataset.append(data_kernel_sm)
        print("")

    print("Raw Ac: ", rw_ac,
          "\nMean: ", np.mean(rw_ac))
    print("\nSmooth Ac: ", sm_ac,
          "\nMean: ", np.mean(sm_ac))
