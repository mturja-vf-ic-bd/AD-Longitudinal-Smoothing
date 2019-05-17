"""
Fit gaussian mixture model with the MMSE and CDR-SB features of the ADNI data
"""
from read_file import read_full_csv_file, read_temporal_mapping
import csv
from sklearn.mixture import GaussianMixture
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import linalg
import itertools
from sklearn import mixture

colors = {"CN": "red", "SMC": "green", "EMCI": "blue", "LMCI": "orange", "AD": "black"}
groups = {"CN": "Control", "SMC": "Self", "EMCI": "Early MCI", "LMCI": "Late MCI", "AD": "Alzheimers"}


def get_baseline_id():
    temp_map = read_temporal_mapping()
    baseline = []
    for key, val in temp_map.items():
        baseline.append(val[0]["network_id"])

    return baseline

def process_data(group=["EMCI", "AD"], only_baseline=False):
    table = read_full_csv_file(col=['DX_bl', 'MMSE', 'CDRSB', 'subject'])
    data_dict = {"CN": [], "SMC": [], "EMCI": [], "LMCI": [], "AD": []}
    baseline = get_baseline_id()

    for row in table:
        if row[0] not in group or (only_baseline and row[3] not in baseline):
            continue

        if row[1] != 'NA' and row[2] != 'NA':
            mmse = float(row[1])
            cdrsb = float(row[2])
            epsilon = 0.1
            data_dict[row[0]].append([np.random.normal(mmse, 10*epsilon),
                                      np.random.normal(cdrsb, epsilon)])

    # Converting to Numpy Array for computational ease
    return data_dict

def plot_data(data, group=["EMCI", "AD"]):
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for key, val in data.items():
        if key not in group:
            continue

        x, y = [], []
        for elem in val:
            x.append(elem[0])
            y.append(elem[1])

        ax.scatter(x, y, alpha=0.8, c=colors[key], edgecolors='none', s=30)

    plt.title("Different stages of AD")
    plt.show()


def plot_results(X, Y_, means, covariances, index, title, color_iter):
    fig = plt.figure()
    splot = fig.add_subplot(1, 1, 1)

    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=30, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    # plt.xlim(-9., 5.)
    # plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("MMSE")
    plt.ylabel("CDR")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    group = ["EMCI", "LMCI", "CN"]
    data = process_data(group, only_baseline=True)
    X = []
    labels = []
    label = 0
    for key, val in data.items():
        X = X + val
        labels = labels + [label] * len(val)
        if len(val) > 0:
            label = label + 1
    X = np.array(X)
    labels = np.array(labels)
    n_class = len(group)
    gmm = mixture.GaussianMixture(n_components=n_class, covariance_type='full').fit(X)
    plot_results(X, labels, gmm.means_, gmm.covariances_, 0,
                 'Gaussian Mixture', [colors[x] for x in group])
    #plot_data(data, group)
