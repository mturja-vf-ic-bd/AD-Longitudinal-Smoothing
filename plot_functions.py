import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib
from sklearn import metrics


def plot_matrix_all(connectomes, fname="matplt", vmin=0, vmax=0.25, savefig=True):
    T = len(connectomes)
    fig = plt.figure(figsize=(6*T, 5))
    plt.rc('font', family='Times New Roman')

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, T),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
    t = 0
    for ax in grid:
        ax.set_axis_off()
        im = ax.matshow(connectomes[t], vmin=vmin, vmax=vmax, cmap='plasma')
        t = t + 1

        cbar = grid.cbar_axes[0].colorbar(im)

        cbar.ax.set_yticks(np.arange(vmin, vmax * 1.1, (vmax - vmin) / 4))
        cbar.ax.tick_params(labelsize=30)
        cbar.ax.set_yticklabels(
            [str(vmin), str(round((vmax - vmin) / 4, 2)), str(round((vmax - vmin) / 2, 2)),
             str(round((vmax - vmin) * 3 / 4, 2)), str(vmax)])

    if savefig:
        #matplotlib.use('Agg')
        fig.tight_layout()
        fig.savefig(fname + ".png")
    else:
        plt.show()


# Plot spectrum of raw and smooth data
def plot_eigen_specturm(rw_spectrum, sm_spectrum, savefig=False):
    T = len(rw_spectrum)
    assert T == len(sm_spectrum), "raw and smooth list size doesn't match"
    fig, axes = plt.subplots(nrows=1, ncols=T, figsize=(5*T, 5))
    t = 0
    for ax in axes.flat:
        ax.plot(rw_spectrum[t], color='r')
        ax.plot(sm_spectrum[t], color='b')
        ax.set_title("t" + str(t))
        ax.set_ylim(np.min(rw_spectrum), np.max(rw_spectrum))
        t = t + 1

    if savefig:
        matplotlib.use('Agg')
        fig.savefig("spectrum_plot.png")
    else:
        plt.show()

def plot_community_structure_variation(rw, sm):
    plt.plot(rw, c='r')
    plt.plot(sm, c='b')
    plt.show()

def plot_communities(X, y):
    X = np.array(X)
    y = np.array(y)
    color = ['b', 'g', 'r']
    label_set = set(y)
    for label in label_set:
        idx = np.nonzero(y == label)
        plt.scatter(X[idx, 0], X[idx, 1], c=color[int(label)-1], alpha=0.3)
    #plt.xlim(0, 0.05)
    #plt.ylim(5, 12)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    class_labels = sorted(classes.keys())
    class_names = [classes[k] for k in class_labels]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    matplotlib.use("Agg")
    fig.savefig(title + ".png")
    #plt.show()
    return ax

def plot_roc(probs, y_test, title, fname):
    import pylab
    pylab.figure(0).clf()
    pylab.figure(figsize=(20, 12))
    pylab.axis('off')
    pylab.rc('font', family='Times New Roman')
    font = {'weight': 'bold',
            'size': 50}

    pylab.rc('font', **font)

    for i, prob in enumerate(probs):
        pred = prob[:, 1]
        fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
        auc = round(metrics.roc_auc_score(y_test, pred), 2)

        label = 'AUC: ' + str(auc)
        width = 15
        if i == 0:
            pylab.plot(fpr, tpr, 'r-', label=label, markersize=30, linewidth=width)
        elif i == 1:
            pylab.plot(fpr, tpr, 'o--', color=[1, 0.7, 0], markersize=30, label=label, linewidth=width)
        else:
            pylab.plot(fpr, tpr, 'bo-', markersize=30, label=label, linewidth=width)

        pylab.legend(loc='lower right', prop={'size': 60})
        pylab.tight_layout()


    #matplotlib.use("Agg")
    pylab.tight_layout()
    pylab.savefig(fname + ".png")

def plot_hub_eval(rw_hub_match, sm_hub_match):
    plt.bar(np.arange(0, 148, 1), rw_hub_match)
    plt.ylim(0, 2)
    plt.show()
    plt.bar(np.arange(0, 148, 1), sm_hub_match)
    plt.ylim(0, 2)
    plt.show()
