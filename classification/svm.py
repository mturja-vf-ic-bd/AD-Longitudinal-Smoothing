import collections
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from utils.helper import *
from Evaluation.modularity_analysis import voi_between_community_structure as NVI
from classification.graph_kernel_svm import smote
from bct import eigenvector_centrality_und
from sklearn.metrics import confusion_matrix, accuracy_score
from plot_functions import plot_roc

from utils.readFile import get_subject_info, readSubjectFiles


def read_data(size=3, sel_label=["1", "2"]):
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
        else:
            if sub_info[sub][0]["DX"] == '1':
                count = 2
            elif sub_info[sub][0]["DX"] == '2':
                count = 1
            elif sub_info[sub][0]["DX"] == '3':
                count = 3

        for c in range(count):
            X1.append(rw_all[c])
            y.append(sub_info[sub][c]["DX"])
            X2.append(sm_all[c])

    y = [0 if a == sel_label[0] else 1 for a in y]
    print(collections.Counter(y))
    return X1, X2, y

def extract_features(g, method='None'):
    """
    Extract features from each node of graph g.
    :param g: adjacency matrix of g
    :param method: specifies what features to extract
    :return: feature for each node
    """

    feat = list(clustering_coef_wu(g))
    feat = feat + list(eigenvector_centrality_und(g))
    feat = feat + list(g.sum(axis=0))
    feat.append(assortativity_wei(g))

    return feat


def compute_kernel_matrix(samples):
    n = len(samples)
    kernel_mat = np.zeros((n, n))

    for i in range(0, n):
        for j in range(i + 1, n):
            kernel_mat[i, j] = NVI(samples[i], samples[j])

    kernel_mat = kernel_mat + kernel_mat.T
    return kernel_mat


def get_train_test(X, y, f=0.2, random_state=1):
    X = [extract_features(x) for x in X]
    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=f, random_state=random_state)
    return X_train, X_test, y_train, y_test


def plot_2d_data(X, y, fig):
    X = np.array(X)
    y = np.array(y)
    for label in np.unique(y):
        ind = [idx for idx in range(len(y)) if y[idx]==label]
        x1 = X[ind[0:1], 0]
        x2 = X[ind[0:1], 1]
        n = len(x1)
        fig.scatter(x1, x2, c=[int(label)*10] * n, cmap='viridis')

    return fig


def classification_on_global_features(X_train, X_test, y_train, y_test, prob=False):
    clf = svm.SVC(random_state=1, kernel='rbf', probability=True, decision_function_shape='ovr', gamma='auto')
    clf.fit(X_train, y_train)
    print("Kernel fitted")
    pred = clf.predict_proba(X_test)
    y_pred = []
    for p in pred:
        if p[0] >= 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)

    y_pred = np.array(y_pred)
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    return y_pred, y_test, pred

def classify(X, y, f=0.3, random_state=1):
    X_train, X_test, y_train, y_test = get_train_test(X, y, f=f, random_state=random_state)
    X_train, y_train = smote(X_train, y_train)
    return classification_on_global_features(X_train, X_test, y_train, y_test, prob)


if __name__ == '__main__':
    sel = [['1', '2'], ['1', '3'], ['2', '3']]
    for sel_label in sel:
        X_rw, X_sm, y = read_data(3, sel_label)
        X_rw_th = threshold_all(X_rw, Args.threshold, 1)

        rw_ac = []
        rw_th = []
        sm_ac = []
        prob = True
        if prob:
            iter = 10
            pred = []
            y_pred, y_test, pred1 = classify(X_rw_th, y, random_state=iter)

            #plot_roc(pred, y_test, title="Thresholded_ROC_" + sel_label[0] + "_" + sel_label[1])
            print("Raw th: ", accuracy_score(y_test, y_pred))

            y_pred, y_test, pred2 = classify(X_rw, y, random_state=iter)
            pred.append(pred2)
            pred.append(pred1)

            #plot_roc(pred, y_test, title="Raw_ROC_" + sel_label[0] + "_" + sel_label[1])
            print("Raw: ", accuracy_score(y_test, y_pred))

            y_pred, y_test, pred3 = classify(X_sm, y, random_state=iter)
            pred.append(pred3)
            plot_roc(pred, y_test, title=["Raw", "Thresholded", "Intrinsic"], fname=sel_label[0]+"_"+sel_label[1])
            print("Smooth: ", accuracy_score(y_test, y_pred))
        else:
            for iter in range(20):
                print("Iter: ", iter)
                y_pred, y_test, pred = classify(X_rw, y, random_state=iter)
                rc = accuracy_score(y_test, y_pred)
                rw_ac.append(rc)
                print("rw: ", rc)

                y_pred, y_test, pred = classify(X_rw_th, y, random_state=iter)
                rc_th = accuracy_score(y_test, y_pred)
                rw_th.append(rc_th)
                print("rw_th: ", rc_th)

                y_pred, y_test, pred = classify(X_sm, y, random_state=iter)
                sm = accuracy_score(y_test, y_pred)
                sm_ac.append(sm)
                print("sm: ", sm)

            print("Mean raw: ", np.mean(rw_ac))
            print("Mean raw thr: ", np.mean(rw_th))
            print("Mean smooth: ", np.mean(sm_ac))

            print("Median itr: ", np.argsort(sm_ac)[len(sm_ac)//2],
                  "\nMedian: ", np.median(sm_ac))
