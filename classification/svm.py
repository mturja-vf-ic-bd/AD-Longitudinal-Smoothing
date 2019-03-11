from bct import *
from sklearn import svm
from utils.readFile import get_subject_info, readSubjectFiles
from sklearn.model_selection import train_test_split
from args import Args
from scipy.cluster.vq import vq, kmeans, whiten
from utils.helper import *
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score


def read_data():
    # Reading one scan( both raw and smooth) per subject and their DX label
    sub_info = get_subject_info(3)
    X1 = []  # raw
    X2 = []  # smooth
    y = []
    for sub in sub_info.keys():
        rw_all, sm_all = readSubjectFiles(sub)
        X1.append(extract_features(rw_all[0]))
        X2.append(extract_features((sm_all[0])))
        y.append(sub_info[sub][0]["DX"])

    return X1, X2, y

def extract_features(g, method='None'):
    """
    Extract features from each node of graph g.
    :param g: adjacency matrix of g
    :param method: specifies what features to extract
    :return: feature for each node
    """

    if method == 'cc':
        return clustering_coef_wu(g)
    elif method == 'deg':
        return np.sum(g, axis=1)
    elif method == 'bc':
        bc = module_degree_zscore(g)
        return bc
    elif method == 'pca':
        eig, _ = np.linalg.eigh(g)
        return eig[len(eig)-4:len(eig) - 2]
    elif method == 'None':
        return g
    elif method == 'flatten':
        return g.flatten()


def get_train_test(X, y, f=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=f, random_state=10)
    sm = SMOTE(random_state=10)
    X_train, y_train = sm.fit_sample(X_train, y_train)
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


if __name__ == '__main__':
    X_rw, X_sm, y = read_data()
    print(len(X_rw))

    X_rw_feat = []
    X_sm_feat = []
    for a in range(len(X_rw)):
        X_rw_feat.append(extract_features(X_rw[a].reshape(148, 148), method='pca'))
        X_sm_feat.append(extract_features(X_sm[a].reshape(148, 148), method='pca'))

    X_rw_train, X_rw_test, y_train, y_test = get_train_test(X_rw_feat, y)
    clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
    scores = cross_val_score(clf, X_rw_train, y_train, cv=5)
    print("Mean:", scores)
    clf.fit(X_rw_train, y_train)
    pred_rw = clf.predict(X_rw_test)
    print(sum((pred_rw == y_test)) / len(y_test))

    X_sm_train, X_sm_test, y_train, y_test = get_train_test(X_sm_feat, y)
    clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
    clf.fit(X_sm_train, y_train)
    scores = cross_val_score(clf, X_sm_train, y_train, cv=5)
    print("Mean:", scores)
    pred_sm = clf.predict(np.nan_to_num(X_sm_test))
    print(sum((pred_sm == y_test)) / len(y_test))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot_2d_data(X_rw_train, y_train, ax1)
    plot_2d_data(X_sm_train, y_train, ax2)

    plt.show()

    center_rw, d1 = kmeans(X_rw_feat, k_or_guess=3)
    center_sw, d2 = kmeans(X_sm_feat, k_or_guess=3)
    print(d1)
    print(d2)
    print(center_rw[0:2, 0:5])
    print(center_sw[0:2, 0:5])



