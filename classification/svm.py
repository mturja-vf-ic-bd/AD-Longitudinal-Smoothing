from bct import *
from sklearn import svm
from utils.readFile import get_subject_info, readSubjectFiles
from sklearn.model_selection import train_test_split
from args import Args
from scipy.cluster.vq import vq, kmeans, whiten


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

def extract_features(g, method='pca'):
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
        return eig[len(eig)-4: len(eig)]


if __name__ == '__main__':
    X_rw, X_sm, y = read_data()

    '''
    X_rw_train, X_rw_test, y_train, y_test = train_test_split(X_rw, y, test_size=0.20, random_state=10)
    clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
    clf.fit(X_rw_train, y_train)
    pred = clf.predict(X_rw_test)
    print(sum((pred == y_test)) / len(y_test))

    X_sm_train, X_sm_test, y_train, y_test = train_test_split(X_sm, y, test_size=0.20, random_state=10)
    clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
    clf.fit(np.nan_to_num(X_sm_train), y_train)
    pred = clf.predict(np.nan_to_num(X_sm_test))
    print(sum((pred == y_test)) / len(y_test))
    '''

    print(kmeans(X_rw, k_or_guess=3))
