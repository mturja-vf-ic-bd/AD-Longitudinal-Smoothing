from Longitudinal_Classifier.cortical_change import cortical_change, read_all_subjects, get_aggr_net, normalize_net
import numpy as np
from matplotlib import pyplot as plt


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res


class GraphSpectrum:
    def __init__(self, A=None):
        if A is not None:
            self.A = A

    def normalize_mat(self, A=None):
        if A is None:
            A = self.A
        D = (A.sum(axis=1)).reshape((1, -1))
        # A = D^(-1/2) A D^(-1/2)
        D = D ** (-0.5)
        D_inv = np.dot(D.T, D)
        A_tld = np.multiply(D_inv, A)
        return A_tld

    def normalized_laplacian(self, A=None):
        if A is None:
            A = self.A
        D = (A.sum(axis=1)).reshape((1, -1))
        # A = D^(-1/2) A D^(-1/2)
        D = D ** (-0.5)
        D_inv = np.dot(D.T, D)
        A_tld = np.multiply(D_inv, A)
        I = np.eye(len(A))
        L = I - A_tld
        return L

    def get_fourier_basis(self, A=None, normalize=True):
        if A is None:
            A = self.A
        # A should be symmetric numpy array
        if normalize:
            L = self.normalized_laplacian(A)
            ld, U = np.linalg.eigh(L)
        else:
            D = (A.sum(axis=1)).reshape((1, -1))
            I = np.eye(A.shape[0])
            np.fill_diagonal(I, D)
            ld, U = np.linalg.eigh(I - A)

        return ld, U

    def rank_reduce(self, A=None, k=10, k2=0):
        if A is None:
            A = self.A
        D = np.diag(A.sum(axis=1))
        ld, U = self.get_fourier_basis(A, False)
        L = np.dot(np.dot(U[:,k2:k], np.diag(ld[k2:k])), U[:,k2:k].T)
        return D - L

    def transform_to_spectrum(self, f):
        ld, U = self.get_fourier_basis()
        return ld, np.dot(U.T, f)

    def transform_to_vertex(self, f_tld):
        U = self.get_fourier_basis()
        return np.dot(U, f_tld)

    def spectral_cluster(self, A=None, k=6, one_hot=True):
        from sklearn.cluster import SpectralClustering
        if A is None:
            A = self.A

        clustering = SpectralClustering(n_clusters=k, assign_labels='discretize',
                                        random_state=0, affinity='precomputed', n_jobs=-1)
        res = clustering.fit_predict(A)
        if one_hot:
            res = get_one_hot(res, k)
        return res

    # def hierarchical_cluster(self, A=None, hier_num=[50, 12]):
    def get_wavelet(self, A=None, k=6):
        if A is None:
            A = self.A

        ld_list = []
        U_list = []
        idx_list = []
        S = self.spectral_cluster(A, k, one_hot=False)
        for i in range(k):
            idx = np.where(S == i)[0]
            ld, U = self.get_fourier_basis(A[idx, :][:, idx])
            ld_list.append(ld)
            U_list.append(U)
            idx_list.append(idx)

        return ld_list, U_list, idx_list


if __name__ == '__main__':
    data, count = read_all_subjects(classes=[0, 3], conv_to_tensor=False)
    net = get_aggr_net(data, label=[0])
    net = normalize_net(net)
    gs = GraphSpectrum(net)
    ld_all, U_all, idx = gs.get_wavelet(k=12)
    sub_names = ["007_S_4911", "005_S_5038", "007_S_2394"]
    cc = cortical_change()
    cc.read_data(sub_names)
    cc.process_data()
    cc.compute_node_slope()

    for sub in sub_names:
        gs = GraphSpectrum(cc.net[sub])
        f = cc.node_feat[sub]

        # m = cc.m[sub]
        # ld, f_tld = gs.transform_to_spectrum(f)
        # _, m_tld = gs.transform_to_spectrum(m)
        #
        # plt.subplot(2, 1, 1)
        # for i in range(f_tld.shape[1]):
        #     plt.plot(ld, f_tld[:, i])
        # plt.subplot(2, 1, 2)
        # plt.xlim(0.05e7, 0.45e7)
        # plt.ylim(-0.11, 0.15)
        # plt.plot(ld, m_tld)
        #
        # plt.show()
