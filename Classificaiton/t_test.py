from read_file import read_all_subjects, get_baselines, get_region_names
import numpy as np
from scipy.stats import ttest_ind
import pickle
from result_analysis import utils
from sklearn import preprocessing


class ttest:
    def __init__(self, data, group=["1", "3"]):
        self.group = group
        self.process_data(data, group)

    def process_data(self, data, group=["1", "3"]):
        if type(data) is dict:
            self.network = []
            self.feature = []
            self.dx = []

            for i in range(len(data["dx_label"])):
                if data["dx_label"][i] in group:
                    self.network.append(data["adjacency_matrix"][i])
                    self.feature.append(data["node_feature"][i][:, 2])
                    self.dx.append(data["dx_label"][i])

    def find_t_stat_node(self):
        feat_g1 = []
        feat_g2 = []
        for i in range(len(self.feature)):
            if self.dx[i] == self.group[0]:
                feat_g1.append(self.feature[i])
            elif self.dx[i] == self.group[1]:
                feat_g2.append(self.feature[i])

        feat_g1 = np.array(feat_g1)
        feat_g2 = np.array(feat_g2)

        p_val = np.zeros(feat_g1.shape[1])
        for i in range(feat_g1.shape[1]):
            a = feat_g1[:, i]
            b = feat_g2[:, i]
            p_val[i] = ttest_ind(a, b)[1]

        return p_val

    def find_t_stat_edge(self):
        feat_g1 = []
        feat_g2 = []
        for i in range(len(self.network)):
            if self.dx[i] == self.group[0]:
                feat_g1.append(self.network[i])
            elif self.dx[i] == self.group[1]:
                feat_g2.append(self.network[i])

        feat_g1 = np.array(feat_g1)
        feat_g2 = np.array(feat_g2)

        p_val = np.zeros(feat_g1.shape[1:])
        for i in range(feat_g1.shape[1]):
            for j in range(feat_g1.shape[2]):
                a = feat_g1[:, i, j]
                b = feat_g2[:, i, j]
                if i != j:
                    p_val[i, j] = ttest_ind(a, b)[1]
                else:
                    p_val[i, j] = 0

        return p_val

    def find_discriminative_pairs(self, p_cut_off=0.05):
        p_val_edge = self.find_t_stat_edge() < p_cut_off
        p_val_node = self.find_t_stat_node() < p_cut_off
        print("Node selected: {}".format(p_val_node.sum()))
        reg_pair = []
        for i in range(p_val_edge.shape[0]):
            for j in range(p_val_edge.shape[1]):
                if p_val_edge[i][j] and p_val_node[i] and p_val_node[j] and i != j:
                    reg_pair.append((i, j))

        return reg_pair

    def find_discriminative_links(self, p_cut_off=0.05):
        p_val_edge = self.find_t_stat_edge() < p_cut_off
        link_pair = []
        for i in range(p_val_edge.shape[0]):
            for j in range(p_val_edge.shape[1]):
                if p_val_edge[i][j] and i != j:
                    link_pair.append((i, j))

        return link_pair

    def get_triplet_data(self, p_cut_off=0.05):
        pairs = self.find_discriminative_pairs(p_cut_off)
        X = []
        label = []
        for i in range(len(self.dx)):
            network = self.network
            row = []
            feat = self.feature
            for (x, y) in pairs:
              row = row + [feat[i][x], feat[i][y], network[i][x, y]]
            X.append(row)
            if self.dx[i] == self.group[0]:
                label.append(0)
            else:
                label.append(1)

        X = np.array(X)
        X = X - X.mean(axis=0)
        X = X / np.std(X, axis=0)
        label = np.array(label)
        return X, label, pairs

    def get_link_data(self, p_cut_off=0.05, pairs=None):
        if pairs is None:
            pairs = self.find_discriminative_links(p_cut_off)
        X = []
        label = []
        for i in range(len(self.dx)):
            network = self.network
            row = []
            for (x, y) in pairs:
                row.append(network[i][x, y])
            X.append(row)
            if self.dx[i] == self.group[0]:
                label.append(0)
            else:
                label.append(1)

        X = np.array(X)
        X = preprocessing.robust_scale(X)
        label = np.array(label)
        return X, label, pairs



if __name__ == '__main__':
    data_set = get_baselines()
    groups = [["1", "2"], ["1", "3"], ["2", "3"]]
    reg_names = np.array(get_region_names())
    reg_list = []
    lasso_edges= get_lasso_edge_set(groups)
    for i, group in enumerate(groups):
        tt = ttest(data_set, group=group)
        reg = tt.find_discriminative_pairs(0.008)
        file_p = 't_pairs_' + group[0] + '_' + group[1] + '.pkl'

        common = set(reg).intersection(set(lasso_edges[i]))
        print("t-pairs: ", len(reg))
        print("lasso: ", len(lasso_edges[i]))
        print("t and lasso: ", len(common))
    # for i in range(len(reg_list) - 1):
    #     intersect = reg_list[i].intersection(reg_list[i + 1])
    #     print("Intersection between {} and {}:\n{}".format(i, i + 1, intersect))
    #     print("Intersection length: {}".format(len(intersect)))