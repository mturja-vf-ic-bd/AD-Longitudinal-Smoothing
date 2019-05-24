from read_file import read_all_subjects, get_baselines, get_region_names
import numpy as np
from scipy.stats import ttest_ind


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
                p_val[i, j] = ttest_ind(a, b)[1]

        return p_val


if __name__ == '__main__':
    data_set = get_baselines()
    groups = [["1", "2"], ["2", "3"], ["1", "3"]]
    count = 40

    tt = ttest(data_set, group=["1", "3"])
    reg_names = np.array(get_region_names())
    p_val_edge = tt.find_t_stat_edge() < 0.05
    p_val_node = tt.find_t_stat_node() < 0.01
    reg_pair = []
    for i in range(p_val_edge.shape[0]):
        for j in range(p_val_edge.shape[1]):
            if p_val_node[i] and p_val_node[j] and i != j:
                reg_pair.append((reg_names[i], reg_names[j]))

    print(reg_pair)

    # reg_names = np.array(get_region_names())
    # reg_list = []
    # for group in groups:
    #     tt = ttest(data_set, group=group)
    #     p_val = abs(tt.find_t_stat_node())
    #     ind = np.argsort(p_val)
    #     reg = set(reg_names[ind[:count]])
    #     reg_list.append(reg)
    #
    # for i in range(len(reg_list) - 1):
    #     intersect = reg_list[i].intersection(reg_list[i + 1])
    #     print("Intersection between {} and {}:\n{}".format(i, i + 1, intersect))
    #     print("Intersection length: {}".format(len(intersect)))