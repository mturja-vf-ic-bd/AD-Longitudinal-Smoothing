from read_file import read_all_subjects, get_baselines
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

        t_score = np.zeros(feat_g1.shape[1])
        for i in range(feat_g1.shape[1]):
            a = feat_g1[:, i]
            b = feat_g2[:, i]
            t_score[i] = ttest_ind(a, b)[0]

        return t_score


if __name__ == '__main__':
    data_set = get_baselines()
    tt = ttest(data_set)
    print(tt.find_t_stat_node())