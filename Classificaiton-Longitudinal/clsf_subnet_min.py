import numpy as np
from read_file import get_baselines
from dang_paper import disc_network_clsf
import torch

class network_mining:
    def __init__(self, data, data_type):
        if data_type == 'baseline-thickness':
            self.F = np.array([x[:, 2] for x in data["node_feature"]])  # 148 dim
            self.A = np.array(data["adjacency_matrix"])  # 148*148 dim
            self.y = np.array([int(x) - 1 for x in data["dx_label"]])


if __name__ == '__main__':
    # nm = network_mining(get_baselines(), 'baseline-thickness')
    # net = nm.A.mean(axis=0)
    # dang_clsf = disc_network_clsf(nm.F, net, nm.y)
    # print(net)
    device = torch.device("cuda:0")
    print(device)
