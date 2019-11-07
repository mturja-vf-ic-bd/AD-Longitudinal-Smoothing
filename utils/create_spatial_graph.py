import json
from args import Args
import os
import torch
from utils.sortDetriuxNodes import sort_matrix
import numpy as np

class CorticoSpatialGraph:
    def __init__(self, filename):
        self.filename = os.path.join(Args.sub_parc_file, filename)

    def read_coord(self):
        with open(self.filename, 'r') as f:
            parc_tb = json.load(f)
        coord_list = []
        for item in parc_tb:
            coord_list.append(item['coord'])

        return coord_list

    def compute_graph(self):
        coord_list = self.read_coord()
        N = len(coord_list)
        coord_list = torch.FloatTensor(coord_list)

        x_i = torch.sum(coord_list ** 2, dim=1)
        adj = x_i.unsqueeze(1).repeat(1, N) + x_i.unsqueeze(0).repeat(N, 1) - \
              2 * torch.matmul(coord_list, torch.transpose(coord_list, 0, 1))
        adj = adj + torch.eye(N)
        adj[adj < 0] = 0
        adj = torch.sqrt(1 / adj * (torch.ones(N, N) - torch.eye(N, N)))
        deg = torch.sum(adj, dim=1, keepdim=True) ** (-0.5)
        norm = torch.matmul(deg.view(N, 1), deg.view(1, N))
        adj = adj * norm
        return adj


if __name__ == '__main__':
    csg = CorticoSpatialGraph('S146119_parcellationTable.json')
    adj = csg.compute_graph().numpy()
    np.savetxt(Args.base_dir + '/cortico_spatial_graph.txt', adj)

    from matplotlib import pyplot as plt
    plt.imshow(sort_matrix(adj)[0], vmin=0.005, vmax=0.03)
    plt.show()


