import torch
from torch_geometric.data import Data
from Longitudinal_Classifier.read_file import *
from torch_geometric.nn import TopKPooling

def convert_to_geom(node_feat, adj_mat, label, normalize=False):
    if normalize:
        adj_mat = adj_mat + adj_mat.T
        deg = adj_mat.sum(axis=1) ** (-0.5)
        deg_norm = np.outer(deg, deg)
        adj_mat = deg_norm * adj_mat
    edge_ind = np.where(adj_mat > 0)
    edge_ind = torch.tensor([edge_ind[0], edge_ind[1]], dtype=torch.long)
    # adj_mat = adj_mat + np.eye(len(adj_mat))
    edge_attr = torch.FloatTensor(adj_mat[adj_mat > 0]).unsqueeze(1)
    # edge_attr = torch.FloatTensor(adj_mat)
    x = torch.FloatTensor(node_feat).unsqueeze(1)
    g = Data(x=x, edge_index=edge_ind, edge_attr=edge_attr, y=label)
    return g

def convert_to_geom_all(node_feat, adj_mat, label):
    G = []
    for i in range(0, len(adj_mat)):
        G.append(convert_to_geom(node_feat[i], adj_mat[i], label[i], True))
    return G


if __name__ == '__main__':
    data = read_subject_data('005_S_5038', conv_to_tensor=True)
    G = convert_to_geom_all(data["node_feature"], data["adjacency_matrix"], data["dx_label"])
    print(len(G))

    topKpooling = TopKPooling(in_channels=1)
    data = G[0]
    for i in range(4):
        x, edge_index, edge_attr, _, _, _ = topKpooling.forward(x=data.x, edge_attr=data.edge_attr, edge_index=data.edge_index)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print(data)

