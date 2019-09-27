import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from Longitudinal_Classifier.read_file import *
from torch_geometric.nn import TopKPooling
from sklearn.metrics import f1_score

def convert_to_geom(node_feat, adj_mat, label, normalize=False, threshold=0.005):
    if normalize:
        adj_mat = adj_mat + adj_mat.T
        deg = adj_mat.sum(axis=1) ** (-0.5)
        deg_norm = np.outer(deg, deg)
        adj_mat = deg_norm * adj_mat
        adj_mat[adj_mat < threshold] = 0
        adj_mat = adj_mat + np.eye(len(adj_mat))
    edge_ind = np.where(adj_mat > 0)
    edge_ind = torch.tensor([edge_ind[0], edge_ind[1]], dtype=torch.long)
    # adj_mat = adj_mat + np.eye(len(adj_mat))
    edge_attr = torch.tensor(adj_mat[adj_mat > 0], dtype=torch.float).unsqueeze(1)
    edge_attr.requires_grad = True
    # edge_attr = torch.FloatTensor(adj_mat)
    x = torch.tensor(node_feat, dtype=torch.float).unsqueeze(1)
    x.requires_grad = True
    g = Data(x=x, edge_index=edge_ind, edge_attr=edge_attr, y=label)
    if Args.cuda:
        g.to(Args.device)
    return g

def convert_to_geom_all(node_feat, adj_mat, label):
    G = []
    for i in range(0, len(adj_mat)):
        G.append(convert_to_geom(node_feat[i], adj_mat[i], label[i], True))
    return G

def get_train_test_fold(x, y, ratio=0.2):
    n_fold = int(1/ratio)
    train_fold = []
    test_fold = []
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    for train_index, test_index in kf.split(x, y):
        train_fold.append(train_index)
        test_fold.append(test_index)

    return train_fold, test_fold

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        # print("pred: ", pred, "\ntrue: ", target)
        return (pred == target).sum() * 100.0 / len(target), f1_score(target.cpu(), pred.cpu(), average='micro')


def getFiedlerVector(adj_mat, threshold=1e-10):
    """
    Finds the fiedler vector for a graph with adj_mat as the adjacency matrix
    :param adj_mat: numpy array with shape: (n, n)
    :param threshold: the eigen values must exceed this threshold to be considered as nonzero
    :return: fiedler vector with shape: (n, 1)
    """
    assert (adj_mat == adj_mat.T).all(), "Adjacency matrix has to be symmetric"
    deg_mat = np.diag(adj_mat.sum(axis=1))
    L = deg_mat - adj_mat
    eigval, eigvec = np.linalg.eigh(L)
    fiedl_val = eigval[eigval > threshold][0]
    fiedl_ind = np.where(eigval == fiedl_val)[0]
    return eigvec[:, fiedl_ind].mean(axis=1, keepdims=True)  # Average of all the fiedler vector


def getFiedlerFeature(batch_adj):
    """
    Computes fiedler vector for each graph in the batch
    :param batch_adj: shape: (b, n, n) where b is the batch size and n is the number of nodes
    :return: fiedler feature vector for each graph in the batch with shape (b, n ,1)
    """

    batch_size, n_nodes, _ = batch_adj.shape
    fiedlerFeature = np.zeros((batch_size, n_nodes, 1))
    for i in range(batch_size):
        fiedlerFeature[i] = getFiedlerVector(batch_adj[i])

    return fiedlerFeature


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

