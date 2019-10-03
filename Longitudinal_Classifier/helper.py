import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from Longitudinal_Classifier.read_file import *
from torch_geometric.nn import TopKPooling
from sklearn.metrics import f1_score
import pickle as pkl

def convert_to_geom(node_feat, adj_mat, label, normalize=False, threshold=0.005):
    if normalize:
        adj_mat = adj_mat + adj_mat.T
        deg = adj_mat.sum(axis=1) ** (-0.5)
        deg_norm = np.outer(deg, deg)
        adj_mat = deg_norm * adj_mat
        adj_mat[adj_mat < threshold] = 0
        # adj_mat = adj_mat + 0.5 * np.eye(len(adj_mat))
    edge_ind = np.where(adj_mat > 0)
    edge_ind = torch.tensor([edge_ind[0], edge_ind[1]], dtype=torch.long)
    # adj_mat = adj_mat + np.eye(len(adj_mat))
    edge_attr = torch.tensor(adj_mat[adj_mat > 0], dtype=torch.float).unsqueeze(1)
    # edge_attr.requires_grad = True
    # edge_attr = torch.FloatTensor(adj_mat)
    node_feat = (node_feat - node_feat.mean()) / node_feat.std()
    x = torch.tensor(node_feat, dtype=torch.float).unsqueeze(1)
    # x.requires_grad = True
    g = Data(x=x, edge_index=edge_ind, edge_attr=edge_attr, y=label)
    if Args.cuda:
        g.to(Args.device)
    return g

def convert_to_geom_all(node_feat, adj_mat, label):
    G = []
    for i in range(0, len(adj_mat)):
        G.append(convert_to_geom(node_feat[i], adj_mat[i], label[i], True))
    return G

def get_betweeness_cen(A):
    import bct
    F = []
    for a in A:
        f = bct.betweenness_wei(a)
        F.append(f)
    return F

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


from Longitudinal_Classifier.networkx_interface import *
class hubInducer:
    def __init__(self, g, hub_count):
        self.hub_count = hub_count
        self.g = convert_to_netx(g)

    def get_hubs(self):
        bt_dict = nx.betweenness_centrality(self.g, weight='weight')
        ordered_dict = sorted(bt_dict.items(), key=lambda kv: kv[0])
        node_betwn = np.array([a[1] for a in ordered_dict])
        idx = node_betwn.argsort()
        n = len(node_betwn)

        return idx[n - self.hub_count:]

    def induce(self, nodelist):
        node_set = set(nodelist)
        node_pair = [(a, b) if a < b else None for a in nodelist for b in nodelist]
        node_pair = set(node_pair)
        node_pair = list(node_pair)
        node_pair.remove(None)
        for a, b in node_pair:
            paths = nx.all_shortest_paths(self.g, source=a, target=b)

            for path in paths:
                if len(path) > 2:
                    for c in path:
                        node_set.add(c)

        return node_set

def process_mat(adj_mat, threshold):
    adj_mat = adj_mat + adj_mat.T
    deg = adj_mat.sum(axis=1) ** (-0.5)
    deg_norm = np.outer(deg, deg)
    adj_mat = deg_norm * adj_mat
    adj_mat[adj_mat < threshold] = 0
    adj_mat[adj_mat > 0] = 1 / adj_mat[adj_mat > 0]
    return adj_mat

def avg_mat(mat_list):
    sum = np.zeros(mat_list[0].shape)
    for mat in mat_list:
        sum = sum + mat
    return sum / len(mat_list)

def get_hub(data):
    hist = {k: 0 for k in range(148)}
    for d in data:
        A = avg_mat(d["adjacency_matrix"])
        # for i in range(len(data["adjacency_matrix"])):
        #     A = data["adjacency_matrix"][i]
        A = process_mat(A, 0.05)

        hi = hubInducer(A, 8)
        hubs = hi.get_hubs()
        node_set = hi.induce(hubs)
        for node in node_set:
            hist[node] = hist[node] + 1
    sorted_hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
    return np.array(sorted([node[0] for node in sorted_hist[0:20]]))

def induce_sub(data, hub_idx):
    for i in range(0, len(data["node_feature"])):
        data["node_feature"] = data["node_feature"][i][hub_idx]
        data["adjacency_matrix"][i] = data["adjacency_matrix"][i][hub_idx, :]
        data["adjacency_matrix"][i] = data["adjacency_matrix"][i][:, hub_idx]
    return data


if __name__ == '__main__':
    data = read_all_subjects(classes=[0, 1, 2], conv_to_tensor=False)
    # hub_idx = get_hub(data)

    # G = convert_to_geom_all(data["node_feature"], data["adjacency_matrix"], data["dx_label"])
    # print(len(G))
    #
    # topKpooling = TopKPooling(in_channels=1)
    # data = G[0]
    # for i in range(4):
    #     x, edge_index, edge_attr, _, _, _ = topKpooling.forward(x=data.x, edge_attr=data.edge_attr, edge_index=data.edge_index)
    #     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #
    # print(data)

