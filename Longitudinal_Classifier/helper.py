import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from Longitudinal_Classifier.read_file import *
from torch_geometric.nn import TopKPooling
from sklearn.metrics import f1_score
import pickle as pkl

def normalize_net(adj_mat, threshold=0.005, self_loop=True, sym=True):
    if sym:
        adj_mat = adj_mat + adj_mat.T
    deg = adj_mat.sum(axis=1) ** (-0.5)
    deg2 = adj_mat.sum(axis=0) ** (-0.5)
    deg_norm = np.outer(deg, deg2)

    adj_mat = deg_norm * adj_mat
    adj_mat[adj_mat < threshold] = 0
    if self_loop:
        adj_mat = adj_mat + 0.5 * np.eye(len(adj_mat))
    return adj_mat

def normalize_feat(x):
    dim = 0 if x.dim() == 2 else 1
    x = (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(x, dim=dim, keepdim=True)
    return x

def convert_to_geom(node_feat, adj_mat, label, normalize=False, threshold=0.005):
    if normalize:
        adj_mat = normalize_net(adj_mat, threshold)
    edge_ind = np.where(adj_mat > 0)
    edge_ind = torch.tensor([edge_ind[0], edge_ind[1]], dtype=torch.long)
    # adj_mat = adj_mat + np.eye(len(adj_mat))
    edge_attr = torch.tensor(adj_mat[adj_mat > 0], dtype=torch.float)
    # edge_attr.requires_grad = True
    # edge_attr = torch.FloatTensor(adj_mat)
    # node_feat = (node_feat - node_feat.mean()) / node_feat.std()
    x = torch.tensor(node_feat, dtype=torch.float).view(1, -1)
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


def getFiedlerVector(adj_mat, c=10):
    """
    Finds the fiedler vector for a graph with adj_mat as the adjacency matrix
    :param adj_mat: numpy array with shape: (n, n)
    :param threshold: the eigen values must exceed this threshold to be considered as nonzero
    :return: fiedler vector with shape: (n, 1)
    """
    # assert (adj_mat == adj_mat.T).all(), "Adjacency matrix has to be symmetric"
    deg_mat = np.diag(adj_mat.sum(axis=1))
    L = deg_mat - adj_mat
    eigval, eigvec = np.linalg.eigh(L)
    sort_idx = np.argsort(eigval)
    eigvec = eigvec[:, sort_idx]
    eigvec = eigvec - np.mean(eigvec, axis=0, keepdims=True)
    eigvec = eigvec / np.std(eigvec, axis=0, keepdims=True)
    return eigvec[:, -c:]


def getFiedlerFeature(batch_adj, feat_dim=10):
    """
    Computes fiedler vector for each graph in the batch
    :param batch_adj: shape: (b, n, n) where b is the batch size and n is the number of nodes
    :return: fiedler feature vector for each graph in the batch with shape (b, n ,1)
    """

    batch_size, n_nodes, _ = batch_adj.shape
    fiedlerFeature = np.zeros((batch_size, n_nodes, feat_dim))
    for i in range(batch_size):
        fiedlerFeature[i] = getFiedlerVector(batch_adj[i], feat_dim)

    return fiedlerFeature


def get_aggr_net(data, reduce='median'):
    for j, d in enumerate(data):
        if j == 0:
            M = np.stack(d["adjacency_matrix"], axis=0)
        else:
            temp = np.stack(d["adjacency_matrix"], axis=0)
            M = np.concatenate((M, temp), axis=0)

    if reduce == 'median':
        return np.percentile(M, q=0.5, axis=0)


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
        data["node_feature"][i] = data["node_feature"][i][:, hub_idx, :]
        data["adjacency_matrix"][i] = data["adjacency_matrix"][i][hub_idx, :]
        data["adjacency_matrix"][i] = data["adjacency_matrix"][i][:, hub_idx]
    return data


def update_parc_table():
    import csv
    label_dict = {"CN" : "1", "SMC" : "1", "EMCI" : "2",  "LMCI": "3", "AD" : "4"}
    dx_dict = {}
    with open(Args.ORIG_DATA) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                dx_dict[row[2]] = label_dict[row[6]]

    import json
    with open(Args.SUB_TO_NET_MAP) as json_file:
        data = json.load(json_file)
        for key, val in data.items():
            for scan in val:
                scan["dx_data"] = dx_dict[scan["network_id"]]

    with open(Args.OTHER_DIR + "/temporal_mapping_baselabel.json", "w") as f:
        json.dump(data, f)


def get_crossectional(data, label=[0,1,2,3]):
    X = []
    Y = []
    for i in range(len(data)):
        # print("{}: {}, {}".format(data[i]['subject'], len(data[i]['node_feature']),
        #                           len(data[i]['dx_label'])))
        for j in range(len(data[i]['node_feature'])):
            if data[i]['dx_label'][j] in label:
                X.append(data[i]['node_feature'][j])
                Y.append(data[i]['dx_label'][j])

    return X, Y


def CMN(cort_thk):
    # cort_thk N*C numpy array for N subjects
    cmn = np.dot(cort_thk, cort_thk.T)
    cmn[cmn < 0] = 0
    np.fill_diagonal(cmn, 0)
    cmn = normalize_net(cmn, 0, False, sym=False)
    cmn = topKthreshold(cmn, 2500)
    return cmn

def topKthreshold(connectome, K=1):
    connectome = np.array(connectome)
    row, col = connectome.shape
    idx = np.argsort(connectome, axis=None)[0:row*col-K][::-1]
    idx_row = idx // col
    idx_col = idx % col
    connectome[idx_row, idx_col] = 0
    return connectome

def read_net_cmn():
    cmn_net = []
    for i in range(4):
        with open('cmn_'+str(i), 'r') as f:
            cmn_net.append(torch.FloatTensor(np.loadtxt(f)))
    return cmn_net


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # update_parc_table()
    data, count = read_all_subjects(classes=[0, 1, 2, 3], conv_to_tensor=False)

    X, Y = get_crossectional(data, [0, 1, 2, 3])
    X = np.array(X)
    std_ = X.std(axis=0, keepdims=True)
    mn = X.mean(axis=0, keepdims=True)

    for i in range(4):
        X, Y = get_crossectional(data, [i])
        X = np.array(X)
        X = (X - mn) / std_

        fig, ax = plt.subplots()
        cmn = CMN(X.T)
        with open('cmn_'+str(i), 'w') as f:
            np.savetxt(f, cmn)

        from utils.sortDetriuxNodes import sort_matrix
        cmn, _ = sort_matrix(cmn)
        im = ax.imshow(cmn)
        fig.tight_layout()
        plt.savefig('CMN_'+str(i))
        # plt.show()
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

