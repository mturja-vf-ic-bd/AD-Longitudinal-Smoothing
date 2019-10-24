import torch.nn as nn
from Longitudinal_Classifier.layer import GraphAttentionLayer
from torch_geometric.nn import dense_diff_pool
import torch.nn.functional as F
import torch
import math

class SimpleLinear(nn.Module):
    def __init__(self, dense_dim):
        super(SimpleLinear, self).__init__()
        self.dns_lr = [
            nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.ReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i - 1], dense_dim[i])
            for i in range(1, len(dense_dim))
        ]

        for i, l in enumerate(self.dns_lr):
            self.add_module('Dense_{}'.format(i), l)

    def forward(self, g, batch):
        x = g.x
        x = x.view(batch, -1)
        for i, l in enumerate(self.dns_lr):
            if i > 0:
                x = F.dropout(x)
            x = l(x)
        # x = torch.mean(x, dim=1)
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # stdv = .5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GAT_DiffPool(nn.Module):
    def __init__(self, in_feat, out_feat, dropout, alpha, c=6):
        super(GAT_DiffPool, self).__init__()
        self.gat = GraphAttentionLayer(in_feat, out_feat, dropout, alpha)
        self.assign_mat = GraphAttentionLayer(in_feat, c, dropout, alpha)
        # self.gat = GraphConvolution(in_feat, out_feat)
        # self.assign_mat = GraphConvolution(in_feat, c)

    def forward(self, X, A):
        Z = F.relu(self.gat(X, A))
        S = F.relu(self.assign_mat(X, A))
        X_new, A_new, link_loss, ent_loss = dense_diff_pool(Z, A, S)
        return X_new, A_new, link_loss, ent_loss

class GDNet(nn.Module):
    def __init__(self, nfeat_seq, dropout, alpha, n_class, c=[8, 4, 1]):
        super(GDNet, self).__init__()
        self.layers = [GAT_DiffPool(nfeat_seq[i-1], nfeat_seq[i], dropout, alpha, c[i-1]) for i in range(1, len(nfeat_seq))]
        for i, l in enumerate(self.layers):
            self.add_module('GATDiffPool_{}'.format(i), l)

        self.dense = nn.Linear(nfeat_seq[-1] * c[-1], n_class)
        self.add_module('Dense', self.dense)

    def forward(self, X, A):
        link_loss_tl = 0
        ent_loss_tl = 0
        for i, l in enumerate(self.layers):
            if i == 0:
                X_new, A_new, link_loss, ent_loss = l(X, A)
            else:
                X_new, A_new, link_loss, ent_loss = l(X_new, A_new)
            link_loss_tl += link_loss
            ent_loss_tl += ent_loss

        X_new = self.dense(X_new.view(X_new.size(0), 1, -1)).squeeze(1)
        return X_new, A_new, link_loss_tl, ent_loss_tl