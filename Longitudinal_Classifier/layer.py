import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        N = input.size(2)
        t = input.size(1)
        b = input.size(0)
        h = torch.matmul(input, self.W)
        a_input = torch.cat([h.repeat(1, 1, N, 1).view(b, t, N * N, -1), h.repeat(1, 1, N, 1)], dim=1).view(b, t, N, -1,
                                                                                          2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(4))
        attention = torch.mul(adj, e)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = F.softmax(attention, dim=3)

        h_prime = torch.matmul(attention, h)
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class WGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(WGATConv, self).__init__(in_channels, out_channels, heads, concat,
                 negative_slope, dropout, bias, **kwargs)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     device=edge_index.device)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, size_i, x_i, x_j):
        print("In WGAT: {}, {}, {}, {}, {}".format(edge_index_i.shape, edge_weight_i.shape, x_i.shape, x_j.shape))


