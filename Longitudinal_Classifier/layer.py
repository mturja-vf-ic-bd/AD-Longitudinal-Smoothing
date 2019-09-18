import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.data import Data


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
    def __init__(self, in_channels, out_channels, heads=1, concat=False,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(WGATConv, self).__init__(in_channels, out_channels, heads, concat,
                 negative_slope, dropout, bias, **kwargs)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        # if size is None and torch.is_tensor(x):
        #     # edge_index, _ = remove_self_loops(edge_index)
        #     # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     device=edge_index.device)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, size_i, x_i, x_j, edge_weight):
        # Compute attention coefficients.
        x_j = x_j * edge_weight
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GATConvPool(nn.Module):
    def __init__(self, in_feat, out_feat, n_heads, dropout, alpha, concat, pooling_ratio=0.5):
        super(GATConvPool, self).__init__()
        self.conv = WGATConv(in_feat, out_feat, concat=concat, heads=n_heads,
                             dropout=dropout, negative_slope=alpha)
        self.pool = TopKPooling(in_feat, ratio=pooling_ratio)

    def forward(self, g):
        x = self.conv(g.x, g.edge_index, g.edge_attr)
        x, edge_index, edge_attr, _, _, _ = self.pool(x=x, edge_index=g.edge_index, edge_attr=g.edge_attr)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=g.y)
        return g
