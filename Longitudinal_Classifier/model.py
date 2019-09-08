import torch
import torch.nn as nn
import torch.nn.functional as F
from Longitudinal_Classifier.layer import GraphAttentionLayer, GATConvPool
import numpy as np
import math


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class LongGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(LongGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=3)
        x = torch.max(x, dim=1)[0]
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class LongGNN(nn.Module):
    def __init__(self, in_feat=[1, 3], n_nodes=148, dropout=0.5, concat=False, alpha=0.2, n_heads=3, n_layer=1, n_class=1, pooling_ratio=0.5):
        super(LongGNN, self).__init__()
        self.n_nodes = n_nodes
        self.n_class = n_class
        self.layer = [GATConvPool(in_feat=in_feat[i], out_feat=in_feat[i+1],
                                  dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads, pooling_ratio=pooling_ratio) for i in range(n_layer)]
        for i, l in enumerate(self.layer):
            self.add_module('GATConvPool_{}'.format(i), l)
        reduced_node_size = n_nodes
        for i in range(n_layer):
            reduced_node_size = math.ceil(reduced_node_size * pooling_ratio)

        self.out_dim = n_class if n_class > 2 else 1
        self.linear = nn.Sequential(nn.Linear(reduced_node_size * in_feat[-1], self.out_dim),
                                    nn.Softmax())

    def forward(self, G_in):
        # G is a list of graph data
        G_out = []
        for g in G_in:
            for l in self.layer:
                g = l(g)
            G_out.append(g)

        out_feat = torch.max(torch.stack([g.x for g in G_out]), 0)[0]
        out_feat = self.linear(out_feat.view(-1, 1).squeeze())
        # for i, g in enumerate(G_out):
        #     if i != 0:
        #         out_feat = torch.stack((out_feat, g.x))

        return out_feat
