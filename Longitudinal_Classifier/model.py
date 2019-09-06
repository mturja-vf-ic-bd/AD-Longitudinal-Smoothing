import torch
import torch.nn as nn
import torch.nn.functional as F
from Longitudinal_Classifier.layer import GraphAttentionLayer, WGATConv
from torch_geometric.data import Data


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


from torch_geometric.nn import GATConv, TopKPooling

class ConvPool(nn.Module):
    def __init__(self, in_feat, out_feat, n_heads, dropout, alpha, pool_ratio=0.5):
        super(ConvPool, self).__init__()
        self.conv = WGATConv(in_feat, out_feat, concat=False, heads=n_heads, dropout=dropout, negative_slope=alpha)
        self.pool = TopKPooling(in_feat, ratio=pool_ratio)

    def forward(self, g):
        g = self.conv.forward(g.x, g.edge_index, g.edge_attr)
        x, edge_index, _, _, _ = self.pool(g)
        g = Data(x=x, edge_index=edge_index)
        return g
