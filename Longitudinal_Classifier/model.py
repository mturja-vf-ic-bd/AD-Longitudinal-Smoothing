import torch
import torch.nn as nn
import torch.nn.functional as F
from Longitudinal_Classifier.layer import GraphAttentionLayer, GATConvTemporalPool, GATConvPool
import numpy as np
import math
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv, global_max_pool
from torch_geometric.nn.glob.sort import *
from Longitudinal_Classifier.arg import Args
from Longitudinal_Classifier.helper import normalize_net

I_prime = (1 - torch.eye(Args.n_nodes, Args.n_nodes).unsqueeze(0)).to(Args.device)

class SimpleGCN(nn.Module):
    def __init__(self, gcn_feat, dense_dim):
        super(SimpleGCN, self).__init__()
        self.gcn_layer = []
        for i in range(len(gcn_feat) - 1):
            self.gcn_layer.append(SAGEConv(gcn_feat[i], gcn_feat[i+1], normalize=True))
            self.add_module('GCN_{}'.format(i), self.gcn_layer[i])
        self.dns_lr = [
            nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.ReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i - 1], dense_dim[i])
            for i in range(1, len(dense_dim))
        ]
        for i, l in enumerate(self.dns_lr):
            self.add_module('Dense_{}'.format(i), l)

    def forward(self, g, batch_size, net):
        x, edge_index, edge_attr, batch = g.x, g.edge_index, g.edge_attr, g.batch
        I = torch.eye(x.size(1)).unsqueeze(0).to(Args.device)
        loss = torch.sum(torch.matmul(torch.matmul(torch.transpose(x.view(batch_size, -1, x.size(1)), 1, 2), net), x.view(batch_size, -1, x.size(1))) * I)
        for i, l in enumerate(self.gcn_layer):
            if i > 0:
                x = F.dropout(x)
            x = l(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = F.leaky_relu(x, negative_slope=0.01)
            loss += torch.sum(torch.matmul(torch.matmul(
                torch.transpose(x.view(batch_size, -1, x.size(1)), 1, 2), net),
                x.view(batch_size, -1, x.size(1))) * I)
            # print((x.data == 0).sum().item(), "/", x.size(0))

        # x = global_max_pool(x, batch)
        x = x.view(batch_size, -1, x.size(1))
        x = torch.max(x, dim=1, keepdim=False)[0]
        for l in self.dns_lr:
            x = l(x)
        return x, loss

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
        for l in self.dns_lr:
            x = l(x)
        # x = torch.mean(x, dim=1)
        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
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
    def __init__(self, in_feat=[1, 3], n_nodes=148, dropout=0.5, concat=False, alpha=0.2, n_heads=3, n_layer=1,
                 n_class=1, pooling_ratio=0.5):
        super(LongGNN, self).__init__()
        self.n_nodes = n_nodes
        self.n_class = n_class
        if not concat:
            self.layer = [GATConvTemporalPool(in_feat=in_feat[i], out_feat=in_feat[i + 1],
                                              dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads,
                                              pooling_ratio=pooling_ratio) for i in range(n_layer)]
        else:
            self.layer = [GATConvTemporalPool(in_feat=in_feat[i] * (n_heads ** (i + 1)),
                                              out_feat=in_feat[i + 1] * (n_heads ** (i + 1)),
                                              dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads,
                                              pooling_ratio=pooling_ratio) if i > 0 else
                          GATConvTemporalPool(in_feat=in_feat[i], out_feat=in_feat[i + 1] * n_heads,
                                              dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads,
                                              pooling_ratio=pooling_ratio) for i in range(n_layer)]

        for i, l in enumerate(self.layer):
            self.add_module('GATConvTemporalPool_{}'.format(i), l)
        reduced_node_size = n_nodes
        for i in range(n_layer):
            reduced_node_size = math.ceil(reduced_node_size * pooling_ratio)

        self.out_dim = n_class if n_class > 2 else 1
        self.concat = concat
        if not concat:
            self.linear = nn.Linear(reduced_node_size * in_feat[-1], self.out_dim)
        else:
            self.linear = nn.Linear(reduced_node_size * in_feat[-1] * n_heads, self.out_dim)

    def forward(self, G_in):
        # G is a list of graph data
        for i, l in enumerate(self.layer):
            if i == 0:
                G_out = l(G_in)
            else:
                G_out = l(G_out)

        out_feat = torch.stack([g.x for g in G_out])
        out_feat = torch.max(out_feat, 0)[0]
        out_feat = self.linear(out_feat.view(1, -1)).squeeze()
        # print(out_feat.data)
        # for i, g in enumerate(G_out):
        #     if i != 0:
        #         out_feat = torch.stack((out_feat, g.x))

        return out_feat


class BaselineGNN(nn.Module):
    def __init__(self, in_feat=[1, 3], n_nodes=20, dropout=0.5, concat=False, alpha=0.2, n_heads=3, n_layer=1,
                 n_class=3, pooling_ratio=0.5, batch_size=32):
        super(BaselineGNN, self).__init__()
        self.n_nodes = n_nodes
        self.n_class = n_class
        self.batch_size = batch_size
        if not concat:
            self.layer = [GATConvPool(in_feat=in_feat[i], out_feat=in_feat[i + 1],
                                      dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads,
                                      pooling_ratio=pooling_ratio, n_nodes = int(Args.n_nodes * pooling_ratio ** i)) for i in range(n_layer)]
        else:
            self.layer = [
                GATConvPool(in_feat=in_feat[i] * (n_heads ** (i + 1)), out_feat=in_feat[i + 1] * (n_heads ** (i + 1)),
                            dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads, n_nodes=int(Args.n_nodes * pooling_ratio ** i),
                            pooling_ratio=pooling_ratio) if i > 0 else
                GATConvPool(in_feat=in_feat[i], out_feat=in_feat[i + 1] * n_heads,
                            dropout=dropout, concat=concat, alpha=alpha, n_heads=n_heads, n_nodes=int(Args.n_nodes * pooling_ratio ** i),
                            pooling_ratio=pooling_ratio) for i in range(n_layer)]

        for i, l in enumerate(self.layer):
            self.add_module('GATConvTemporalPool_{}'.format(i), l)
        reduced_node_size = n_nodes
        for i in range(n_layer):
            reduced_node_size = math.ceil(reduced_node_size * pooling_ratio)

        self.out_dim = n_class if n_class > 2 else 1
        self.concat = concat
        dense_dim = [reduced_node_size * in_feat[-1], self.out_dim]
        self.dns_lr = [nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.ReLU()) if i < len(dense_dim) - 1 else
                       nn.Linear(dense_dim[i - 1], dense_dim[i])
                       for i in range(1, len(dense_dim))]
        for i, l in enumerate(self.dns_lr):
            self.add_module('Dense_{}'.format(i), l)
        # if not concat:
        #     self.linear = nn.Linear(reduced_node_size * in_feat[-1], self.out_dim)
        # else:
        #     self.linear = nn.Linear(reduced_node_size * in_feat[-1] * n_heads, self.out_dim)

    def forward(self, g, batch_size):
        # G is a list of graph data
        for i, l in enumerate(self.layer):
            if i == 0:
                g_out = l(g)
            else:
                g_out = l(g_out)
        x = g_out.x.view(batch_size, -1)
        for d in self.dns_lr:
            x = d(x)
        # # x = self.linear(x)
        return x


class ReconNet(nn.Module):
    def __init__(self,  gcn_feat=[164, 32, 32]):
        super(ReconNet, self).__init__()
        self.gcn_layer = []
        # self.sum_feat = sum(gcn_feat) - gcn_feat[0]
        self.sum_feat = gcn_feat[-1]
        self.encode = nn.Sequential(nn.Linear(gcn_feat[0], gcn_feat[1]), nn.ReLU())
        self.rho_mean = nn.Sequential(nn.Linear(self.sum_feat, self.sum_feat), nn.ReLU())
        self.rho_std = nn.Sequential(nn.Linear(self.sum_feat, self.sum_feat), nn.ReLU())
        for i in range(1, len(gcn_feat) - 1):
            self.gcn_layer.append(SAGEConv(gcn_feat[i], gcn_feat[i+1], normalize=True))
            self.add_module('GCN_{}'.format(i-1), self.gcn_layer[i-1])
            nn.init.normal_(self.gcn_layer[i-1].weight, 0, 1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, g):
        x, edge_index, edge_attr, batch, batch_size = g.x, g.edge_index, g.edge_attr, g.batch, g.num_graphs
        # x = self.encode(x)
        for i, l in enumerate(self.gcn_layer):
            x_new = l(x=x, edge_index=edge_index, edge_weight=edge_attr)
            x_new = F.relu(x_new)
            if i < len(self.gcn_layer) - 1:
                x_new = F.dropout(x_new)
            # if i == 0:
            #     x_multi = x
            # x_multi = torch.cat((x_multi, x_new), dim=1)
            x = x_new
        # x_multi = x_multi.view(batch_size, -1, x_multi.size(1))
        x = x.view(batch_size, -1, x.size(1))
        mu = self.rho_mean(x)
        logvar = self.rho_std(x)
        # z = self.reparameterize(mu, logvar)
        # x_std = x.std(dim=2, keepdim=True)
        # x = (x - x.mean(dim=2, keepdim=True)) / x_std
        A_recon = torch.matmul(mu, torch.transpose(mu, 1, 2)) * I_prime
        A_mask = torch.sigmoid(torch.matmul(logvar, torch.transpose(logvar, 1, 2))) * I_prime
        deg = (torch.sum(A_recon, dim=1, keepdim=True) + 1e-10) ** (-0.5)
        norm = torch.matmul(torch.transpose(deg, 1, 2), deg)
        A_recon = A_recon * norm
        return A_recon, A_mask

class LinearGraphVAE(nn.Module):
    def __init__(self):
        super(LinearGraphVAE, self).__init__()
        self.layer = []