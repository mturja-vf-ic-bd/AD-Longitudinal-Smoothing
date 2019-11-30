import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Longitudinal_Classifier.helper import get_aggr_net, normalize_net
from Longitudinal_Classifier.read_file import read_all_subjects
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from Longitudinal_Classifier.spectrum_analysis import GraphSpectrum

device = "cuda"

class ChebConv(nn.Module):
    def __init__(self, k=3, d=10, dense_dim=[148, 64, 2]):
        super(ChebConv, self).__init__()

        self.k = k
        self.d = d
        self.w = nn.Parameter(torch.rand(k, d))

        self.encoder = [
            nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.ReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i - 1], dense_dim[i])
            for i in range(1, len(dense_dim))
        ]

        for i, l in enumerate(self.encoder):
            self.add_module('encoder_{}'.format(i), l)

    def forward(self, f, L):
        f_out = torch.zeros((f.size(0), f.size(1), self.d)).to(device)
        for l in range(self.d):
            f_lin = f.clone()
            for k in range(1, self.k + 1):
                op = (self.w[k-1, l] / self.w[:, l].sum()) * (L ** k)
                f_lin += torch.matmul(f, op)
            f = F.dropout(F.relu(f_lin))
            f_out[:,:,l] = f

        if self.k > 0:
            f_out = torch.max(f_out, dim=2)[0]
        else:
            f_out = f / self.d
        for i, l in enumerate(self.encoder):
            f_out = l(f_out)
            f_out = F.dropout(f_out)

        return f_out

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(pred, label):
    cl_loss = F.cross_entropy(pred, label)
    return cl_loss

def accuracy(pred, label):
    pred = torch.max(pred, dim=1)[1]
    return (pred == label).sum() * 100 / len(label)

def train(epoch, data, label, L, w=None):
    model.train()
    optimizer.zero_grad()
    pred = model(data, L)
    loss = loss_function(pred, label)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}, Loss: {:.6f}'.format(
        epoch,
        loss.item()))
    return loss.item()

def test(epoch, data, label, L, w=None):
    model.eval()
    with torch.no_grad():
        pred = model(data, L)
        test_loss = loss_function(pred, label).item()
    print('====> Epoch: {} Test Loss: {:.4f}'.format(epoch, test_loss))
    return test_loss

def expand_data(X, y):
    noise = np.random.normal(0, 0.2, X.shape)
    X_aug = X + noise
    X_new = np.concatenate((X, X_aug), axis=0)
    y_new = np.concatenate((y, y), axis=0)
    return X_new, y_new


if __name__ == "__main__":
    data, count = read_all_subjects(classes=[0, 3], conv_to_tensor=False)
    net = get_aggr_net(data, label=[0])
    gs = GraphSpectrum(net)
    L = gs.normalized_laplacian()
    L = torch.FloatTensor(L).to(device)
    f = []
    label = []
    for d in data:
        for i in range(len(d["node_feature"])):
            f.append(d["node_feature"][i])
            label.append(d["dx_label"][i])

    label = np.array(label)
    label[label == np.min(label)] = 0
    label[label == np.max(label)] = 1

    # a = (label == 0).sum() * 1.0
    # b = (label == 1).sum() * 1.0
    # w = torch.FloatTensor([(a+b)/a, (a+b)/b]).to(device)

    tr_data, ts_data, tr_label, ts_label = train_test_split(f, label, random_state=1)
    smt = SMOTE(random_state=1)
    tr_data, tr_label = smt.fit_sample(tr_data, tr_label)
    # tr_data, tr_label = expand_data(tr_data, tr_label)
    tr_data = torch.FloatTensor(tr_data).to(device)
    ts_data = torch.FloatTensor(ts_data).to(device)
    tr_label = torch.LongTensor(tr_label).to(device)
    ts_label = torch.LongTensor(ts_label).to(device)

    model = ChebConv().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.01)
    epochs = 5000
    PATH = "/home/mturja/AD-Longitudinal-Smoothing/Longitudinal_Classifier/cheb_model_" + str(epochs) + "_" + str(model.k) + "_" + str(model.d)

    # model.load_state_dict(torch.load(PATH, map_location=device))
    tr_ls = []
    ts_ls = []
    for epoch in range(1, epochs + 1):
        tr_ls.append(train(epoch, tr_data, tr_label, L))
        ts_ls.append(test(epoch, ts_data, ts_label, L))

    from matplotlib import pyplot as plt
    plt.plot(tr_ls)
    plt.plot(ts_ls)
    plt.show()

    pred = model(ts_data, L)
    print("{:2f}%".format(accuracy(pred, ts_label)))

    # Save model
    torch.save(model.state_dict(), PATH)
