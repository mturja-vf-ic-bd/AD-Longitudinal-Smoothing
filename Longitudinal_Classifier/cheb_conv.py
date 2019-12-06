import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle as pkl

from Longitudinal_Classifier.helper import get_aggr_net, normalize_net
from Longitudinal_Classifier.read_file import read_all_subjects
import numpy as np
from sklearn.model_selection import train_test_split
from Longitudinal_Classifier.spectrum_analysis import GraphSpectrum
from skorch import NeuralNetClassifier
from Longitudinal_Classifier.helper import get_train_test_fold
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


class ChebConv(nn.Module):
    def __init__(self, L, nhops=0, nkernel=0, dense_dim=None, dropout=0.5):
        super(ChebConv, self).__init__()

        if dense_dim is None:
            dense_dim = [148, 148, 148, 64, 4]
        self.L = L
        self.nhops = nhops
        self.nkernel = nkernel
        self.dropout = dropout
        # self.attn = nn.Parameter(torch.ones(148, 148) - torch.eye(148))
        self.w = nn.Parameter(torch.rand(nhops, nkernel))
        if nkernel > 0:
            self.comb_scale = nn.Linear(nkernel, nkernel)
        self.encoder = [
            nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.LeakyReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i - 1], dense_dim[i])
            for i in range(1, len(dense_dim))
        ]

        for i, l in enumerate(self.encoder):
            self.add_module('encoder_{}'.format(i), l)

        self.apply(init_weights)

    def forward(self, f):
        L = self.L
        if self.nkernel > 0:
            f_out = torch.zeros((f.size(0), f.size(1), self.nkernel)).to(device)
        for l in range(self.nkernel):
            f_lin = f.clone()
            for k in range(1, self.nhops + 1):
                op = (self.w[k - 1, l] / self.w[:, l].sum()) * (L ** k)
                f_lin += torch.matmul(f, op)
            # f = F.dropout(F.relu(f_lin))
            f = F.leaky_relu(f_lin)
            f_out[:, :, l] = f

        if self.nhops > 0 and self.nkernel > 0:
            f_out = F.leaky_relu(self.comb_scale(f_out))
            f_out = torch.max(f_out, dim=2)[0]
            # f_out = f_out.view(f_out.size(0), -1)
            # f_out = f_out[:, :, -1]
        else:
            f_out = f
        for i, l in enumerate(self.encoder):
            f_out = l(f_out)
            if len(self.encoder) - 1 > i > 0:
                f_out = F.dropout(f_out, p=self.dropout)

        return f_out


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ClassificationLoss(nn.Module):
    def __init__(self, w):
        super(ClassificationLoss, self).__init__()
        self.w = w

    def forward(self, y_pred, y_true):
        cl_loss = F.cross_entropy(y_pred, y_true, self.w)
        return cl_loss


def loss_function(pred, label, w):
    cl_loss = F.cross_entropy(pred, label, w)
    return cl_loss


def accuracy(pred, label):
    from sklearn.metrics import confusion_matrix
    pred = torch.max(pred, dim=1)[1]
    return (pred == label).sum() * 100 / len(label), confusion_matrix(label.detach().cpu().numpy(),
                                                                      pred.detach().cpu().numpy())


def train(epoch, data, label, L, w=None):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    loss = loss_function(pred, label, w)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}, Loss: {:.6f}'.format(
        epoch,
        loss.item()))
    return loss.item()


def test(epoch, data, label, L, w=None):
    model.eval()
    with torch.no_grad():
        pred = model(data)
        test_loss = loss_function(pred, label, w).item()
    print('====> Epoch: {} Test Loss: {:.4f}'.format(epoch, test_loss))
    return test_loss


def expand_data(X, y):
    noise = np.random.normal(0, 0.2, X.shape)
    X_aug = X + noise
    X_new = np.concatenate((X, X_aug), axis=0)
    y_new = np.concatenate((y, y), axis=0)
    return X_new, y_new


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    data, count = read_all_subjects(classes=[0, 2, 3], conv_to_tensor=False)
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

    f = np.array(f)
    ld, f = gs.transform_to_spectrum(f.T)
    f = f.T
    f = (f - f.mean(axis=0)) / f.std(axis=0)
    label = np.array(label)
    # label[label == np.min(label)] = 0
    # label[label == np.max(label)] = 1

    # a = (label == 0).sum() * 1.0
    # b = (label == 1).sum() * 1.0
    # w = torch.FloatTensor([(a+b)/a, (a+b)/b]).to(device)
    w = 1 / count
    w[torch.isinf(w)] = 0

    try:
        with open('fold.p', 'rb') as fl:
            fold_dict = pkl.load(fl)
            train_fold, test_fold = fold_dict["train_fold"], fold_dict["test_fold"]
    except:
        train_fold, test_fold = get_train_test_fold(f, label)
        with open('fold.p', 'wb') as fl:
            pkl.dump({"train_fold": train_fold, "test_fold": test_fold}, fl)

    # tr_data, ts_data, tr_label, ts_label = train_test_split(f, label, random_state=1)
    # smt = SMOTE(random_state=1)
    # tr_data, tr_label = smt.fit_sample(tr_data, tr_label)
    # tr_data, tr_label = expand_data(tr_data, tr_label)

    acc_list = []
    conf_list = []
    for i in range(0, len(train_fold)):
        tr_idx = train_fold[i]
        ts_idx = test_fold[i]
        tr_data = torch.FloatTensor(f[tr_idx]).to(device)
        ts_data = torch.FloatTensor(f[ts_idx]).to(device)
        tr_label = torch.LongTensor(label[tr_idx]).to(device)
        ts_label = torch.LongTensor(label[ts_idx]).to(device)

        model = ChebConv(L).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        epochs = 3000
        PATH = "/home/mturja/AD-Longitudinal-Smoothing/Longitudinal_Classifier/cheb_model_" + str(epochs) + "_" + str(
            model.nhops) + "_" + str(model.nkernel) + "_" + str(i)
        # epochs = 2000
        # model.load_state_dict(torch.load(PATH, map_location=device))
        tr_ls = []
        ts_ls = []
        for epoch in range(1, epochs + 1):
            tr_ls.append(train(epoch, tr_data, tr_label, L, w))
            ts_ls.append(test(epoch, ts_data, ts_label, L, w))

        from matplotlib import pyplot as plt

        plt.plot(tr_ls)
        plt.plot(ts_ls)
        plt.show()
        with torch.no_grad():
            model.eval()
            pred = model(ts_data)
            acc, conf = accuracy(pred, ts_label)
            acc_list.append(acc.item())
            conf_list.append(conf)
            print("{:2f}%".format(acc))
            print(conf)

        # Save model
        torch.save(model.state_dict(), PATH)
        # plt.show()
    print(acc_list)

    with open('conf_list' + "_" + str(model.nhops) + "_" + str(model.nkernel), 'wb') as f:
        pkl.dump(conf_list, f)
        pkl.dump(acc_list, f)
    # print(conf_list)
    # net = NeuralNetClassifier(
    #     ChebConv,
    #     module__L=L,
    #     module__nhops=2,
    #     module__nkernel=5,
    #     criterion=ClassificationLoss,
    #     criterion__w=w,
    #     max_epochs=7000,
    #     optimizer=optim.SGD,
    #     optimizer__lr=1e-3,
    #     device=device,
    # )
    # net.fit(tr_data, tr_label)
    # # print()
    #
    # pred = net.predict(ts_data)
    # acc, conf = accuracy(pred, ts_label)
    # print("{:2f}%".format(acc))
    # print(conf)



