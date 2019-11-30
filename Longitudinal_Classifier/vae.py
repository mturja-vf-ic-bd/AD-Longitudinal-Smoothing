from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from Longitudinal_Classifier.read_file import read_all_subjects

device = "cuda"

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        dense_dim = [148, 128, 64, 16]
        self.encoder = [
            nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.ReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i - 1], dense_dim[i])
            for i in range(1, len(dense_dim))
        ]

        self.decoder = [
            nn.Sequential(nn.Linear(dense_dim[i], dense_dim[i - 1]), nn.ReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i], dense_dim[i - 1])
            for i in range(len(dense_dim) - 1, 0, -1)
        ]

        self.mu = nn.Linear(dense_dim[-1], dense_dim[-1])
        self.sig = nn.Linear(dense_dim[-1], dense_dim[-1])
        self.classifier = nn.Linear(dense_dim[-1], 2)

        for i, l in enumerate(self.encoder):
            self.add_module('encoder_{}'.format(i), l)
        for i, l in enumerate(self.decoder):
            self.add_module('decoder_{}'.format(i), l)

    def classify(self, z):
        pred = self.classifier(z)
        return pred

    def encode(self, x):
        for i, l in enumerate(self.encoder):
            x = l(x)
            x = F.dropout(x)
        return self.mu(x), self.sig(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        for i, l in enumerate(self.decoder):
            x = l(x)
            if i < len(self.decoder) - 1:
                x = F.dropout(x)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 148))
        z = self.reparameterize(mu, logvar)
        pred = self.classify(mu)
        return self.decode(z), pred, mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, pred, x, label, mu, logvar, w):
    BCE = F.mse_loss(recon_x, x)
    cl_loss = F.cross_entropy(pred, label, weight=w)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    print("Recon: {}, KLD: {}, cl: {}".format(BCE.item(), KLD.item(), cl_loss.item()))
    return cl_loss


def train(epoch, data, label, w=None):
    model.train()
    optimizer.zero_grad()
    recon_batch, pred, mu, logvar = model(data)
    loss = loss_function(recon_batch, pred, data, label, mu, logvar, w)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}, Loss: {:.6f}'.format(
        epoch,
        loss.item()))
    return loss.item()

def test(epoch, data, label, w=None):
    model.eval()
    with torch.no_grad():
        recon_batch, pred, mu, logvar = model(data)
        test_loss = loss_function(recon_batch, pred, data, label, mu, logvar, w).item()
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

    tr_data, ts_data, tr_label, ts_label = train_test_split(f, label)
    smt = SMOTE(random_state=1)
    tr_data, tr_label = smt.fit_sample(tr_data, tr_label)
    # tr_data, tr_label = expand_data(tr_data, tr_label)
    tr_data = torch.FloatTensor(tr_data).to(device)
    ts_data = torch.FloatTensor(ts_data).to(device)
    tr_label = torch.LongTensor(tr_label).to(device)
    ts_label = torch.LongTensor(ts_label).to(device)

    PATH = "/home/mturja/AD-Longitudinal-Smoothing/Longitudinal_Classifier/vae_model"

    model = VAE().to(device)
    # model.load_state_dict(torch.load(PATH, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20000
    tr_ls = []
    ts_ls = []
    for epoch in range(1, epochs + 1):
        tr_ls.append(train(epoch, tr_data, tr_label))
        ts_ls.append(test(epoch, ts_data, ts_label))

    from matplotlib import pyplot as plt
    plt.plot(tr_ls)
    plt.plot(ts_ls)
    plt.show()

    # Save model
    torch.save(model.state_dict(), PATH)
