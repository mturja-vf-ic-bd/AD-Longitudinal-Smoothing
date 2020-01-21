from sklearn.decomposition import PCA
import numpy as np
import torch
from Longitudinal_Classifier.vae import VAE

from Longitudinal_Classifier.read_file import read_all_subjects


class Visualization:
    def pca(self, f):
        # f is a S*N matrix
        pca = PCA(n_components=2)
        f_red = pca.fit_transform(X=f)
        return f_red

    def vae(self, F):
        PATH = "/home/mturja/AD-Longitudinal-Smoothing/Longitudinal_Classifier/vae_model"
        device = torch.device('cuda')
        model = VAE().to(device)
        model.load_state_dict(torch.load(PATH, map_location=device))
        model.eval()
        F = torch.FloatTensor(F).to(device)
        mu, logvar = model.encode(F.view(-1, 148))
        z = model.reparameterize(mu, logvar)
        z = model.classify(z).detach().cpu().numpy()
        return z


if __name__ == '__main__':
    data, count = read_all_subjects(classes=[0, 3], conv_to_tensor=False)
    F = []
    label = []
    for d in data:
        for i in range(len(d["node_feature"])):
            F.append(d["node_feature"][i])
            label.append(d["dx_label"][i])

    F = np.array(F)
    label = np.array(label)
    vis = Visualization()
    F = vis.vae(F)
    X = vis.pca(F)
    # X = vis.vae(F)

    from matplotlib import pyplot as plt
    idx_0 = np.where(label == 0)[0]
    idx_3 = np.where(label == 3)[0]
    plt.scatter(X[idx_0, 0], X[idx_0, 1])
    plt.scatter(X[idx_3, 0], X[idx_3, 1])
    plt.show()
