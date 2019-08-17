from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class adni_loader(Dataset):
    """ADNI data loader"""

    def __init__(self, dataset):
        net = []
        for i in range(len(dataset["adjacency_matrix"])):
            net.append(self.process_adj_mat(dataset["adjacency_matrix"][i]))
        self.network = torch.from_numpy(np.array(net)).float()
        self.label = torch.from_numpy(np.array(dataset["dx_label"]))
        self.thck = torch.from_numpy(np.array(dataset["node_feature"])).float()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {"network" : self.network[idx], "thck" : self.thck[idx, :, 2], "label": self.label[idx]}
        return sample

    def process_adj_mat(self, A):
        B = []
        for i in range(len(A)):
            B = B + list(A[i, i+1:len(A)].flatten())
        return np.array(B)


def get_adni_loader(dataset, batch_size):
    dataset = adni_loader(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=batch_size)
    return dataloader

