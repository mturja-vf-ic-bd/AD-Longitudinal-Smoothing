from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class adni_loader(Dataset):
    """ADNI data loader"""

    def __init__(self, dataset):
        self.network = torch.from_numpy(np.array(dataset["adjacency_matrix"])).float()
        self.label = torch.from_numpy(np.array(dataset["dx_label"]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {"network" : self.network[idx], "label": self.label[idx]}
        return sample


def get_adni_loader(dataset, batch_size):
    dataset = adni_loader(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=batch_size)
    return dataloader

