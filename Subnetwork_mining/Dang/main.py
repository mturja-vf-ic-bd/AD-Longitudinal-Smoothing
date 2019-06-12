import torch
import torch.nn as nn
from torch.autograd import Variable
from read_file import read_all_subjects, get_baselines, get_strat_label
from helpers import get_train_test_fold
from arg import Args
import numpy as np
from ADNI_loader import get_adni_loader
from model import LogisticRegression

def train(train_dataset):
    train_loader = get_adni_loader(train_dataset, batch_size)
    model = LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training the Model
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            nets = batch["network"]
            labels = batch["label"]
            nets = Variable(nets.view(-1, input_size))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(nets)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))




if __name__ == '__main__':
    print("Classification using dong's method. "
          "Paper Link: https://www.academia.edu/36809655/Subnetwork_Mining_with_Spatial_and_Temporal_Smoothness")

    # Read DataSet
    y_strat = get_strat_label()
    data_set = get_baselines(net_dir=Args.NETWORK_DIR, label=y_strat)
    folds = get_train_test_fold(data_set)

    # Hyper Parameters
    shape_net = data_set["adjacency_matrix"][0].shape
    input_size = shape_net[0] * shape_net[1]
    num_classes = len(np.unique(data_set["dx_label"]))
    print(num_classes)
    num_epochs = 5
    batch_size = 10
    learning_rate = 0.001

    for train_fold, test_fold in folds:
        train(train_fold)

