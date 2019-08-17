import torch
import torch.nn as nn
from torch.autograd import Variable
from read_file import read_all_subjects, get_baselines, get_strat_label
from helpers import get_train_test_fold
from arg import Args
import numpy as np
from ADNI_loader import get_adni_loader
from model import LogisticRegression
from matplotlib import pyplot as plt
import networkx as nx

num_epochs = 50
batch_size = 60
learning_rate = 0.005

def get_group_graphs(dataset):
    l = dataset['dx_label']
    n = dataset['adjacency_matrix']
    net_dict = {}
    for i in range(len(l)):
        if l[i] not in net_dict:
            net_dict[l[i]] = [n[i]]
        else:
            net_dict[l[i]].append(n[i])

    for key, val in net_dict.items():
        val = np.array(val).mean(axis=0)
        val = line_graph(val)
        D = np.diag(val.sum(axis=1))
        L = D - val
        net_dict[key] = torch.from_numpy(L)
        net_dict[key] = net_dict[key].type(torch.FloatTensor)
    return net_dict

def line_graph(A):
    """
    Convert to line_graph
    :param A: input adjacency matrix
    :return B: output adjacency matrix
    """

    G = nx.from_numpy_array(A)
    L = nx.line_graph(G)
    B = nx.to_numpy_array(L)
    return B

def spatial_loss(param, net_dict):
    if len(param.shape) == 1:
        return 0
    loss = 0
    for key, val in net_dict.items():
        loss += torch.matmul(torch.matmul(param[key], val), param[key])

    return loss


def thck_loss(edge_param, node_param):
    # Convert to upper triangular matrix
    n = node_param.shape[1]
    total_loss = 0
    for i in range(len(edge_param)):
        param = 10*torch.triu(torch.ones(n, n), 1) + 20*torch.tril(torch.ones(n, n), -1)
        param[param == 10] = edge_param[i]
        param[param == 20] = edge_param[i]

        # match shape
        nd_p = node_param[i].repeat(n, 1)
        loss = (nd_p - param)**2
        total_loss += loss.sum()
    return total_loss


def train(train_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    train_loader = get_adni_loader(train_dataset, batch_size)
    model = LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training the Model
    loss_plot = []
    net_dict = get_group_graphs(train_dataset)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            nets = batch["network"]
            labels = batch["label"]
            thck = batch["thck"]
            n_node = thck.shape[1]
            n_edge = nets.shape[1]
            input = torch.cat((nets, thck), 1)
            input = Variable(input.view(-1, input_size))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, labels)
            sp_loss = 0
            # print("Loss: ", loss)
            l1_crit = nn.L1Loss(size_average=False)
            reg_loss = 0
            temp_loss = 0
            l2_loss = nn.MSELoss(size_average=False)
            th_ls = 0
            for param in model.parameters():
                if len(param.shape) > 1:
                    edge_param = param[:, :n_edge]
                    node_param = param[:, n_edge:]
                    reg_loss += l1_crit(param, torch.zeros_like(param))
                    #sp_loss += spatial_loss(edge_param, net_dict)
                    for i in range(1, len(param)):
                        temp_loss += l2_loss(param[i], param[i-1])

                    #th_ls = thck_loss(edge_param, node_param)

            alpha_1 = torch.tensor(2.)
            alpha_2 = torch.tensor(0.)
            alpha_sp = torch.tensor(0.)
            alpha_thck = torch.tensor(0.)
            loss = loss + alpha_1 * reg_loss + alpha_2 * temp_loss + alpha_sp * sp_loss + \
                   alpha_thck * th_ls
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            loss_plot.append(loss)
            print('Epoch: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, loss.item()))

            # print("L1 Loss: {}\nSpatial Loss: {}\nTemporal Smoothness: {}\n".format(reg_loss, sp_loss, temp_loss))

    plt.plot(loss_plot)
    plt.show()
    for param in model.linear.parameters():
        print("Zeros: ", (param < 1e-9).sum())
    return model


def test(model, test_dataset):
    test_loader = get_adni_loader(test_dataset, batch_size)
    # Test the Model
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            nets = batch["network"]
            labels = batch["label"]
            thck = batch["thck"]
            n_edge = nets.shape[1]
            input = torch.cat((nets, thck), 1)
            input = Variable(input.view(-1, input_size))
            labels = Variable(labels)
            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    acc = (100 * correct / total)
    print('Accuracy of the model %d %%' % acc)
    return acc


if __name__ == '__main__':
    print("Classification using dong's method. "
          "Paper Link: https://www.academia.edu/36809655/Subnetwork_Mining_with_Spatial_and_Temporal_Smoothness")

    # Read DataSet
    y_strat = get_strat_label()
    data_set = get_baselines(net_dir=Args.NETWORK_DIR, label=y_strat)
    folds = get_train_test_fold(data_set)

    # Hyper Parameters
    shape_net = data_set["adjacency_matrix"][0].shape
    input_size = shape_net[0] * (shape_net[1] - 1) // 2 + shape_net[0]
    num_classes = len(np.unique(data_set["dx_label"]))
    print(num_classes)

    acc_list = []
    i = 0
    for train_fold, test_fold in folds:
        model = train(train_fold)
        acc_list.append(test(model, test_fold))
        torch.save(model.state_dict(), 'model_' + str(i))
        i = i + 1
    print("mean accuracy: ", np.mean(acc_list))
