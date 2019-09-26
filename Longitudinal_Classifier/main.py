from Longitudinal_Classifier.read_file import *
from Longitudinal_Classifier.model import LongGAT, GATConvTemporalPool, LongGNN, BaselineGNN
from Longitudinal_Classifier.helper import convert_to_geom
from Longitudinal_Classifier.helper import accuracy, get_train_test_fold
from Longitudinal_Classifier.debugger import plot_grad_flow
import torch
import torch_geometric.data.dataloader as loader
from operator import itemgetter


# Prepare data
data = read_all_subjects(classes=[0, 1, 2])
G = []
for d in data:
    for i in range(len(d["node_feature"])):
        G.append(convert_to_geom(d["node_feature"][i], d["adjacency_matrix"][i], d["dx_label"][i]))
print("Data read finished !!!")

train_idx, test_idx = get_train_test_fold(G, [g.y for g in G])
train_idx = train_idx[0]
test_idx = test_idx[0]
train_data = list(itemgetter(*train_idx)(G))
test_data = list(itemgetter(*test_idx)(G))
train_loader = loader.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = loader.DataLoader(test_data, batch_size=32)

# Prepare model
model = BaselineGNN(in_feat=[1, 5, 5], dropout=0.3, concat=False,
                alpha=0.2, n_heads=1, n_layer=2, n_class=3, pooling_ratio=0.4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.5)
lossFunc = torch.nn.CrossEntropyLoss()

model.to(Args.device)

def train_baseline():
    model.train()

    loss_all = 0
    i = 0
    acc = 0
    for data in train_loader:
        data = data.to(Args.device)
        optimizer.zero_grad()
        output = model(data, data.num_graphs)
        loss = lossFunc(output, data.y)
        loss.backward()
        # if i == 0:
        #     plot_grad_flow(model.named_parameters())
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        a, f1 = accuracy(output, data.y)
        acc = acc + f1
        # print("Acc: {.2}%".format(acc))
        i = i + 1
    return loss_all / len(G), acc / i


def test(loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        acc = 0
        f1_score = 0
        i = 0
        for data in loader:
            data = data.to(Args.device)
            pred = model(data, data.num_graphs).detach().cpu()
            label = data.y.detach().cpu()
            predictions.append(pred)
            labels.append(label)
            a, f1 = accuracy(pred, label)
            acc = acc + a
            f1_score = f1_score + f1
            i = i + 1
    return acc / i, f1 / i

def train(data, epoch):
    """
    Train the Longitudinal model
    :param data: Graph Object from torch-geometric
    :param target: class labels
    :return:
    """

    output = torch.empty((len(data), Args.n_class))
    target = torch.empty(len(data), dtype=torch.long)
    loss = torch.zeros(1).cuda()
    optimizer.zero_grad()
    for i, d in enumerate(data):
        # print("Data: ", i)
        out = model(d)
        loss = loss + lossFunc(out.view(1, -1), torch.LongTensor([d[-1].y]).cuda())
        output[i] = out
        target[i] = d[-1].y
        # print(output[i])
    # loss = lossFunc(output, target)

    loss.backward()
    # plot_grad_flow(model.named_parameters())
    optimizer.step()
    acc, f1 = accuracy(output, target)
    print("Epoch: {} Loss: {}, Accuracy: {}%, F1: {}% ".format(epoch, loss.data, acc.data, f1))
    return loss.data, acc.data


if __name__ == '__main__':
    loss = []
    ac = []
    for i in range(100):
        lss, acc = train_baseline()
        loss.append(lss)
        ac.append(acc)
        print("Epoch: {}, Loss: {:0.3f}, f1: {:0.2f}".format(i, lss, acc))

    from matplotlib import pyplot as plt
    # plt.ylim(500)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(ac)
    plt.show()

    test_acc, test_f1 = test(test_loader)
    print("Test Accuracy: {:0.2f}%, \nF1 Score: {:0.2f}".format(test_acc, test_f1))