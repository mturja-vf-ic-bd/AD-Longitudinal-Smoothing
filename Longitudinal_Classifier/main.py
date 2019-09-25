from Longitudinal_Classifier.read_file import *
from Longitudinal_Classifier.model import LongGAT, GATConvTemporalPool, LongGNN
from Longitudinal_Classifier.helper import convert_to_geom_all
from Longitudinal_Classifier.helper import accuracy
from Longitudinal_Classifier.debugger import plot_grad_flow
import torch
import torch.utils.data.dataloader as loader

data = read_all_subjects()
G = []
for d in data:
    G.append(convert_to_geom_all(d["node_feature"], d["adjacency_matrix"], d["dx_label"]))
print("Data read finished !!!")
tr = loader.DataLoader(G, batch_size=32)
for data in tr:
    print(data.pos)

model = LongGNN(in_feat=[1, 5, 5, 5], dropout=0.1, concat=True,
                alpha=0.2, n_heads=1, n_layer=3, n_class=3, pooling_ratio=0.7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5, weight_decay=0.1)
lossFunc = torch.nn.CrossEntropyLoss()

model.to(Args.device)


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
    acc = accuracy(output, target)
    print("Epoch: {} Loss: {}, Accuracy: {}% ".format(epoch, loss.data, acc.data))
    return loss.data, acc.data
        # print(output)


if __name__ == '__main__':
    loss = []
    acc = []
    for i in range(1000):
        lss, ac = train(G, i)
        loss.append(loss)
        acc.append(acc)

    from matplotlib import pyplot as plt
    plt.plot(loss)
    plt.title("Loss")
    plt.show()
    plt.plot(acc)
    plt.title("Accuracy")
    plt.show()
