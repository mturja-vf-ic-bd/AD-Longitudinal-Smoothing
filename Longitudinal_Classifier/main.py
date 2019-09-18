from Longitudinal_Classifier.read_file import *
from Longitudinal_Classifier.model import LongGAT, GATConvPool, LongGNN
from Longitudinal_Classifier.helper import convert_to_geom_all
import torch
from Longitudinal_Classifier.helper import accuracy

data = read_all_subjects()
G = []
for d in data:
    G.append(convert_to_geom_all(d["node_feature"], d["adjacency_matrix"], d["dx_label"]))
print("Data read finished !!!")

model = LongGNN(in_feat=[1, 1, 1], dropout=0.7, concat=False,
                alpha=0.2, n_heads=3, n_layer=2, n_class=3, pooling_ratio=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.5)
lossFunc = torch.nn.CrossEntropyLoss()

if Args.cuda:
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
    for i, d in enumerate(data):
        output[i] = model(d)
        target[i] = d[-1].y
        # print(output[i])
    loss = lossFunc(output, target)
    acc = accuracy(output, target)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        print("Epoch: {} Loss: {}, Accuracy: {}% ".format(epoch, loss.data, acc.data))
        # print(output)


if __name__ == '__main__':
    for i in range(500):
        train(G, i)
