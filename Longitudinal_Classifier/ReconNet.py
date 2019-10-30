from Longitudinal_Classifier.model import ReconNet
from Longitudinal_Classifier.debugger import plot_grad_flow
import torch
import torch_geometric.data.dataloader as loader
from operator import itemgetter
import timeit
from Longitudinal_Classifier.helper import *

start = timeit.default_timer()
# Prepare data
data, count = read_all_subjects(classes=[0, 1, 2, 3], conv_to_tensor=False)

count = 1 / count
count[torch.isinf(count)] = 0

G = []
Y = []
for d in data:
    for i in range(len(d["node_feature"])):
        G.append(convert_to_geom(d["node_feature"][i], d["adjacency_matrix"][i], d["dx_label"][i]))

print("Data read finished !!!")
stop = timeit.default_timer()
print('Time: ', stop - start)

train_idx, test_idx = get_train_test_fold(G, [g.y for g in G])
train_idx = train_idx[0]
test_idx = test_idx[0]
train_data = list(itemgetter(*train_idx)(G))
test_data = list(itemgetter(*test_idx)(G))
train_loader = loader.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = loader.DataLoader(test_data, batch_size=32)

model = ReconNet()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.01)
lossFunc = torch.nn.CrossEntropyLoss(weight=count)

def train_baseline(epoch):
    model.train()

    loss_all = 0
    i = 0
    acc = 0
    for data in train_loader:
        data = data.to(Args.device)
        # data.x = (data.x - torch.mean(data.x, dim=0)) / torch.std(data.x, dim=0)
        # data.x = normalize_feat(data.x)
        data.x = data.x.view(-1, 1)
        optimizer.zero_grad()
        recon = model(data)
        # if i == 0 and epoch % 50 == 0:
        #     plot_grad_flow(model.named_parameters())
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss
        #     }, Args.MODEL_CP_PATH + "/model_chk_" + str(epoch))
        #
        # loss_all += data.num_graphs * loss.item()
        # optimizer.step()
        # a, f1 = accuracy(output, data.y)
        # acc = acc + a
        # # print("Acc: {.2}%".format(acc))
        # i = i + 1
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
            data.x = (data.x - torch.mean(data.x, dim=0)) / torch.std(data.x, dim=0)
            data.x = data.x.view(-1, 1)
            pred = model(data, data.num_graphs, net).detach().cpu()
            # pred = model(data).detach().cpu()

            print("Out: ", pred.data)
            label = data.y.detach().cpu()
            predictions.append(pred)
            labels.append(label)
            a, f1 = accuracy(pred, label)
            acc = acc + a
            f1_score = f1_score + f1
            i = i + 1
            print("Pred: ", torch.argmax(pred, dim=1))
            print("GT: ", label)
    return acc / i, f1 / i


if __name__ == '__main__':
    loss = []
    ac = []
    prev_lss = 0
    for i in range(3000):
        lss, acc = train_baseline(i)
        loss.append(lss)
        ac.append(acc)
        print("Epoch: {}, Loss: {:0.3f}, acc: {:0.2f}".format(i, lss, acc))
        # if (prev_lss - lss) ** 2 < 1e-8:
        #     break
        # prev_lss = lss

    from matplotlib import pyplot as plt
    # plt.ylim(500)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.savefig('loss.png')
    plt.show()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(ac)
    plt.savefig('acc.png')
    plt.show()

    test_acc, test_f1 = test(test_loader)
    print("Test Accuracy: {:0.2f}%, \nF1 Score: {:0.2f}".format(test_acc, test_f1))