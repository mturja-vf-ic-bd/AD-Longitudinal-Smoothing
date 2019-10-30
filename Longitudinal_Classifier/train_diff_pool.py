from Longitudinal_Classifier.layer2 import GDNet
from Longitudinal_Classifier.debugger import plot_grad_flow
import torch
import torch.utils.data.dataloader as loader
from operator import itemgetter
import timeit
from Longitudinal_Classifier.helper import *

start = timeit.default_timer()
# Prepare data
data, count = read_all_subjects(classes=[0, 3], conv_to_tensor=False)
net_struct = get_aggr_net(data)
net_struct = torch.FloatTensor(normalize_net(net_struct)).to(Args.device)
net_cmn = read_net_cmn()
X, Y = get_crossectional(data)
S, S_com = get_cluster_assignment_matrix()
X = np.concatenate((X[:, :, np.newaxis], np.repeat(S_com[np.newaxis, :, :], X.shape[0], axis=0)), axis=2)
count = 1 / count
count[torch.isinf(count)] = 0

print("Data read finished !!!")
stop = timeit.default_timer()
print('Time: ', stop - start)

train_idx, test_idx = get_train_test_fold(X, Y)
train_idx = train_idx[0]
test_idx = test_idx[0]
train_x = X[train_idx]
train_y = Y[train_idx]
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
train_x = train_x.reshape(train_x.shape[0], -1)
train_x, train_y = sm.fit_resample(train_x, train_y)
train_x = torch.FloatTensor(train_x.reshape(train_x.shape[0], 148, -1)).to(Args.device)
train_y = torch.LongTensor(train_y).to(Args.device)
test_x = torch.FloatTensor(list(itemgetter(*test_idx)(X))).to(Args.device)
test_y = torch.LongTensor(list(itemgetter(*test_idx)(Y))).to(Args.device)

# Prepare model
model = GDNet([8, 128, 64], dropout=0.2, alpha=0.01, n_class=4, c=[24, 8]).to(Args.device)
print("Cuda Available: ", torch.cuda.is_available(),
      "\nDevice: ", Args.device)


optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, weight_decay=0.05)
lossFunc = torch.nn.CrossEntropyLoss(weight=count)

def train_baseline(epoch, train_x, train_y):
    model.train()

    train_x = normalize_feat(train_x)
    optimizer.zero_grad()
    output, _, link_loss, ent_loss = model(train_x, net_struct)
    # output = model(data)
    loss = 50*lossFunc(output, train_y)
    loss += link_loss + ent_loss
    loss.backward()
    if epoch % 20 == 1:
        plot_grad_flow(model.named_parameters())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, Args.MODEL_CP_PATH + "/model_chk_" + str(epoch))

    optimizer.step()
    acc, f1 = accuracy(output, train_y)
    return loss.item(), acc

def test(test_x, test_y):
    model.eval()

    with torch.no_grad():
        test_x = normalize_feat(test_x)
        pred = model(test_x, net_struct).detach().cpu()
        # pred = model(data).detach().cpu()

        label = test_y.detach().cpu()
        acc, f1 = accuracy(pred, label)
        print("Pred: ", torch.argmax(pred, dim=1))
        print("GT: ", label)
    return acc, f1


if __name__ == '__main__':
    loss = []
    ac = []
    prev_lss = 0
    for i in range(1000):
        lss, acc = train_baseline(i, train_x, train_y)
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

    test_acc, test_f1 = test(test_x, test_y)
    print("Test Accuracy: {:0.2f}%, \nF1 Score: {:0.2f}".format(test_acc, test_f1))