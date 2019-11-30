from Longitudinal_Classifier.model import ReconNet
from Longitudinal_Classifier.debugger import plot_grad_flow
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import torch_geometric.data.dataloader as loader
from operator import itemgetter
import timeit
from Longitudinal_Classifier.helper import *
from matplotlib import pyplot as plt
from utils.sortDetriuxNodes import sort_matrix
from Longitudinal_Classifier.logger import Logger
start = timeit.default_timer()
# Prepare data
data, count = read_all_subjects(classes=[0, 1, 2, 3], conv_to_tensor=False)

# Structural Mask
net = get_aggr_net(data)
net = normalize_net(net)
net = torch.FloatTensor(net).to(Args.device)

# Cortico Spatial Network
csg = torch.FloatTensor(np.loadtxt('/home/mturja/tmp/AD-Long/cortico_spatial_graph.txt'))
csg[csg < 0.005] = 0
csg[csg > 0] = 1
csg = (torch.eye(Args.n_nodes) - csg).to(Args.device)

count = 1 / count
count[torch.isinf(count)] = 0

print("Device: ", Args.device)

G = []
Y = []
I_node = torch.eye(148)
for d in data:
    for i in range(len(d["node_feature"])):
        G.append(convert_to_geom(d["node_feature"], d["adjacency_matrix"][i], d["dx_label"][i], d["age"][i], normalize=True))
        # G.append(convert_to_geom(I_node, d["adjacency_matrix"][i], d["dx_label"][i], normalize=True, threshold=0.001))


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

S, S_con = get_cluster_assignment_matrix()
S = torch.FloatTensor(S).unsqueeze(0).to(Args.device)

model = ReconNet(gcn_feat=[164, 32, 16])
model.to(Args.device)

I_prime = (1 - torch.eye(Args.n_nodes, Args.n_nodes).unsqueeze(0)).to(Args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, weight_decay=0.01)

logger = Logger('./logs')

def train_baseline(epoch):
    model.train()

    loss_all = 0
    i = 0
    N = 0
    for data in train_loader:
        data = data.to(Args.device)
        # data.x = (data.x - torch.mean(data.x, dim=0)) / torch.std(data.x, dim=0)
        # data.x = normalize_feat(data.x)
        optimizer.zero_grad()
        recon, x, x_loss = model(data)
        # KLD = -0.5 / Args.n_nodes * torch.mean(torch.sum(
        #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        gt = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        # loss_mask = torch.mean(-net * torch.log(mask+1e-5) - (1 - net) * torch.log(1 - mask + 1e-5)) + torch.mean(mask * torch.log(1 - mask + 1e-5))
        recon_loss = F.mse_loss(recon, gt, reduce=None)
        spatial_loss = torch.sum(torch.matmul(torch.matmul(torch.transpose(x, 1, 2), gt), x) * torch.eye(x.size(2)).to(Args.device))
        # loss = loss + loss_mask + torch.mean(torch.abs(mask))
        n = recon.size(1)
        l1_loss_1 = torch.sum(torch.abs(recon)[:, 0:n//2, n//2:n]) + torch.sum(torch.abs(recon)[:, n//2:n, 0:n//2])
        l1_loss_2 = torch.sum(torch.abs(recon)[:, 0:n // 2, 0:n // 2]) + torch.sum(torch.abs(recon)[:, n // 2:n, n // 2:n])

        # modularity_loss_inter = torch.sum(torch.matmul(torch.matmul(torch.transpose(S, 1, 2), recon), S))
        # modularity_loss_intra = torch.sum(torch.matmul(S, torch.transpose(S, 1, 2)) * I_prime * recon)
        loss = 1e3 * recon_loss + 1e-2*spatial_loss
        loss.backward()
        optimizer.step()

        if i == 0 and epoch % 20 == 0:
            print("Loss: {:.3f}, Spatial Loss: {:.3f}, L1 Loss 1: {:.3f}, L1 Loss 2: {:.3f}".format(loss.data, spatial_loss.data, l1_loss_1.data, l1_loss_2.data))
            # 1. Log scalar values (scalar summary)
            info = {'recon_loss': recon_loss.item(), 'recon_loss_x': x_loss, 'spatial_loss': spatial_loss}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch + 1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
                # logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # 3. Log training images (image summary)
            # info = {'images': gt[:10].cpu().numpy()}
            #
            # for tag, images in info.items():
            #     logger.image_summary(tag, images, epoch + 1)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, Args.MODEL_CP_PATH + "/model_chk_modloss" + str(epoch))
        i = i + 1
        loss_all += loss.item()
        N = N + data.num_graphs
    return loss_all / N

def test(epoch):
    model.eval()
    loss = 0
    N = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            data = data.to(Args.device)
            recon, x, _ = model(data)
            gt = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            loss += F.mse_loss(recon, gt, reduce=None)
            N = N + data.num_graphs
            gt = to_dense_adj(data.edge_index, batch=data.batch, edge_attr=data.edge_attr)

            if i == 0 and epoch % 50 == 0:
                plt.subplot(1, 2, 1)
                plt.imshow(sort_matrix(recon[0].detach().cpu().data.numpy())[0])
                plt.subplot(1, 2, 2)
                plt.imshow(sort_matrix(gt[0].detach().cpu().data.numpy())[0])
                plt.show()
                i = 1

    return loss / N


if __name__ == '__main__':
    loss_train = []
    loss_test = []
    # model.load_state_dict(torch.load(Args.MODEL_CP_PATH + "/model_chk_modloss" + str(700)))
    # model.eval()
    for i in range(0, 30000):
        loss_tr = train_baseline(i)
        loss_ts = test(i)
        loss_train.append(loss_tr)
        loss_test.append(loss_ts)
        if i % 50 == 0:
            print("Epoch: {}, Train Loss: {:0.5f}, Test Loss: {:0.5f}".format(i, loss_tr * 1e3, loss_ts * 1e3))

    logger.close()
    plt.plot(loss_train, 'r')
    plt.plot(loss_test, 'b')
    plt.show()