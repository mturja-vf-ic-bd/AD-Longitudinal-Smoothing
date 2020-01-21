from Longitudinal_Classifier.model import GUnetAE
from torch.utils.tensorboard import SummaryWriter
from random import seed
from torch_geometric.utils import to_dense_adj
from matplotlib import pyplot as plt
from utils import sortDetriuxNodes
from Longitudinal_Classifier.helper import *

def process_net(net, thr):
    net = normalize_net(net)
    net = net + np.dot(net.T, net)
    net = normalize_net(net)
    net[net > thr] = 1
    net[net < thr] = 0
    return net

id = '016_S_5057'
data = read_subject_data(subject_id=id)
G = []
for i in range(len(data["adjacency_matrix"])):
    net = data["adjacency_matrix"][i]
    net, _ = sortDetriuxNodes.sort_matrix(process_net(net, 0.01))

    # plt.imshow(net)
    # plt.show()
    g = convert_to_geom(np.eye(Args.n_nodes), net, data["dx_label"][i],
                        change_feat=False, add_label=False, add_dim=False)
    G.append(g)

# Prepare model
PATH = "model_unet.p"
writer = SummaryWriter("log")
seed(1)

model = GUnetAE(in_channel=148, hidden_channel=8, out_channel=8, depth=3).to(Args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=0.001)

epoch = 2000
emb = []

# model.load_state_dict(torch.load(PATH))
# Train the model
model.train()
for e in range(epoch):
    optimizer.zero_grad()
    loss = torch.zeros(1).to(Args.device)
    recon_lst = []
    orig_lst = []
    for i, g in enumerate(G):
        g_in = to_dense_adj(edge_index=g.edge_index)
        x, g_rec = model(g)
        # loss += torch.sum((g_in - g_rec)**2)
        loss += torch.sum(-g_in * torch.log(g_rec) - (1-g_in)*torch.log(1-g_rec))
        if e == epoch - 1:
            emb.append(x)

    loss.backward()
    optimizer.step()

    writer.add_scalar('Train/Loss', loss.item(), e)
    if e % 10 == 0:
        print('epoch: {}, loss: {}'.format(e, loss.item()))
        # for i in range(len(recon_lst)):
        #     orig = orig_lst[i]
        #     recon = recon_lst[i]
        #     writer.add_histogram('Train/orig_' + str(i), orig, e)
        #     writer.add_histogram('Train/recon_' + str(i), recon, e)

        for param in model.named_parameters():
            writer.add_histogram('Train/weight_hist_' + param[0], param[1], e)

    writer.flush()

# Save model
torch.save(model.state_dict(), PATH)
for i, x in enumerate(emb):
    np.savetxt('node_emb' + str(i) + '.txt', x.detach().cpu().numpy())


