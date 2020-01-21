import numpy as np
from utils import sortDetriuxNodes
from args import Args
from matplotlib import pyplot as plt, colors


def draw_embeddings(z):
    from matplotlib import cm
    norm = colors.Normalize(vmin=0, vmax=148)
    viridis = cm.get_cmap('viridis', 12)
    for i in range(0, len(z)):
        x, y = z[i][0], z[i][1]
        plt.scatter(x, y, c=viridis(norm(i)))

for i in range(3):
    z = np.loadtxt('node_emb' + str(i) + '.txt')
    # idx = sortDetriuxNodes.sort_matrix()
    # z = z[idx,:]
    draw_embeddings(z)
    plt.show()