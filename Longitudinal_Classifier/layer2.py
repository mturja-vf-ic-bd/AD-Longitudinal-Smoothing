import torch.nn as nn

class SimpleLinear(nn.Module):
    def __init__(self, dense_dim):
        super(SimpleLinear, self).__init__()
        self.dns_lr = [
            nn.Sequential(nn.Linear(dense_dim[i - 1], dense_dim[i]), nn.ReLU()) if i < len(dense_dim) - 1 else
            nn.Linear(dense_dim[i - 1], dense_dim[i])
            for i in range(1, len(dense_dim))
        ]

        for i, l in enumerate(self.dns_lr):
            self.add_module('Dense_{}'.format(i), l)

    def forward(self, g, batch):
        x = g.x
        x = x.view(batch, -1)
        for l in self.dns_lr:
            x = l(x)
        # x = torch.mean(x, dim=1)
        return x