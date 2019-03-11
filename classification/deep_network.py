import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


if __name__ == '__main__':
    dtype = torch.float
    device = torch.device("cpu")

    # Define Input and Output
    N = 141
    n_feature = 148 * 148
    n_classes = 3
    H = 1000

    net = NeuralNet(n_feature, H, n_classes)
    feature_set = torch.randn(N, n_feature, device=device, dtype=dtype)
    true_labels = torch.from_numpy(np.random.randint(0, 3, size=N))
    loss = nn.CrossEntropyLoss()

    for epoch in range(100):
        

    """
    

    W1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    W2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
    alpha = 1e-6

    for itr in range(500):
        # Forward pass
        pred = feature_set.mm(W1).clamp(min=0).mm(W2)
        loss = F.cross_entropy(pred.long(), output.long())
        print(itr, loss)
        loss.backward()
        with torch.no_grad():
            W1 -= alpha * W1.grad
            W2 -= alpha * W2.grad

            # Manually zero the gradients after updating weights
            W1.grad.zero_()
            W2.grad.zero_()
    """



