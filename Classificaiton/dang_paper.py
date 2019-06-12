class disc_network_clsf:
    def __init__(self, F, net, label):
        self.F = F
        self.A = net
        self.y = label