from complex_torch import *
import math


def normalize(z):
    return Complex(r=torch.ones_like(z.abs), theta=z.angle)


class ComplexLayer:
    def __init__(self, in_nodes, out_nodes, device=torch.device("cpu")):
        self.weight = Complex(x=torch.empty(out_nodes, in_nodes).normal_(0.0, pow(1.0, -0.5)),
                              y=torch.empty(out_nodes, in_nodes).normal_(0.0, pow(1.0, -0.5))).to(device)

    def __call__(self, z):
        return mm(self.weight, z)


class ComplexNN:
    def __init__(self, nodes, device=torch.device("cpu")):
        self.device = device
        self.nodes = nodes
        self.nodes[0] += 1
        self.layers = []
        for i in range(len(self.nodes) - 1):
            self.layers.append(ComplexLayer(self.nodes[i], self.nodes[i + 1], device=self.device))
        self.zs = []

    def __call__(self, z):
        z = Complex(x=torch.cat([z.real.to("cpu"), torch.ones(1, 1)]),
                    y=torch.cat([z.imag.to("cpu"), torch.zeros(1, 1)])).to(self.device)
        self.zs = [z]
        for layer in self.layers:
            self.zs.append(normalize(layer(self.zs[-1])))
        return self.zs[-1]

    def train(self, label, n_category, n_part=1):
        ts = [Complex(r=torch.ones_like(self.zs[-1].abs), theta=torch.ones_like(self.zs[-1].angle) * (
                    (label + (i + 1) / (n_part + 1)) * (2 * math.pi / n_category))).to(self.device) for i in
              range(n_part)]
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                loss = ts[0] - self.zs[-1]
                for t in ts:
                    new_loss = (t - self.zs[-1])
                    if new_loss.abs.item() < loss.abs.item():
                        loss = new_loss
            else:
                loss /= self.nodes[i]
            dw = mm(loss, self.zs[i].conjugate().transpose()) / self.nodes[i]
            self.layers[i].weight += dw
