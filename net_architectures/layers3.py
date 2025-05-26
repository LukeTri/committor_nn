import torch
from torch import nn

MUELLERMINA = torch.tensor([0.62347076, 0.02807048])
MUELLERMINB = torch.tensor([-0.55821361, 1.44174872])

xa = -3
ya = 3
xb = 0
yb = 4.5
FACEMINA = torch.tensor([xa, ya])
FACEMINB = torch.tensor([xb, yb])

potential_func = "mueller"

radius = 0.1


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.tanh1 = nn.Tanh()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, num_classes)

        self.sig3 = nn.Sigmoid()

    def forward(self, x):
        x.requires_grad = True
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)

        out = self.fc_out(out)
        out = self.sig3(out)
        out = out.squeeze()
        out = (1 - chi_A(x)) * ((1 - chi_B(x)) * out + chi_B(x))

        return out

def chi_A(x):
    m = torch.nn.Tanh()
    if potential_func == "face":
        return 0.5 - 0.5 * m(1000 * ((x - FACEMINA).pow(2).sum(1) - (radius + 0.02) ** 2))
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINA).pow(2).sum(1) - (radius + 0.02) ** 2))


def chi_B(x):
    m = torch.nn.Tanh()
    if potential_func == "face":
        return 0.5 - 0.5 * m(1000 * ((x - FACEMINB).pow(2).sum(1) - (radius + 0.02) ** 2))
    return 0.5 - 0.5 * m(1000 * ((x - MUELLERMINB).pow(2).sum(1) - (radius + 0.02) ** 2))