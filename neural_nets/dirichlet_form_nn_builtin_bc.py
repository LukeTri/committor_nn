from neural_nets.Dataset import Dataset

import torch.nn as nn
import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
import torch
import neural_nets.file_reader as fr
import os

FILE_PATH = os.path.join(os.getcwd(), "data")
NN_PATH = "/nets/test"
EM_PATH = "/samples/mueller_standard_b=0.1_n=200000.csv"
FE_PATH = "/fe_results/fe_mueller_b=0.1.csv"

MUELLERMINA = torch.tensor([0.62347076, 0.02807048])
MUELLERMINB = torch.tensor([-0.55821361, 1.44174872])

xa=-3
ya=3
xb=0
yb=4.5
FACEMINA = torch.tensor([xa, ya])
FACEMINB = torch.tensor([xb, yb])

potential_func = "mueller"
art_temp = False
metadyanamics = False


radius = 0.1
epsilon = 0.05

step_size = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 2
hidden_size = 20
output_size = 1
num_epochs = 1400
batch_size = 50000
learning_rate = 0.1
num_classes = 1
momentum = 0.90

# Sampling parameters
b = 1 / 10
b_prime = 1 / 20



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.sig3 = nn.Sigmoid()

    def forward(self, x):
        x.requires_grad = True
        self.fc1.weight.requires_grad = True
        self.fc3.weight.requires_grad = True

        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)
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


def main():
    x_train, y_train, fem_samples, fem_vals, updaters = fr.get_data(FILE_PATH, FE_PATH, EM_PATH)

    training_set = Dataset(x_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9992)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(training_generator)
    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(training_generator):
            optimizer.zero_grad()
            outputs = model(samples)
            outputs = torch.autograd.grad(outputs, samples, allow_unused=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
            outputs = outputs.pow(2).sum(1)
            if art_temp:
                if potential_func == "face":
                    outputs = outputs * torch.exp(-(b - b_prime) * torch.tensor(fp.face_non_vectorized(
                        samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy())))
                else:
                    outputs = outputs * torch.exp(-(b - b_prime) * torch.tensor(mp.MuellerPotentialNonVectorized(
                        samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy())))
            elif metadyanamics:
                if potential_func == "face":
                    pass
                else:
                    outputs = outputs * torch.exp(b * (mp.get_updated_offset_gaussian(samples, updaters)))

            loss = torch.sqrt(outputs.sum())
            loss.backward()

            optimizer.step()
            # Backward and optimize

            if (epoch + 1) % 5 == 0 and i % 3 == 0:
                model_vals = model(torch.tensor(fem_samples, dtype=torch.float)).detach().numpy()
                ind = mp.MuellerPotentialVectorized(fem_samples) < -36
                model_vals = model_vals[ind]
                temp_fem = fem_vals[ind]
                print('FEM Loss: {:.4f}'.format(np.sqrt(np.sum((temp_fem - model_vals) ** 2 / len(temp_fem)))))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item() / batch_size))
        scheduler.step()
        # if (epoch + 1) % 5 == 0:
            # print(scheduler.state_dict()['_last_lr'])

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, FILE_PATH + NN_PATH)
    # data/mueller_model_standard_b=0.033_n=100000_hs=20_ep=20_lr=0.000001_bs=10000.pth

    if potential_func == "face":
        m = np.arange(-5, 4, 0.1)
        p = np.arange(-3, 7, 0.1)

    else:
        p = np.arange(-.5, 2, 0.05)
        m = np.arange(-1.5, 1.5, 0.05)

    X, Y = np.meshgrid(m, p)
    Z = np.zeros((len(p), len(m)))
    for i in range(len(m)):
        for j in range(len(p)):
            tens = torch.tensor([[X[j][i], Y[j][i]]])
            tens = tens.float()
            Z[j][i] = model(tens)

    plt.pcolormesh(X, Y, Z, shading='gouraud')
    plt.colorbar()
    if potential_func == "face":
        fp.plot_contours()
    else:
        mp.plot_contours()
    plt.ylabel("Y")
    plt.xlabel("X")

    plt.show()


if __name__ == "__main__":
    main()
