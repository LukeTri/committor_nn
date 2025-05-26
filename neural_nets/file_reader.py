from neural_nets.Dataset import Dataset

import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
import torch

MUELLERMINA = torch.tensor([0.62347076, 0.02807048])
MUELLERMINB = torch.tensor([-0.55821361, 1.44174872])

xa=-3
ya=3
xb=0
yb=4.5
FACEMINA = torch.tensor([xa, ya])
FACEMINB = torch.tensor([xb, yb])

potential_func = "mueller"
art_temp = True
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


def get_data(FILE_PATH, FE_PATH, EM_PATH):
    file = open(FILE_PATH + FE_PATH)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()

    fem_vals = np.zeros(len(rows))

    fem_samples = np.zeros((len(rows), 2))

    for i in range(len(rows)):
        fem_samples[i] = np.array([rows[i][0], rows[i][1]], dtype=float)
        fem_vals[i] = float(rows[i][2])
    file = open(FILE_PATH + EM_PATH)
    csvreader = csv.reader(file)
    next(csvreader)
    rows = []
    rows2 = []
    cutoff = False
    for row in csvreader:
        if row[0] == "S":
            cutoff = True
        elif not cutoff:
            rows.append(row)
        else:
            rows2.append(row)
    file.close()

    arr = np.zeros((len(rows), len(rows[0])))
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            arr[i][j] = float(rows[i][j])

    if len(rows2) == 0:
        updaters = []
    else:
        updaters = np.zeros((len(rows2), len(rows2[0])))
        for i in range(len(rows2)):
            for j in range(len(rows2[i])):
                updaters[i][j] = float(rows2[i][j])

    Npts,d = np.shape(arr)
    X = arr[0:Npts:step_size, 0]
    Y = arr[0:Npts:step_size, 1]

    plt.scatter(X, Y)
    if potential_func == "face":
        fp.plot_contours()
    else:
        mp.plot_contours()
    plt.show()

    x = np.linspace(-1,1,50)
    y = np.linspace(-1,1,50)
    x_grid, y_grid = np.meshgrid(x,y)
    v_mpot_grid = np.zeros((len(x_grid), len(x_grid[0])))
    for i in range(len(x_grid)):
        for j in range(len(x_grid[i])):
            temp = torch.tensor(np.array([x_grid[i][j], y_grid[i][j]]))
            temp = temp.unsqueeze(0)
            if potential_func == "face":
                v_mpot_grid[i][j] = 0
            else:
                v_mpot_grid[i][j] = mp.get_updated_offset_gaussian(temp, updaters)

    # plt.contour(x_grid, y_grid, v_mpot_grid, np.linspace(np.amin(v_mpot_grid), np.amax(v_mpot_grid), 20))
    # plt.show()

    x_train = np.vstack((X, Y)).T
    y_train = np.zeros(len(x_train))

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    return x_train, y_train, fem_samples, fem_vals, updaters