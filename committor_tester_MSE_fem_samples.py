import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# layers = 2, act1 = tanh, act2 = tanh
from neural_nets.net_architectures.layers2_act1ReLU_act2tanh import NeuralNet as net1
from neural_nets.net_architectures.layers2_act1tanh_act2tanh import NeuralNet as net2
from neural_nets.net_architectures.layers3 import NeuralNet as net3

from matplotlib import colors
from scipy import interpolate
import os

nn_type = "1"

FILE_PATH = os.path.join(os.getcwd(), "data")
NN_PATH = "/nets/direct_delta=0.02_3"
FE_PATH = "/fe_results/fe_mueller_b=0.1.csv"

potential_func = "mueller"

input_size = 2
hidden_size = 20
output_size = 1
num_classes = 1
learning_rate = 0.000001

if nn_type == "1":
    model = net1(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
else:
    model = net2(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load(FILE_PATH + NN_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()


file = open(FILE_PATH + FE_PATH)
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
        rows.append(row)
file.close()

fem_vals = np.zeros(len(rows))

fem_samples = np.zeros((len(rows), 2))

for i in tqdm(range(len(rows))):
    fem_samples[i] = np.array([rows[i][0], rows[i][1]], dtype=float)
    fem_vals[i] = float(rows[i][2])

ind = np.zeros(len(fem_samples), dtype=bool)

for i in range(len(ind)):
    if -1.5 < fem_samples[i][0] < 1.5 and -0.5 < fem_samples[i][1] < 2:
        ind[i] = True

plots = fem_samples[ind]
plotv = fem_vals[ind]

plt.scatter(plots[:,0], plots[:,1],c = plotv, cmap="plasma")
mp.plot_contours()
plt.show()

MSE = 0

model_vals = model(torch.tensor(fem_samples, dtype=torch.float)).detach().numpy()

if potential_func == "face":
    ind = fp.face_vectorized(fem_samples) < 5
else:
    ind = mp.MuellerPotentialVectorized(fem_samples) < -36

model_vals = model_vals[ind]
fem_vals = fem_vals[ind]

x = (fem_vals - model_vals) ** 2

print(np.mean(x))
print(np.sum(x))

RMSE = np.sqrt(np.sum((fem_vals - model_vals) ** 2 / len(fem_vals)))

print(RMSE)