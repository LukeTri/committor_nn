import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.net_architectures.layers2_act1ReLU_act2tanh import NeuralNet as net1
from neural_nets.net_architectures.layers2_act1tanh_act2tanh import NeuralNet as net2
#from diffusion_map_mueller_nn import NeuralNet
from matplotlib import colors
import os

FILE_PATH = os.path.join(os.getcwd(), "data")
OUTPUT_PATH = "/samples/mueller_iterative_delta=0.01_2"
SAMPLE_PATH = "/samples/mueller_delta_b=0.05_n=1000000_delta=0.01_2.csv"
NN_PATH = "/nets/net_mueller_b=0.1_art_temp=0.05_n=1000000_step=5_hs=50_layers=2"

input_size = 2
hidden_size = 50
output_size = 1
num_classes = 1
learning_rate = 0.000001

step_size = 1


file = open(FILE_PATH + SAMPLE_PATH)
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

Npts, d = np.shape(arr)
x = arr[0:Npts:step_size, 0]
y = arr[0:Npts:step_size, 1]


model = net2(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


checkpoint = torch.load(FILE_PATH + NN_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

n = 600
potential_func = "mueller"
adjustment_power = 1
base_prob = 0.4

if potential_func == "face":
    ind = fp.face_non_vectorized(x,y) < 5
else:
    ind = mp.MuellerPotentialNonVectorized(x,y) < -15

x = x[ind]
y = y[ind]

print(x, y)

ind = np.zeros(len(x),dtype=bool)
rands = np.random.rand(len(x))
for i in tqdm(range(len(x))):
    tens = torch.tensor([x[i], y[i]])
    tens = tens.float()
    tens = tens.unsqueeze(0)
    val = model(tens)
    prob = 1 if torch.min(val, 1-val) > 0.05 else 0.5
    ind[i] = (prob > rands[i])

x, y = x[ind], y[ind]

data = np.vstack((x,y)).T
data = np.ndarray.tolist(data)

header = ['X','Y']
with open(FILE_PATH + OUTPUT_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
