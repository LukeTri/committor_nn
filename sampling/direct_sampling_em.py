import csv
import math

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from neural_nets.dirichlet_form_nn_builtin_bc import NeuralNet
#from diffusion_map_mueller_nn import NeuralNet
from matplotlib import colors
import os

FILE_PATH = os.path.join(os.getcwd(), "data")
OUTPUT_PATH = "/samples/mueller_direct_em_b=0.05_points=100000_delta=0.01.csv"
SAMPLE_PATH = "/samples/mueller_delta_b=0.05_n=1000000_delta=0.01_2.csv"
potential_func = "mueller"

step_size = 1
b = 1/20
points = 100000
delta = 0.02
dim_delta = delta / np.sqrt(2)

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



plt.scatter(x, y)
plt.show()

plt.hist2d(x,y,bins=25)
plt.colorbar()
mp.plot_contours()
plt.show()

if potential_func == "mueller":
    probs = np.exp(-mp.MuellerPotentialNonVectorized(x, y) * b)
else:
    probs = np.exp(-fp.face_non_vectorized(x, y) * b)

print(probs)

rand_seeds = np.random.rand(len(x))
probs = probs / np.sum(probs) * points

samples = []

for i in tqdm(range(len(probs))):
    rem, integer = math.modf(probs[i])
    if rem > rand_seeds[i]:
        integer += 1
    rand_offs = (np.random.rand(int(integer), 2) - 0.5) * 2
    for rand_num in rand_offs:
        rand_num[0] = rand_num[0] * dim_delta
        rand_num[1] = rand_num[1] * dim_delta
    points = rand_offs + np.array([x[i], y[i]])
    for temp in points:
        samples.append(temp)

header = ['X', 'Y']

with open(FILE_PATH + OUTPUT_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(samples)
