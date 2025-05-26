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
OUTPUT_PATH = "/samples/mueller_direct_b=0.05_points=100000_n=8000_4.csv"
potential_func = "mueller"

n = 8000
x_start = -1.5
x_end = 1.5
y_start = -0.5
y_end = 2
x_len = (x_end - x_start) / n
y_len = (y_end - y_start) / n
b = 1/20
points = 100000

x = np.linspace(x_start, x_end, n)
y = np.linspace(y_start, y_end, n)

x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()

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
    rand_offs = np.random.rand(int(integer), 2) - 0.5
    for rand_num in rand_offs:
        rand_num[0] = rand_num[0] * x_len
        rand_num[1] = rand_num[1] * y_len
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
