import csv

from euler_maruyama import euler_maruyama_white_noise_mueller as mp
from euler_maruyama import euler_maruyama_white_noise_face as fp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

FILE_PATH = os.path.join(os.getcwd(), "data")
SAMPLE_PATH = "/samples/mueller_standard_b=0.05_n=1000000.csv"
OUTPUT_PATH = "/samples/mueller_delta_b=0.05_n=1000000_delta=0.005.csv"

step_size = 1
delta = 0.005

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

samples = np.zeros((len(rows), len(rows[0])))
for i in range(len(rows)):
    for j in range(len(rows[i])):
        samples[i][j] = float(rows[i][j])


samples = samples[::step_size]

print(samples)

plt.scatter(samples[:,0], samples[:,1])
plt.show()

np.random.shuffle(samples)

del_rarefied = np.zeros(len(samples), dtype=bool)

del_rarefied[0] = True
for i in tqdm(range(1,len(samples))):
    valids = samples[del_rarefied]
    transpose = np.array([[samples[i][0], samples[i][1]]])
    if np.min(np.linalg.norm(valids - transpose, axis=1)) > delta:
        del_rarefied[i] = True

samples = samples[del_rarefied]

sam_len = len(samples)
del_rarefied = np.ones(len(samples), dtype=bool)
for i in tqdm(range(1,len(samples))):
    transpose = np.array([[samples[i][0], samples[i][1]]])
    temp = samples[np.arange(sam_len) != i]
    if np.min(np.linalg.norm(temp - transpose, axis=1)) > 2 * delta:
        del_rarefied[i] = False

samples = samples[del_rarefied]

plt.scatter(samples[:,0], samples[:,1])
mp.plot_contours()
plt.show()

header = ['X', 'Y']

with open(FILE_PATH + OUTPUT_PATH, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(samples)
